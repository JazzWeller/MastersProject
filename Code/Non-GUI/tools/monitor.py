#!/usr/bin/env python3
"""One screen per run (Agent Training Plan, Milestone M10): the *derived*
numbers, not the raw ones.

    python -m tools.monitor                 # every run under $KEYFORGE_DATA/runs
    python -m tools.monitor tier2-wt-s0     # one run, in detail
    python -m tools.monitor --watch 60 ID   # refresh every 60 s

The three numbers that actually drive decisions, with the plan's health
bands:

| Number | Healthy | Unhealthy means |
|---|---|---|
| Games/hour | within 60% of projection (gate G5) | re-plan the tier before continuing |
| Actor:learner ratio | ~0.5-2 | >>2: the learner can't keep up. <<1 with the learner idle: actors starved |
| Promotions | keep coming (gate G6: none in the last 3 gates) | converged, stalled, or broken |

Everything else in metrics.jsonl is for diagnosis once one of those goes
wrong. Projections come from `--projected-games-per-hour` (default: the
plan's own estimate, 1,500) and are replaced by measurement as soon as a
run has one.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from collections import defaultdict
from typing import Dict, List, Optional

from agent.telemetry import Run, iter_runs, read_jsonl

PLAN_GAMES_PER_HOUR = 1500.0
STALL_GATES = 3  # G6: this many gates in a row without a promotion


def _gpu() -> Optional[str]:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        util, used, total = [x.strip() for x in out.splitlines()[0].split(",")]
        return f"GPU {util}% busy, {used}/{total} MiB"
    except Exception:  # noqa: BLE001 -- the monitor must never crash on a missing tool
        return None


def summarize(run: Run, projected_gph: float = PLAN_GAMES_PER_HOUR, window_minutes: float = 30.0) -> Dict[str, object]:
    metrics = run.metric_records()
    journal = run.journal_entries()
    now = time.time()
    recent = [m for m in metrics if now - m["t"] <= window_minutes * 60] or metrics[-50:]
    by_source: Dict[str, List[dict]] = defaultdict(list)
    for m in recent:
        by_source[m["source"]].append(m)

    def total_rate(prefix: str, key: str) -> float:
        # Sum of each source's average rate over the window.
        out = 0.0
        for src, recs in by_source.items():
            if src.startswith(prefix):
                dt = sum(r["interval"] for r in recs)
                n = sum(r["counts"].get(key, 0) for r in recs)
                out += n / dt if dt else 0.0
        return out

    games_per_hour = 3600.0 * total_rate("actor", "games")
    finished = any(j["event"] == "run_finished" for j in journal)
    actor_recs = [m for m in metrics if m["source"].startswith("actor")]
    if finished and actor_recs:
        # A finished run's rate is the whole run's, not its idle tail's
        # (Tier 2 round 3 read 850 games/h, after the actors had stopped,
        # against ~5,000 while they played).
        span = max(m["t"] for m in actor_recs) - min(m["t"] - m["interval"] for m in actor_recs)
        games_per_hour = 3600.0 * sum(m["counts"].get("games", 0) for m in actor_recs) / span if span > 0 else 0.0
    produced = total_rate("learner", "positions_produced")
    # Positions produced over positions *budgeted* by the learner's steps
    # (one step per `positions_per_step` new positions) -- not over
    # positions consumed, which counts every reuse of a position (~batch /
    # positions_per_step of them by design). ~1: the learner keeps pace;
    # >> 1: it can't. It can't run ahead (it waits for new data), so actor
    # starvation shows as ~1 with a low steps/second, not as < 1.
    from agent.config import positions_per_step

    pps = positions_per_step(dict(run.config.get("selfplay", {}), batch=run.config.get("selfplay", {}).get("batch", 1024)))
    steps_rate = total_rate("learner", "gradient_steps")
    # (no data arriving -- the actors are done, or not started -- says nothing either way)
    ratio = produced / (steps_rate * pps) if steps_rate and produced else None
    learner = [m for m in metrics if m["source"] == "learner"]
    # The newest record that carries each kind of reading: an idle
    # interval's record has no losses (and older learners wrote no gauges
    # while idle), and must not read as "step 0, no losses".
    last = next((m for m in reversed(learner) if "step" in m["gauges"]), None)
    last_losses = next((m for m in reversed(learner) if any(k.startswith("loss_") for k in m["means"])), None)
    games_done = int(last["gauges"].get("games", 0)) if last else 0
    budget = int(run.config.get("selfplay", {}).get("games", 0))
    remaining = max(0, budget - games_done)
    eta_hours = remaining / games_per_hour if games_per_hour else None

    gates = [j for j in journal if j["event"] == "gate"]
    promotions = [j for j in journal if j["event"] == "promoted"]
    slope = None
    if len(gates) >= 2:
        # Gate scores against the moving best: sustained > 0.5 means the
        # network keeps improving; the slope over games is the G6 read.
        xs = [g["game_index"] for g in gates[-6:]]
        ys = [g["score"] for g in gates[-6:]]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        den = sum((x - mx) ** 2 for x in xs)
        slope = (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den * 10_000) if den else None

    gate_every = run.config.get("selfplay", {}).get("gate_every_steps")
    step = int(last["gauges"].get("step", 0)) if last else 0
    steps_per_sec = total_rate("learner", "gradient_steps")
    to_gate_hours = None
    if gate_every and steps_per_sec and not finished:
        to_gate_hours = ((gate_every - step % gate_every) / steps_per_sec) / 3600.0

    flags = []
    interval = float(run.config.get("telemetry", {}).get("interval_seconds", 60))
    if learner and not finished and now - learner[-1]["t"] > 3 * interval:
        flags.append(f"learner silent for {(now - learner[-1]['t']) / 60:.0f} min -- blocked or dead (actors may still be writing shards)")
    if games_per_hour and games_per_hour < 0.6 * projected_gph:
        flags.append(f"G5: games/hour {games_per_hour:.0f} is under 60% of the projected {projected_gph:.0f} -- re-plan the tier")
    if ratio is not None and ratio > 2:
        flags.append(f"actor:learner {ratio:.2f} >> 2 -- the learner can't keep up")
    if ratio is not None and ratio < 0.5:
        flags.append(f"actor:learner {ratio:.2f} << 0.5 -- inconsistent (the learner shouldn't outrun its data); check positions_per_step")
    # G6 reads promotions, not the score slope: every gate plays the newest
    # best, so scores fall as it improves even while promotions continue.
    recent_gates = gates[-STALL_GATES:]
    if (games_done >= 25_000 and len(recent_gates) == STALL_GATES
            and not any(g.get("verdict") == "H1" for g in recent_gates)):
        flags.append(f"G6: no promotion in the last {STALL_GATES} gates -- bank the promoted checkpoint and stop; "
                     "spend the time on the matrix")
    last_val = next((m for m in reversed(learner) if "val_loss_value" in m["means"]), None)
    validation = {k[4:]: round(v, 4) for k, v in last_val["means"].items() if k.startswith("val_loss_")} if last_val else {}
    train_v = last_losses["means"].get("loss_value") if last_losses else None
    if train_v is not None and "loss_value" in validation and validation["loss_value"] > 1.5 * train_v + 0.05:
        flags.append(f"value loss {train_v:.3f} in training vs {validation['loss_value']:.3f} on held-out games -- "
                     "memorizing (train each outcome once: selfplay.value_reuse 1, or sample_reuse 1)")
    ent = last_losses["means"].get("policy_entropy") if last_losses else None
    if ent is not None and ent < 0.2:
        flags.append(f"policy entropy {ent:.3f} is collapsing -- raise root noise / check the temperature schedule")
    return {
        "run": run.run_id,
        "config_hash": run.config_hash[:12],
        "tier": run.config.get("tier"),
        "games_done": games_done,
        "budget": budget,
        "games_per_hour": round(games_per_hour, 1),
        "projected_games_per_hour": projected_gph,
        "eta_hours": round(eta_hours, 2) if eta_hours is not None else None,
        "actor_learner_ratio": round(ratio, 3) if ratio is not None else None,
        "gradient_steps": step,
        "steps_per_second": round(steps_per_sec, 3),
        "hours_to_next_gate": round(to_gate_hours, 3) if to_gate_hours is not None else None,
        "gates": len(gates),
        "promotions": max(0, len(promotions) - 1),
        "last_gate_score": gates[-1]["score"] if gates else None,
        "gate_score_slope_per_10k_games": round(slope, 4) if slope is not None else None,
        "gate_running": bool(last["gauges"].get("gate_running")) if last else False,
        "losses": {k: round(v, 4) for k, v in (last_losses["means"].items() if last_losses else ()) if k.startswith("loss_")},
        "validation_losses": validation,
        "policy_entropy": round(ent, 4) if ent is not None else None,
        "interventions": [j for j in journal if j["event"] not in ("gate", "promoted", "run_created")][-5:],
        "flags": flags,
    }


def render(s: Dict[str, object]) -> str:
    lines = [
        f"== {s['run']}  (tier {s['tier']}, config {s['config_hash']})",
        f"   games {s['games_done']:,}/{s['budget']:,}   {s['games_per_hour']} games/h (projected {s['projected_games_per_hour']:.0f})"
        f"   ETA {s['eta_hours']} h",
        f"   actor:learner {s['actor_learner_ratio']}   steps {s['gradient_steps']:,} ({s['steps_per_second']}/s)"
        f"   next gate in {s['hours_to_next_gate']} h",
        f"   gates {s['gates']}{' (one running)' if s['gate_running'] else ''}, promotions {s['promotions']},"
        f" last score {s['last_gate_score']}, slope/10k games {s['gate_score_slope_per_10k_games']}",
        f"   losses {s['losses']}   entropy {s['policy_entropy']}",
    ]
    if s.get("validation_losses"):
        lines.append(f"   held-out {s['validation_losses']}")
    for f in s["flags"]:
        lines.append(f"   !! {f}")
    for j in s["interventions"]:
        lines.append(f"   journal @{j['game_index']}: {j['event']} " + json.dumps({k: v for k, v in j.items() if k not in ('t', 'event', 'game_index', 'config_hash')})[:140])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*")
    parser.add_argument("--watch", type=float, default=None)
    parser.add_argument("--projected-games-per-hour", type=float, default=PLAN_GAMES_PER_HOUR)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    while True:
        ids = args.runs or list(iter_runs())
        out = []
        gpu = _gpu()
        if gpu and not args.json:
            out.append(gpu)
        for rid in ids:
            try:
                s = summarize(Run.open(rid), args.projected_games_per_hour)
            except FileNotFoundError:
                continue
            out.append(json.dumps(s, default=str) if args.json else render(s))
        print("\n".join(out) if out else "no runs yet", flush=True)
        if not args.watch:
            return
        time.sleep(args.watch)


if __name__ == "__main__":
    main()

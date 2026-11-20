#!/usr/bin/env python3
"""The paper's supporting numbers and figures, from what is already under `results/` -- no games are played.

    python -m tools.paper_stats [--results results] [--out results/paper]

Writes to `<out>/`:
- `paper_stats.md`: everything below as readable tables;
- `compute.csv`: per run -- games, wall-clock hours (journal: created to finished, pauses included), learner hours
  (the learner's metric intervals summed: time it was actually up), games per learner hour, gates, promotions, the
  hours spent gating, and the host the learner ran on;
- `gates.csv`: every gate -- step, games so far, its score against the incumbent, promoted or not;
- `losses.csv`: the learner's mean training losses over time, plus each run's held-out `final_validation`;
- `elo_curve.csv`: each tested checkpoint's Elo (HeuristicBot = 0) from the `curve-<run>` ladders (run.py
  eval-curves), placed at the number of self-play games it had seen;
- `hyperparameters.md`: the settings that define the main runs, side by side;
- `fig_search.pdf/.png`, `fig_training.pdf/.png`, `fig_losses.pdf/.png` when matplotlib is installed.

Seed spread and the bidding confidence intervals are in `paper_stats.md`. Everything is read from exported files, so
rerun it after each `run.py export-*`.
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import math
import os
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOSSES = ("loss_policy", "loss_value", "loss_multi", "policy_entropy")


# --------------------------------------------------------------------------- runs
def load_runs(results: str) -> Dict[str, dict]:
    """Run id -> {dir, group, journal, config}: every exported self-play run (a directory with a journal)."""
    runs = {}
    for path in sorted(glob.glob(os.path.join(results, "*", "*", "journal.jsonl"))):
        d = os.path.dirname(path)
        rid = os.path.basename(d)
        with open(path, encoding="utf-8") as f:
            journal = [json.loads(line) for line in f if line.strip()]
        cfg = {}
        if os.path.exists(os.path.join(d, "config.json")):
            with open(os.path.join(d, "config.json"), encoding="utf-8") as f:
                cfg = json.load(f)
        runs[rid] = {"dir": d, "group": os.path.basename(os.path.dirname(d)), "journal": journal, "config": cfg}
    return runs


def learner_metrics(run: dict) -> List[dict]:
    p = os.path.join(run["dir"], "learner_metrics.jsonl.gz")
    if not os.path.exists(p):
        return []
    out = []
    with gzip.open(p, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def compute_rows(runs: Dict[str, dict]) -> List[dict]:
    rows = []
    for rid, r in runs.items():
        j = r["journal"]
        if not j:
            continue
        fin = [e for e in j if e["event"] == "run_finished"]
        games = fin[-1]["game_index"] if fin else max(e.get("game_index", 0) for e in j)
        wall = ((fin[-1]["t"] if fin else j[-1]["t"]) - j[0]["t"]) / 3600
        m = learner_metrics(r)
        up = sum(x.get("interval", 0) for x in m) / 3600
        gates = [e for e in j if e["event"] == "gate"]
        rows.append({"run": rid, "group": r["group"], "finished": bool(fin), "games": games,
                     "wall_hours": round(wall, 2), "learner_hours": round(up, 2) if m else None,
                     "games_per_learner_hour": round(games / up) if m and up else None,
                     "gates": len(gates),
                     "promotions": sum(1 for e in j if e["event"] == "promoted" and e.get("reason") != "warm start"),
                     "gate_hours": round(sum(e.get("seconds", 0) for e in gates) / 3600, 2),
                     "host": next((x.get("host") for x in m if x.get("host")), None),
                     "steps": fin[-1]["step"] if fin else None})
    return rows


def gate_rows(runs: Dict[str, dict]) -> List[dict]:
    rows = []
    for rid, r in runs.items():
        promoted = {e["checkpoint"] for e in r["journal"] if e["event"] == "promoted"}
        for e in r["journal"]:
            if e["event"] == "gate":
                rows.append({"run": rid, "gate": e["number"], "step": e.get("step"), "games": e.get("game_index"),
                             "score": e.get("score"), "promoted": e.get("candidate") in promoted,
                             "hours": round(e.get("seconds", 0) / 3600, 3)})
    return rows


def loss_rows(runs: Dict[str, dict]) -> List[dict]:
    rows = []
    for rid, r in runs.items():
        for x in learner_metrics(r):
            means, g = x.get("means") or {}, x.get("gauges") or {}
            if not means:
                continue
            rows.append({"run": rid, "kind": "train", "step": g.get("step"), "games": g.get("games"),
                         **{k: round(means[k], 5) for k in LOSSES if k in means}})
        for e in r["journal"]:
            if e["event"] == "final_validation":
                rows.append({"run": rid, "kind": "held_out_final", "step": e.get("step"), "games": e.get("game_index"),
                             "loss_policy": e.get("val_loss_policy"), "loss_value": e.get("val_loss_value"),
                             "loss_multi": e.get("val_loss_multi")})
    return rows


# --------------------------------------------------------------------------- ladders
def _ladders(results: str):
    from tools.report_tables import Ladders, load_ladders

    return Ladders(load_ladders(results))


def elo_curve_rows(L, runs: Dict[str, dict]) -> List[dict]:
    """Each `curve-<run>` ladder's checkpoints, with the self-play games each had seen."""
    rows = []
    for name in sorted(L.l):
        if not name.startswith("curve-"):
            continue
        rid = name[len("curve-"):]
        j = runs.get(rid, {}).get("journal", [])
        gate_games = {e["number"]: (e.get("game_index"), e.get("step")) for e in j if e["event"] == "gate"}
        fin = [e for e in j if e["event"] == "run_finished"]
        for r in L.l[name]["ratings"]:
            p = r["player"]
            at = p.split("#")[0].partition("@")[2]
            if at.startswith("g") and at[1:].isdigit():
                games, step = gate_games.get(int(at[1:]), (None, None))
            elif at == "p0":
                games, step = 0, 0
            elif at == "final" and fin:
                games, step = fin[-1]["game_index"], fin[-1]["step"]
            else:
                continue
            e = L.elo(name, p)
            if e is None:
                continue
            rows.append({"run": rid, "checkpoint": at, "games": games, "step": step, "elo": round(e[0], 1),
                         "se": round(e[1], 1)})
    rows.sort(key=lambda x: (x["run"], x["games"] if x["games"] is not None else -1))
    return rows


SEED_GROUPS = [("Real-deck agent", ["v1:real-v1", "v1:real-v1-s1", "v1:real-v1-s2"], ["anchors-real", "baseline"]),
               ("Specialist, within-turn training", ["v1:m9-wt-1", "v1:m9-wt-2"], ["m9-real", "baseline"]),
               ("Specialist, full-game training", ["v1:m9-fg-1", "v1:m9-fg-2"], ["m9-real", "baseline"])]


def seed_rows(L) -> List[dict]:
    rows = []
    for label, players, ladders in SEED_GROUPS:
        for lad in ladders:
            vals = [(p, L.elo(lad, p)) for p in players]
            vals = [(p, v) for p, v in vals if v is not None]
            if len(vals) < 2:
                continue
            xs = [v[0] for _p, v in vals]
            mean = sum(xs) / len(xs)
            sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1))
            rows.append({"agent": label, "ladder": lad, "seeds": len(xs), "elos": [round(x) for x in xs],
                         "mean": round(mean, 1), "sd": round(sd, 1), "range": round(max(xs) - min(xs), 1),
                         "fit_se": round(max(v[1] for _p, v in vals), 1)})
    return rows


# --------------------------------------------------------------------------- bidding
def wilson(k: float, n: int, z: float = 1.96):
    if not n:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 3), round(c + h, 3)


def bidding_rows(results: str) -> List[dict]:
    p = os.path.join(results, "sas", "sas_report.json")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        sas = json.load(f)
    rows = []
    for b in sas["bands"]:
        if b["format"] != "adaptive":
            continue
        n = b.get("results") or 0
        g3 = round((b.get("game3_rate") or 0) * n)
        k = round((b.get("bid_winner_game3_rate") or 0) * g3)
        rows.append({"opponent": "itself" if b["a"] == b["b"] else b["b"], "gap": b["gap"], "matches": n,
                     "game3": g3, "game3_ci": wilson(g3, n), "mean_bid": b.get("mean_bid"),
                     "bid_winner_wins_game3": round(k / g3, 3) if g3 else None, "ci": wilson(k, g3),
                     "match_score": b.get("a_score"), "match_score_ci": wilson(round((b.get("a_score") or 0) * n), n)})
    return rows


# --------------------------------------------------------------------------- hyperparameters
KEYS = [("search", k) for k in ("regime", "simulations", "c_puct", "dirichlet_alpha", "dirichlet_eps", "determinization",
                                 "resample", "leaf", "rollout", "temperature_high", "temperature_low", "temperature_moves",
                                 "leaves_in_flight", "virtual_loss")] + \
       [("selfplay", k) for k in ("games", "workers", "games_per_worker", "batch", "lr", "lr_min", "weight_decay",
                                   "grad_clip", "buffer_positions", "sample_reuse", "policy_reuse", "value_reuse",
                                   "playout_cap_full", "playout_cap_small", "playout_cap_full_fraction",
                                   "gate_every_steps", "gate_games", "gate_sprt", "gate_fallback_z", "precision",
                                   "resign_consecutive", "loss_weights")] + \
       [("network", k) for k in ("d_model", "layers", "heads", "ff", "card_embed", "policy_head", "multi_select")] + \
       [("pool", k) for k in ("deck_source", "deck_list", "split", "decks", "format", "max_turns")] + \
       [("dmc", k) for k in ("epsilon_start", "epsilon_end", "epsilon_decay_games")] + [("seed", None)]
HYPER_RUNS = ["m9-wt-1", "m9-fg-1", "m11-v1", "real-v1", "real-v1-s1", "dmc-1", "dmc-long"]


def hyper_table(runs: Dict[str, dict]) -> str:
    cols = [r for r in HYPER_RUNS if r in runs and runs[r]["config"]]
    if not cols:
        return "(no run configs found)\n"
    out = ["| setting | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for sec, key in KEYS:
        vals = []
        for r in cols:
            c = runs[r]["config"]
            v = c.get(sec) if key is None else (c.get(sec) or {}).get(key)
            vals.append("" if v is None else json.dumps(v) if isinstance(v, (list, dict)) else str(v))
        if any(vals):
            out.append(f"| {sec}{'.' + key if key else ''} | " + " | ".join(vals) + " |")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- figures
def figures(out: str, L, curve: List[dict], gates: List[dict], losses: List[dict], results: str) -> List[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    from tools.report_names import name
    from tools.report_tables import T2_NETS, T2_SETTINGS

    made = []

    def save(fig, stem):
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(out, f"{stem}.{ext}"), bbox_inches="tight", dpi=200)
        plt.close(fig)
        made.append(stem)

    plt.rcParams.update({"font.size": 8, "font.family": "serif"})
    # Search at play time: Elo against simulations, one line per network; the network alone as a dashed level.
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    sims = {"#sims=10": 10, "#sims=30": 30, "": 100, "#sims=300": 300, "#sims=1000": 1000}
    any_line = False
    for lad, base, deck in T2_NETS:
        pts = [(sims[o], L.elo(lad, base + o)) for o, _lab in T2_SETTINGS if o in sims]
        pts = [(x, e) for x, e in pts if e is not None]
        if not pts:
            continue
        any_line = True
        line = ax.errorbar([x for x, _ in pts], [e[0] for _, e in pts], yerr=[1.96 * e[1] for _, e in pts], marker="o",
                           ms=3, capsize=2, label=name(base))
        alone = L.elo(lad, base + "#search=none")
        if alone is not None:
            ax.axhline(alone[0], ls="--", lw=0.8, color=line[0].get_color())
    if any_line:
        ax.set_xscale("log")
        ax.set_xlabel("Simulations per decision")
        ax.set_ylabel("Elo (HeuristicBot = 0)")
        ax.legend(fontsize=6, frameon=False)
        ax.grid(alpha=0.3)
        save(fig, "fig_search")
    else:
        plt.close(fig)
    # Training: Elo against self-play games where the curve ladders exist, else each gate's score vs the incumbent.
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    if curve:
        for rid in sorted({r["run"] for r in curve}):
            pts = [r for r in curve if r["run"] == rid and r["games"] is not None]
            ax.errorbar([p["games"] / 1000 for p in pts], [p["elo"] for p in pts], yerr=[1.96 * p["se"] for p in pts],
                        marker="o", ms=3, capsize=2, label=name("v1:" + rid))
        ax.set_ylabel("Elo (HeuristicBot = 0)")
    else:
        for rid in ("m9-wt-1", "m11-v1", "real-v1"):
            pts = [g for g in gates if g["run"] == rid and g["games"] is not None]
            if pts:
                ax.plot([p["games"] / 1000 for p in pts], [p["score"] for p in pts], marker="o", ms=3, label=name("v1:" + rid))
        ax.axhline(0.5, ls="--", lw=0.8, color="grey")
        ax.set_ylabel("Gate score vs. incumbent")
    ax.set_xlabel("Self-play games (thousands)")
    ax.legend(fontsize=6, frameon=False)
    ax.grid(alpha=0.3)
    save(fig, "fig_training")
    # Losses: training value and policy loss against games, with each run's held-out final values as markers.
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.4))
    for rid in ("m9-wt-1", "m11-v1", "real-v1"):
        tr = [r for r in losses if r["run"] == rid and r["kind"] == "train" and r.get("games") is not None]
        ho = [r for r in losses if r["run"] == rid and r["kind"] == "held_out_final"]
        for ax, key in zip(axes, ("loss_value", "loss_policy")):
            pts = [(r["games"] / 1000, r[key]) for r in tr if r.get(key) is not None][1:]  # [1:]: the first minute's spike
            if pts:
                k = 25  # minutes: a moving average, the per-minute means being noisy
                ys = [sum(y for _x, y in pts[max(0, i - k + 1):i + 1]) / len(pts[max(0, i - k + 1):i + 1])
                      for i in range(len(pts))]
                line = ax.plot([x for x, _y in pts], ys, lw=0.9, label=name("v1:" + rid))
                for h in ho:
                    ax.plot([h["games"] / 1000], [h[key]], marker="x", color=line[0].get_color())
    # Lines: training loss, a 25-minute moving mean; crosses: the run's held-out loss at its end.
    for ax, t in zip(axes, ("Value loss", "Policy loss")):
        ax.set_xlabel("Self-play games (thousands)")
        ax.set_ylabel(t)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=6, frameon=False)
    save(fig, "fig_losses")
    return made


# --------------------------------------------------------------------------- output
def write_csv(path: str, rows: List[dict]) -> None:
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def md_table(rows: List[dict], keys: Optional[List[str]] = None) -> str:
    if not rows:
        return "(none yet)\n"
    keys = keys or list(rows[0])
    out = ["| " + " | ".join(keys) + " |", "|---" * len(keys) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if r.get(k) is None else str(r.get(k)) for k in keys) + " |")
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=os.path.join(HERE, "results"))
    parser.add_argument("--out", default=os.path.join(HERE, "results", "paper"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    runs = load_runs(args.results)
    L = _ladders(args.results)
    comp, gates, losses = compute_rows(runs), gate_rows(runs), loss_rows(runs)
    curve, seeds, bids = elo_curve_rows(L, runs), seed_rows(L), bidding_rows(args.results)
    write_csv(os.path.join(args.out, "compute.csv"), comp)
    write_csv(os.path.join(args.out, "gates.csv"), gates)
    write_csv(os.path.join(args.out, "losses.csv"), losses)
    write_csv(os.path.join(args.out, "elo_curve.csv"), curve)
    hyper = hyper_table(runs)
    with open(os.path.join(args.out, "hyperparameters.md"), "w", encoding="utf-8") as f:
        f.write("# Hyperparameters of the main runs\n\n" + hyper)
    made = figures(args.out, L, curve, gates, losses, args.results)
    finished = [r for r in comp if r["finished"]]
    total_wall = sum(r["wall_hours"] for r in finished)
    total_up = sum(r["learner_hours"] or 0 for r in finished)
    md = ["# Supporting numbers for the paper", "",
          "Generated by `python -m tools.paper_stats` from the exported results; no games played.", "",
          "## Compute", "",
          f"{len(finished)} finished self-play runs: {total_wall:,.0f} wall-clock hours, {total_up:,.0f} learner hours "
          "(one GPU node each; the learner hours exclude time the run was paused).", "",
          md_table(comp, ["run", "group", "games", "wall_hours", "learner_hours", "games_per_learner_hour", "gates",
                          "promotions", "gate_hours", "host"]), "",
          "## Seed spread", "",
          "Elo (HeuristicBot = 0) of the same recipe under different seeds. `sd` is the between-seed standard "
          "deviation; `fit_se` the largest single rating's standard error in that ladder, for comparison.", "",
          md_table(seeds, ["agent", "ladder", "seeds", "elos", "mean", "sd", "range", "fit_se"]), "",
          "## Training curve (Elo)", "",
          "From the curve ladders (`run.py eval-curves`); each checkpoint against HeuristicBot and the run's final "
          "network.", "",
          md_table(curve, ["run", "checkpoint", "games", "step", "elo", "se"]), "",
          "## Adaptive bidding, with 95% Wilson intervals", "",
          "`bid_winner_wins_game3` near 0.5 means the winning bid roughly equalized the decks; `ci` is its interval "
          "over the matches that reached game 3.", "",
          md_table(bids, ["opponent", "gap", "matches", "game3", "game3_ci", "mean_bid", "bid_winner_wins_game3", "ci",
                          "match_score", "match_score_ci"]), "",
          "## Gates", "", f"{len(gates)} gates in all; per gate in `gates.csv`.", "",
          "## Hyperparameters", "", hyper, "",
          "## Figures", "", ", ".join(f"`{m}.pdf`" for m in made) if made else "(matplotlib not installed)", ""]
    with open(os.path.join(args.out, "paper_stats.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"wrote {args.out}/paper_stats.md, compute.csv, gates.csv, losses.csv, elo_curve.csv, hyperparameters.md"
          + (f", figures {', '.join(made)}" if made else ""))


if __name__ == "__main__":
    main()

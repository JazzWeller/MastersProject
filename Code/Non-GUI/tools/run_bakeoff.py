#!/usr/bin/env python3
"""The bake-off (Agent Training Plan, Milestone M9): diagnostics first, then
the training arms, then the evaluation matrix -- each stage resumable, each
writing into one bake-off directory ($KEYFORGE_DATA/bakeoff/<name>/).

    python -m tools.run_bakeoff --name main --bc <bc.kfc> --stages diagnostics
    python -m tools.run_bakeoff --name main --bc <bc.kfc> --stages train --seeds 2
    python -m tools.run_bakeoff --name main --bc <bc.kfc> --stages matrix --matrix-games 700

**diagnostics (gate G3).** The hidden-information diagnostic and the
search curve (tools/diag_*.py), with the heuristic evaluator -- they need no
trained network, and can cancel whole arms. The plan's decision rules are
applied, written in advance, and recorded in `decisions.json`:

| Finding | Consequence |
|---|---|
| Hidden-info gap < 3 points | drop the belief arm (student estimator only) |
| Search curve flat 10 -> 100 sims | DMC becomes the primary design |
| Search curve still steep at 1,000 | turn on the fast fork backend; re-budget |

**train.** The two search regimes x `--seeds` seeds, each an
`ml.selfplay_train` run warm-started from the BC checkpoint (runs are
resumable; re-running this stage continues unfinished ones). `--dmc` adds
the M8 arm. `--train-games` budget per run, `--cut` applies the plan's "if the
budget shrinks" list (1: drop mismatched cells, 2: one seed, 3: one arm).

**matrix.** train regime x play regime x leaf estimator (8 cells) plus the
anchors -- RandomBot, HeuristicBot, the BC network with no search, the DMC
agent, every promoted checkpoint -- round-robin over paired seeds
(ml/arena.py), then Bradley-Terry ratings with bootstrap intervals and an
explicit non-transitivity report (sim/rate.py). Compute is reported per
player (simulations/decision), never folded into one number.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional

from sim import data_root

_NON_GUI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = ("within_turn", "full_game")
ARM_CONFIGS = {"within_turn": "tier2_selfplay_within_turn.json", "full_game": "tier3_selfplay_full_game.json"}


def _dir(name: str) -> str:
    d = os.path.join(data_root.resolve("bakeoff"), name)
    os.makedirs(d, exist_ok=True)
    return d


def _load(path: str, default):
    return json.load(open(path)) if os.path.exists(path) else default


def _save(path: str, data) -> None:
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=str)
    os.replace(path + ".tmp", path)


# --------------------------------------------------------------- diagnostics
def decide(hidden: List[dict], curve: List[dict]) -> dict:
    """The M9 decision rules, applied mechanically."""
    out = {"rules": []}
    if hidden:
        gap = hidden[0]["a_score"] - hidden[-1]["a_score"]
        se = (hidden[0]["standard_error"] ** 2 + hidden[-1]["standard_error"] ** 2) ** 0.5
        out["hidden_info_gap"] = round(gap, 4)
        out["hidden_info_gap_se"] = round(se, 4)
        out["belief_arm"] = "drop" if gap < 0.03 else "keep"
        out["rules"].append(f"hidden-information gap {gap:+.3f} (+/- {se:.3f}) -> belief arm: {out['belief_arm']}")
    if curve:
        by = {r["simulations"]: r["a_score"] for r in curve}
        if 10 in by and 100 in by:
            flat = by[100] - by[10] < max(0.02, 2 * curve[0]["standard_error"])
            out["dmc_primary"] = bool(flat)
            out["rules"].append(f"search curve 10 -> 100 sims: {by[10]:.3f} -> {by[100]:.3f} -> "
                                f"{'flat: DMC becomes the primary design' if flat else 'search helps'}")
        top = max(by)
        prev = max([s for s in by if s < top], default=None)
        if prev is not None:
            steep = by[top] - by[prev] > max(0.02, 2 * curve[0]["standard_error"])
            out["fast_fork"] = bool(steep)
            out["rules"].append(f"search curve {prev} -> {top} sims: {by[prev]:.3f} -> {by[top]:.3f} -> "
                                f"{'still steep: turn on the fast fork backend and re-budget' if steep else 'saturating'}")
    return out


def stage_diagnostics(d: str, args) -> dict:
    from tools.diag_hidden_info import run_search_conditions
    from tools.diag_search_curve import run_search_levels

    path = os.path.join(d, "diagnostics.json")
    diag = _load(path, {})
    for regime in ARMS:
        key = f"hidden_{regime}"
        if key not in diag:
            diag[key] = run_search_conditions(regime, args.diag_sims, args.diag_seeds, args.workers)
            _save(path, diag)
        key = f"curve_{regime}"
        if key not in diag:
            diag[key] = run_search_levels(regime, args.curve_levels, args.diag_seeds, args.workers)
            _save(path, diag)
    decisions = {r: decide(diag[f"hidden_{r}"], diag[f"curve_{r}"]) for r in ARMS}
    _save(os.path.join(d, "decisions.json"), decisions)
    for r, dec in decisions.items():
        for rule in dec["rules"]:
            print(f"[G3 {r}] {rule}", flush=True)
    return decisions


# --------------------------------------------------------------------- train
def run_ids(args) -> Dict[str, str]:
    arms = list(ARMS)
    seeds = args.seeds
    if args.cut >= 2:
        seeds = 1
    if args.cut >= 3:
        arms = ["within_turn"]
    ids = {}
    for arm in arms:
        for s in range(seeds):
            ids[f"{arm}/s{s}"] = f"{args.name}-{arm.replace('_', '-')}-s{s}"
    if args.dmc:
        ids["dmc/s0"] = f"{args.name}-dmc-s0"
    return ids


def stage_train(d: str, args) -> None:
    ids = run_ids(args)
    for key, run_id in ids.items():
        arm = key.split("/")[0]
        seed = int(key.split("/s")[1])
        cfg_name = "tier4_dmc.json" if arm == "dmc" else ARM_CONFIGS[arm]
        from agent import config as config_mod

        # Each seed is its own config file (the seed is part of the hash).
        cfg = config_mod._read_file(os.path.join(config_mod.CONFIG_DIR, cfg_name))
        cfg["seed"] = seed
        cfg_path = os.path.join(d, f"{run_id}.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=1)
        cmd = [sys.executable, "-m", "ml.selfplay_train", "--run", run_id, "--config", cfg_path, "--port", str(6150 + seed)]
        if args.bc:
            cmd += ["--init", args.bc]
        if arm == "dmc":
            cmd += ["--mode", "dmc"]
        if args.train_games:
            cmd += ["--games", str(args.train_games)]
        print(f"[train] {' '.join(cmd)}", flush=True)
        code = subprocess.call(cmd, cwd=_NON_GUI)
        if code != 0:
            raise SystemExit(f"training run {run_id} exited with {code} -- re-run this stage to resume it")


# -------------------------------------------------------------------- matrix
def _best_checkpoint(run_id: str) -> Optional[str]:
    from agent.telemetry import runs_root

    best = os.path.join(runs_root(), run_id, "best.json")
    if not os.path.exists(best):
        return None
    h = json.load(open(best))["checkpoint"]
    return os.path.join(runs_root(), run_id, "checkpoints", h + ".kfc")


def build_players(args, decisions: dict):
    from ml.arena import Player
    from ml.checkpoints import load_model
    from ml.infer_server import TorchModel

    device = args.device
    models = {}

    def model(path):
        if path not in models:
            net, _m, _o = load_model(path)
            models[path] = TorchModel(net, device)
        return models[path]

    players = [Player("random", "bot", bot="random"), Player("heuristic", "bot", bot="heuristic")]
    if args.bc:
        players.append(Player("bc-no-search", "net", model(args.bc)))
    ids = run_ids(args)
    leaves = ["student"]
    if all(decisions.get(r, {}).get("belief_arm") != "drop" for r in ARMS):
        leaves.append("belief_oracle")
    for key, run_id in ids.items():
        arm = key.split("/")[0]
        ckpt = _best_checkpoint(run_id)
        if ckpt is None:
            continue
        if arm == "dmc":
            players.append(Player("dmc", "net", model(ckpt), mode="q"))
            continue
        play_regimes = ARMS if args.cut < 1 else (arm,)
        for play in play_regimes:
            for leaf in leaves:
                players.append(Player(f"train={arm}/{key.split('/')[1]} play={play} leaf={leaf}", "search", model(ckpt),
                                      regime=play, leaf=leaf, simulations=args.sims))
    return players


def stage_matrix(d: str, args, decisions: dict) -> dict:
    from ml.arena import play_paired
    from sim.rate import bradley_terry, bradley_terry_intervals, non_transitivity_report

    path = os.path.join(d, "matrix.json")
    matrix = _load(path, {"pairings": {}})
    players = build_players(args, decisions)
    seeds_per = max(1, args.matrix_games // 4)
    for i, (a, b) in enumerate(itertools.combinations(players, 2)):
        key = f"{a.name} || {b.name}"
        if key in matrix["pairings"]:
            continue
        if a.kind == "bot" and b.kind == "bot":
            continue
        t0 = time.time()
        first = 5_000_000 + 1000 * i
        rep = play_paired(a, b, range(first, first + seeds_per), max_turns=200, concurrency=args.concurrency,
                          log=lambda m: print(m, flush=True))
        matrix["pairings"][key] = {"a": a.name, "b": b.name, "a_wins": rep.a_wins, "b_wins": rep.b_wins, "draws": rep.draws,
                                   "games": rep.games, "score": rep.score, "seconds": round(time.time() - t0, 1)}
        _save(path, matrix)
        print(f"[matrix] {key}: {rep.score:.3f} over {rep.games}", flush=True)
    games = []
    wr = {}
    for p in matrix["pairings"].values():
        games += [(p["a"], p["b"], 1.0)] * p["a_wins"] + [(p["a"], p["b"], 0.0)] * p["b_wins"] + [(p["a"], p["b"], 0.5)] * p["draws"]
        wr[(p["a"], p["b"])] = p["score"]
        wr[(p["b"], p["a"])] = 1 - p["score"]
    ratings = bradley_terry(games, anchor="heuristic")
    ci = bradley_terry_intervals(games, anchor="heuristic", samples=args.bootstrap)
    matrix["ratings"] = {n: {"rating": round(r.rating, 4), "ci95": [round(x, 4) for x in ci.get(n, (r.rating, r.rating))], "games": r.games}
                         for n, r in sorted(ratings.items(), key=lambda kv: -kv[1].rating)}
    matrix["non_transitivity"] = non_transitivity_report(ratings, wr)
    matrix["compute"] = {p.name: {"kind": p.kind, "simulations_per_decision": p.simulations if p.kind == "search" else (1 if p.kind == "net" else 0)} for p in players}
    _save(path, matrix)
    write_report(d, decisions, matrix)
    return matrix


def write_report(d: str, decisions: dict, matrix: dict) -> None:
    lines = ["# Bake-off", "", "## G3 decisions", ""]
    for r, dec in decisions.items():
        for rule in dec.get("rules", []):
            lines.append(f"- **{r}**: {rule}")
    lines += ["", "## Ratings (Bradley-Terry, HeuristicBot = 0)", "", "| Player | rating | 95% CI | games | sims/decision |", "|---|---|---|---|---|"]
    for n, r in matrix.get("ratings", {}).items():
        c = matrix.get("compute", {}).get(n, {})
        lines.append(f"| {n} | {r['rating']:+.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['games']} | {c.get('simulations_per_decision', '--')} |")
    lines += ["", "## Non-transitivity", ""] + ([f"- {x}" for x in matrix.get("non_transitivity", [])] or ["- none found"])
    with open(os.path.join(d, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="main")
    parser.add_argument("--stages", default="diagnostics,train,matrix")
    parser.add_argument("--bc", default=None, help="the M3 behaviour-cloning checkpoint (warm start + anchor)")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--dmc", action="store_true")
    parser.add_argument("--train-games", type=int, default=None, help="game budget per training run (default: the config's)")
    parser.add_argument("--matrix-games", type=int, default=700, help="games per matrix pairing")
    parser.add_argument("--cut", type=int, default=0, help="the plan's budget-cut list: 0 none .. 3")
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--diag-sims", type=int, default=100)
    parser.add_argument("--diag-seeds", type=int, default=50)
    parser.add_argument("--curve-levels", type=int, nargs="+", default=[1, 10, 100, 1000])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    d = _dir(args.name)
    stages = set(args.stages.split(","))
    decisions = _load(os.path.join(d, "decisions.json"), {})
    if "diagnostics" in stages:
        decisions = stage_diagnostics(d, args)
    if "train" in stages:
        stage_train(d, args)
    if "matrix" in stages:
        stage_matrix(d, args, decisions)
        print(f"report: {os.path.join(d, 'report.md')}")


if __name__ == "__main__":
    main()

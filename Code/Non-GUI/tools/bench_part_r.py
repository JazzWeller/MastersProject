#!/usr/bin/env python3
"""The benchmarks Part R is judged by (Agent Observation Plan, R0, R6, R8):

- `engine`: raw engine decisions/s, HeuristicBot against itself, views on;
- `forks`: the cost of an exact fork at sampled decisions of HeuristicBot
  games, by backend (`replay`, `copy`, and `auto` -- copy off a cached
  boundary snapshot) and by where the decision falls: at a boundary
  (CHOOSE_ACTION / CHOOSE_HOUSE / TAKE_ARCHIVE, nothing resolving) or
  mid-resolution, where `copy` was refused before R6;
- `search`: simulations/s of the M5 searches at 100 simulations with the
  heuristic evaluator (G3's settings; full-game with quiet leaves), single
  process, against HeuristicBot.

    python -m tools.bench_part_r --out bench.json [--execution compiled]

Single-threaded on purpose: run it on an otherwise idle machine, and repeat
a run before believing a difference under ~10%.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time

from bots.heuristic_bot import HeuristicBot
from keyforge.config import GameConfig
from keyforge.enums import BOUNDARY_KINDS
from keyforge.game import Game


def _game(seed: int, execution=None) -> Game:
    config = GameConfig(decks=("fignor", "igor"), seed=seed, max_turns=200)
    return Game(config, execution=execution) if execution else Game(config)


def bench_engine(n_games: int, execution=None) -> dict:
    _game(0, execution)  # loads the compiled routines (once per process) outside the timing
    decisions = 0
    t0 = time.perf_counter()
    for g in range(n_games):
        game = _game(g, execution)
        bots = {1: HeuristicBot(seed=g), 2: HeuristicBot(seed=g + 1)}
        while not game.is_over:
            d = game.pending_decision
            game.submit(bots[d.player].decide(game.view_for(d.player), d))
            decisions += 1
    seconds = time.perf_counter() - t0
    return {"games": n_games, "decisions": decisions, "seconds": round(seconds, 2),
            "decisions_per_second": round(decisions / seconds)}


def _time(fn, repeat: int) -> float:
    best = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best.append(time.perf_counter() - t0)
    return statistics.median(best)


def bench_forks(n_games: int, per_game: int = 12, repeat: int = 5, execution=None) -> dict:
    rng = random.Random(0)
    rows = {"boundary": {}, "mid_resolution": {}}
    for g in range(n_games):
        game = _game(1000 + g, execution)
        bots = {1: HeuristicBot(seed=g), 2: HeuristicBot(seed=g + 1)}
        sample_at = None
        while not game.is_over:
            if sample_at is None:
                sample_at = set(rng.sample(range(10, 200), per_game))
            d = game.pending_decision
            if len(game.choice_record) in sample_at:
                where = "boundary" if d.kind in BOUNDARY_KINDS else "mid_resolution"
                for backend in ("replay", "copy", "auto"):
                    if backend == "copy" and where == "mid_resolution" and not getattr(game, "copy_anywhere", False):
                        continue
                    if backend == "replay":
                        fn = game.fork
                    elif backend == "copy":
                        fn = game.copy
                    else:
                        fn = game._fork_from_snapshot if where == "mid_resolution" else game.copy
                    rows[where].setdefault(backend, []).append(_time(fn, repeat) * 1000)
            game.submit(bots[d.player].decide(game.view_for(d.player), d))
    out = {}
    for where, by_backend in rows.items():
        out[where] = {b: {"n": len(v), "median_ms": round(statistics.median(v), 3), "mean_ms": round(statistics.fmean(v), 3)}
                      for b, v in by_backend.items() if v}
    return out


def bench_search(n_seeds: int, simulations: int = 100) -> dict:
    from sim.parallel_eval import AgentSpec, evaluate_parallel

    out = {}
    for regime, kw in (("within_turn", {}), ("full_game", {"quiet_leaves": True})):
        name = "search-" + regime.replace("_", "-")
        res = evaluate_parallel(AgentSpec.of(name, simulations=simulations, **kw), AgentSpec.of("heuristic"),
                                range(4_000_000, 4_000_000 + n_seeds), workers=1, max_turns=200)
        sps = res.get("mean_seconds_per_search") or 0.0
        out[regime] = {"games": res["games"], "score": res["a_score"], "ms_per_search": round(sps * 1000, 2),
                       "simulations_per_second": round(simulations / sps) if sps else None}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=None)
    parser.add_argument("--execution", default=None, help="the engine's execution mode (Part R: native / compiled)")
    parser.add_argument("--engine-games", type=int, default=60)
    parser.add_argument("--fork-games", type=int, default=20)
    parser.add_argument("--search-seeds", type=int, default=5)
    parser.add_argument("--only", default="engine,forks,search")
    args = parser.parse_args()
    only = set(args.only.split(","))
    report = {"execution": args.execution or "native"}
    if "engine" in only:
        report["engine"] = bench_engine(args.engine_games, args.execution)
        print("engine", report["engine"], flush=True)
    if "forks" in only:
        report["forks"] = bench_forks(args.fork_games, execution=args.execution)
        print("forks", json.dumps(report["forks"]), flush=True)
    if "search" in only:
        report["search"] = bench_search(args.search_seeds)
        print("search", report["search"], flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()

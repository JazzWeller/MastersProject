#!/usr/bin/env python3
"""Paired-seed strength of any registered agent, in parallel (Agent
Training Plan, M5 gate G2 and the M9 anchors).

    python -m tools.eval_search --agent search-within-turn --sims 200 --seeds 100 --workers 8
    python -m tools.eval_search --agent search-full-game --opponent random --seeds 50 --gate 0.95

`--gate X` prints PASS/FAIL against a score threshold (G2: >= 0.55 vs
HeuristicBot at 200 simulations; M5 also asks >= 0.95 vs RandomBot), and
`--run ID` journals the verdict into that run.
"""

from __future__ import annotations

import argparse
import json

from sim.parallel_eval import AgentSpec, evaluate_parallel, wilson_interval


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", default="search-within-turn")
    parser.add_argument("--sims", type=int, default=200)
    parser.add_argument("--resample", default=None, help="own_deck | opponent_private | all (search agents)")
    parser.add_argument("--opponent", default="heuristic")
    parser.add_argument("--seeds", type=int, default=100, help="paired seeds (4 games each)")
    parser.add_argument("--first-seed", type=int, default=2_000_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--gate", type=float, default=None)
    parser.add_argument("--gate-name", default="G2")
    parser.add_argument("--run", default=None)
    parser.add_argument("--out", default=None, help="write the JSON result here")
    args = parser.parse_args()

    kwargs = {}
    if args.agent.startswith("search-"):
        kwargs["simulations"] = args.sims
        if args.resample:
            kwargs["resample"] = args.resample
    a = AgentSpec.of(args.agent, **kwargs)
    b = AgentSpec.of(args.opponent)
    seeds = range(args.first_seed, args.first_seed + args.seeds)
    res = evaluate_parallel(a, b, seeds, workers=args.workers, max_turns=args.max_turns)
    lo, hi = wilson_interval(res["a_wins"] + 0.5 * res["draws"], res["games"])
    res["interval95"] = [round(lo, 4), round(hi, 4)]
    if args.gate is not None:
        res["gate"] = {"name": args.gate_name, "threshold": args.gate, "verdict": "PASS" if res["a_score"] >= args.gate else "FAIL"}
    print(json.dumps(res, indent=1, default=str))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1, default=str)
    if args.run and args.gate is not None:
        from agent.telemetry import Run

        Run.open(args.run).journal(
            f"gate_{args.gate_name}", game_index=0, verdict=res["gate"]["verdict"], score=res["a_score"],
            games=res["games"], agent=args.agent, simulations=args.sims, opponent=args.opponent,
        )


if __name__ == "__main__":
    main()

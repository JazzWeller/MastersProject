#!/usr/bin/env python3
"""Search strength of trained networks, v2 or v1, on paired seeds (Agent
Observation Plan, O10's search rungs). WSL, CUDA.

    python -m tools.eval_v2 --a v2:tier0b2-a8 --b heuristic --sims 100 --seeds 100 --workers 8
    python -m tools.eval_v2 --a v2:tier0b2-a6 --b v1:tier0b-a0 --sims 100 --seeds 100

A player is `v2:<run id>` (its network-v2 checkpoint, searched with the
`student_v2` leaf in its history's request form), `v1:<run id>` (a v1
checkpoint, the `student` leaf) or a bot name (`heuristic`). Both searchers
use the same regime, simulations and O3's `chance_exact` determinization.
Each seed is played in both seat orders and both deck assignments
(`ml.arena.play_paired`); seeds are split over `--workers` processes, each
with its own copy of the networks on the GPU.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time


def _checkpoint(run_id: str) -> str:
    from sim import data_root

    root = data_root.resolve(os.path.join("runs", run_id))
    with open(os.path.join(root, "bc_report.json"), "r", encoding="utf-8") as f:
        ckpt = json.load(f)["checkpoint"]
    store = os.path.join(root, "checkpoints")
    return next(os.path.join(store, n) for n in os.listdir(store) if n.startswith(ckpt))


def _player(spec: str, args, name: str):
    from ml.arena import Player

    if ":" not in spec:
        return Player(name=name, kind="bot", bot=spec)
    kind, run_id = spec.split(":", 1)
    if kind == "v2":
        from agent.search.leaf_v2 import history_form
        from ml.checkpoints import load_model_v2
        from ml.infer_v2 import TorchModelV2

        net, _meta = load_model_v2(_checkpoint(run_id))
        return Player(name=name, kind="search", model=TorchModelV2(net, "cuda", compile=False), regime=args.regime,
                      leaf="student_v2", history=history_form(net.history_arch), simulations=args.sims,
                      determinization="chance_exact")
    if kind == "v1":
        from ml.infer_server import load_torch_model

        return Player(name=name, kind="search", model=load_torch_model(_checkpoint(run_id), "cuda"), regime=args.regime,
                      leaf="student", simulations=args.sims, determinization="chance_exact")
    raise ValueError(f"unknown player {spec!r} (v2:<run> | v1:<run> | a bot name)")


def _work(task):
    seeds, args = task
    from ml.arena import play_paired

    a, b = _player(args.a, args, "a"), _player(args.b, args, "b")
    rep = play_paired(a, b, seeds, max_turns=args.max_turns, concurrency=args.concurrency)
    return {"games": rep.games, "a_wins": rep.a_wins, "b_wins": rep.b_wins, "draws": rep.draws, "forfeits": rep.forfeits,
            "turns": rep.turns, "by_seat": {str(k): v for k, v in rep.by_seat.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", default="heuristic")
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--regime", default="within_turn", choices=("within_turn", "full_game"))
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--first-seed", type=int, default=3_000_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=16, help="games in flight per worker")
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    chunks = [seeds[k::args.workers] for k in range(args.workers) if seeds[k::args.workers]]
    t0 = time.time()
    with mp.get_context("spawn").Pool(len(chunks)) as pool:
        parts = pool.map(_work, [(c, args) for c in chunks])
    tot = {k: sum(p[k] for p in parts) for k in ("games", "a_wins", "b_wins", "draws", "forfeits")}
    score = (tot["a_wins"] + 0.5 * tot["draws"]) / tot["games"] if tot["games"] else 0.0
    from sim.parallel_eval import wilson_interval

    lo, hi = wilson_interval(tot["a_wins"] + 0.5 * tot["draws"], tot["games"])
    turns = [t for p in parts for t in p["turns"]]
    res = dict(tot, a=args.a, b=args.b, sims=args.sims, regime=args.regime, seeds=args.seeds, a_score=round(score, 4),
               interval95=[round(lo, 4), round(hi, 4)], mean_turns=round(sum(turns) / len(turns), 2) if turns else None,
               seconds=round(time.time() - t0, 1))
    print(json.dumps(res, indent=1))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()

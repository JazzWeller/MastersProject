#!/usr/bin/env python3
"""CLI entry point for one self-play actor worker (Agent Interface Plan,
Milestone H) -- spawned by `sim.actor.run_self_play`, one OS process per
worker, matching this project's established cross-process pattern
(tests/_contract_worker.py, tests/_determinism_worker.py) rather than
`multiprocessing.Pool`.

Run directly for a single worker: `python -m sim.actor_worker --shard-path
run/worker_0000.jsonl --n-games 100 --worker-id 0 --run-seed 0`.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.actor import run_actor  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-path", required=True)
    parser.add_argument("--n-games", type=int, required=True)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--run-seed", type=int, default=0)
    parser.add_argument("--p1-deck", default="fignor")
    parser.add_argument("--p2-deck", default="igor")
    parser.add_argument("--agent1", default="random")
    parser.add_argument("--agent2", default="random")
    parser.add_argument("--max-turns", type=int, default=200)
    args = parser.parse_args(argv)

    played = run_actor(
        shard_path=args.shard_path, n_games=args.n_games, decks=(args.p1_deck, args.p2_deck),
        agent1=args.agent1, agent2=args.agent2, run_seed=args.run_seed, worker_id=args.worker_id,
        max_turns=args.max_turns,
    )
    print(f"worker {args.worker_id}: played {played} game(s), shard now has {args.n_games} total")


if __name__ == "__main__":
    main()

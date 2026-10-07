"""Agent Observation Plan, Milestone O5: the v2 encoding's fuzz sweep.

    python -m tools.o5_acceptance --games 1000 [--workers 8] [--every 10]

For `--games` games per pool (tools/o1_acceptance.py's pools), encodes
every decision for both viewers (an unknown vocabulary entry raises) and,
every `--every` decisions, checks the bytes are identical in a constrained
and a chance_exact world (no leak, I1). Reports the encode cost.
"""

from __future__ import annotations

import argparse
import random
import sys
import time

from agent.features_v2 import encode_v2
from keyforge.enums import Resample
from keyforge.game import Game
from tools.o1_acceptance import POOLS, bots_for, game_config


def _worker(args):
    pool, i, every = args
    config = game_config(pool, i)
    game = Game(config)
    bots = bots_for(i, config.seed)
    n = encodes = worlds = 0
    spent = 0.0
    try:
        while not game.is_over:
            for v in (1, 2):
                t0 = time.perf_counter()
                base = encode_v2(game, v).to_bytes()
                spent += time.perf_counter() - t0
                encodes += 1
                if n % every == 0 and game.copy_anywhere:
                    for sampler in ("constrained", "chance_exact"):
                        w = game.fork_determinized(v, random.Random(n), Resample.ALL, backend="copy", sampler=sampler)
                        worlds += 1
                        if encode_v2(w, v).to_bytes() != base:
                            return pool, i, [f"decision {n}, viewer {v}, {sampler}: the bytes differ"], encodes, worlds, spent
            d = game.pending_decision
            game.submit(bots[d.player].decide(game.view_for(d.player), d))
            n += 1
    except Exception as ex:
        return pool, i, [f"decision {n}: {type(ex).__name__}: {ex}"], encodes, worlds, spent
    return pool, i, [], encodes, worlds, spent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)
    jobs = [(pool, i, args.every) for pool in POOLS for i in range(args.games)]
    t0 = time.time()
    if args.workers > 1:
        import multiprocessing as mp
        results = mp.Pool(args.workers).imap_unordered(_worker, jobs, chunksize=4)
    else:
        results = map(_worker, jobs)
    failures = encodes = worlds = 0
    spent = 0.0
    for pool, i, problems, e, w, s in results:
        encodes, worlds, spent = encodes + e, worlds + w, spent + s
        if problems:
            failures += 1
            print(f"{pool} game {i}:", *problems, sep="\n  ", flush=True)
    print(f"{len(jobs)} games, {encodes} encodes ({spent / max(encodes, 1) * 1e6:.0f} us each), {worlds} worlds, "
          f"{failures} with problems, {time.time() - t0:.0f} s")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

"""Fork-based `multiprocessing.Pool` self-play worker pool (Agent Interface
Plan, Milestone H) -- the WSL/Linux-specific alternative to `sim.actor`'s
`run_self_play`, which uses `subprocess.Popen` everywhere instead
specifically to avoid this exact platform split. Both call the SAME
`sim.actor.run_actor`/`game_seed` and write the SAME shard-file layout, so a
shard produced by one is indistinguishable from a shard produced by the
other -- this module exists only to let a caller who's already committed to
running on WSL (this project's own choice of self-play platform) actually
get the "fast pool startup and copy-on-write sharing of warmed state" the
plan's own platform note describes, which `subprocess.Popen` -- every
worker re-importing everything from scratch, on any platform -- cannot.

**POSIX/WSL/Linux only.** `multiprocessing.get_context("fork")` raises on
Windows (no `fork()` there, same underlying reason `keyforge.branching_fork`
needs `available()`). Guard any call with `available()`.

**The CUDA rule applies here too**: never call `run_self_play_pool` from a
process that has already initialized CUDA (e.g. one holding a `bots.
torch_model` model) -- `fork()` does not survive that, exactly as documented
in `keyforge/branching_fork.py`.
"""

from __future__ import annotations

import multiprocessing
import os
from typing import List, Tuple

from sim.actor import run_actor


def available() -> bool:
    """Whether the `fork` start method exists in this process -- `False`
    on Windows."""
    return "fork" in multiprocessing.get_all_start_methods()


def _pool_worker(args: Tuple) -> int:
    shard_path, n_games, decks, agent1, agent2, run_seed, worker_id, max_turns = args
    return run_actor(shard_path, n_games, decks, agent1, agent2, run_seed, worker_id, max_turns)


def run_self_play_pool(
    n_workers: int,
    games_per_worker: int,
    shard_dir: str,
    run_seed: int = 0,
    decks: Tuple[str, str] = ("fignor", "igor"),
    agent1: str = "random",
    agent2: str = "random",
    max_turns: int = 200,
) -> List[str]:
    """Same contract and shard layout as `sim.actor.run_self_play`, backed
    by a `fork`-context `multiprocessing.Pool` of `n_workers` instead of one
    `subprocess.Popen` per worker. Raises `RuntimeError` if `fork` isn't
    available (Windows)."""
    if not available():
        raise RuntimeError("run_self_play_pool needs the 'fork' start method (POSIX/WSL/Linux only)")
    os.makedirs(shard_dir, exist_ok=True)
    shard_paths = [os.path.join(shard_dir, f"worker_{i:04d}.jsonl") for i in range(n_workers)]
    tasks = [
        (shard_path, games_per_worker, decks, agent1, agent2, run_seed, worker_id, max_turns)
        for worker_id, shard_path in enumerate(shard_paths)
    ]
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(n_workers) as pool:
        pool.map(_pool_worker, tasks)
    return shard_paths

"""Self-play actor loop (Agent Interface Plan, Milestone H): P worker
processes each play games and write sharded, append-only trajectory files
-- one worker's shard is written to by nobody else, so no coordination
between workers is needed beyond the shard directory.

Uses `subprocess.Popen`, not `multiprocessing.Pool`: this project's own
established pattern for cross-process work (tests/_contract_worker.py,
tests/_determinism_worker.py) already sidesteps `multiprocessing`'s
Windows-`spawn`-vs-POSIX-`fork` differences entirely, and there is no
reason to take on a second cross-process mechanism for the same job. The
plan's own platform note applies here exactly as it does to any other
worker pool: on WSL, `multiprocessing` defaults to `fork` (fast startup,
copy-on-write sharing) if a caller chooses to use it instead; on Windows
only `spawn` exists. `subprocess.Popen` needs neither -- every worker
re-imports everything either way, on every platform, which is the simplest
thing that is correct everywhere.

**Resumable.** `run_actor` counts the games already recorded in its shard
file and plays only the remainder, so restarting an interrupted worker with
the same arguments picks up where it left off. Each game's seed is derived
from `(run_seed, worker_id, game_index)` (sha256-based, portable across
interpreters -- see keyforge/keyed_random.py's docstring on why not
`random.Random.randrange`), so it never depends on how many games happen to
already be on disk, only on the index being resumed from.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from typing import List, Tuple

from bots.registry import make_agent
from keyforge.config import GameConfig
from sim.generate import play_and_record, read_shard, write_shard

_NON_GUI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "actor_worker.py")


def game_seed(run_seed: int, worker_id: int, game_index: int) -> int:
    """Deterministic per-`(run_seed, worker_id, game_index)` seed, built on
    sha256 (a stable, documented algorithm) -- matches
    `sim.driver.derive_agent_seed`'s own derivation style."""
    digest = hashlib.sha256(f"{run_seed!r}|{worker_id}|{game_index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def run_actor(
    shard_path: str,
    n_games: int,
    decks: Tuple[str, str],
    agent1: str,
    agent2: str,
    run_seed: int,
    worker_id: int,
    max_turns: int = 200,
) -> int:
    """Plays games until `shard_path` holds `n_games` of them, appending
    only whatever is still missing. Returns how many games this call
    actually played (0 if the shard was already complete -- the
    idempotent-restart case)."""
    already = len(read_shard(shard_path)) if os.path.exists(shard_path) else 0
    played = 0
    for game_index in range(already, n_games):
        seed = game_seed(run_seed, worker_id, game_index)
        config = GameConfig(decks=decks, seed=seed, max_turns=max_turns)
        agents = {1: make_agent(agent1, seed=seed), 2: make_agent(agent2, seed=seed + 1)}
        trajectory, _privileged = play_and_record(config, agents)
        write_shard(shard_path, [trajectory])
        played += 1
    return played


def run_self_play(
    n_workers: int,
    games_per_worker: int,
    shard_dir: str,
    run_seed: int = 0,
    decks: Tuple[str, str] = ("fignor", "igor"),
    agent1: str = "random",
    agent2: str = "random",
    max_turns: int = 200,
) -> List[str]:
    """Launches `n_workers` OS subprocesses, each running `actor_worker.py`
    against its own shard file `shard_dir/worker_{id:04d}.jsonl`. Blocks
    until every worker exits; raises if any exited non-zero. Returns the
    shard paths, in worker order."""
    os.makedirs(shard_dir, exist_ok=True)
    shard_paths = [os.path.join(shard_dir, f"worker_{i:04d}.jsonl") for i in range(n_workers)]
    procs = []
    for worker_id, shard_path in enumerate(shard_paths):
        argv = [
            sys.executable, _WORKER,
            "--shard-path", shard_path, "--n-games", str(games_per_worker),
            "--worker-id", str(worker_id), "--run-seed", str(run_seed),
            "--p1-deck", decks[0], "--p2-deck", decks[1],
            "--agent1", agent1, "--agent2", agent2, "--max-turns", str(max_turns),
        ]
        procs.append(subprocess.Popen(argv, cwd=_NON_GUI_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
    for p, shard_path in zip(procs, shard_paths):
        output, _ = p.communicate()
        if p.returncode != 0:
            raise RuntimeError(f"actor worker for {shard_path!r} exited with code {p.returncode}:\n{output}")
    return shard_paths

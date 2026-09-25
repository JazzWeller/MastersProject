"""Where pipeline output lives (Agent Interface Plan, Milestone 0).

Trajectory shards, checkpoints, JSONL results and caches all go under one
data root, set by the `KEYFORGE_DATA` environment variable. Under WSL it
should point at the Linux filesystem (e.g. `~/keyforge-data`), not at the
Windows checkout under `/mnt/c`, where heavy file I/O is slow.

Pipeline entry points pass their output locations through `resolve()`: an
absolute path is used as given, and a relative one lands under the data
root. So `run_self_play(..., shard_dir="run-7/shards")` writes to
`$KEYFORGE_DATA/run-7/shards` on either platform.
"""

from __future__ import annotations

import os

ENV_VAR = "KEYFORGE_DATA"
DEFAULT_DIRNAME = "keyforge-data"


def data_root() -> str:
    """`$KEYFORGE_DATA` if set, else `~/keyforge-data`. Not created here --
    whoever writes into it creates what it needs."""
    root = os.environ.get(ENV_VAR)
    if root:
        return os.path.abspath(os.path.expanduser(root))
    return os.path.join(os.path.expanduser("~"), DEFAULT_DIRNAME)


def resolve(path: str) -> str:
    """An absolute `path` unchanged; a relative one under `data_root()`."""
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(data_root(), path))

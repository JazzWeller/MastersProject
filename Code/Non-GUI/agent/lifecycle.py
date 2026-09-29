"""Process lifecycle for a run's child processes (Agent Training Plan, M7:
"shards are append-only and worker-exclusive"). Standard library only.

- `exit_with_parent()`: a child (actor, inference server, gate) exits when
  the learner that started it dies -- a killed learner otherwise leaves
  orphans that keep appending to shards and holding the server port while
  a relaunched run starts its own.
- `exclusive(path)`: an advisory lock beside a file that only one process
  may write (an actor's shard). A second writer fails loudly instead of
  interleaving frames. POSIX only (the runs live in WSL); a no-op elsewhere.
"""

from __future__ import annotations

import os
import threading
import time


def exit_with_parent(poll_seconds: float = 2.0, exit_code: int = 3) -> None:
    """Starts a daemon thread that ends this process once its parent is
    gone (re-parented). In-flight work is abandoned; everything durable is
    already crash-safe (shards truncate a half frame on resume)."""
    parent = os.getppid()
    if parent <= 1:
        return

    def watch():
        while True:
            time.sleep(poll_seconds)
            if os.getppid() != parent:
                os._exit(exit_code)

    threading.Thread(target=watch, name="exit-with-parent", daemon=True).start()


class ShardBusy(RuntimeError):
    pass


def exclusive(path: str):
    """Takes an exclusive, non-blocking lock on `path + ".lock"` for the
    life of the process (the returned handle must stay referenced).
    Raises `ShardBusy` if another live process holds it."""
    try:
        import fcntl
    except ImportError:  # Windows: no advisory locks; runs are WSL-only
        return None
    handle = open(path + ".lock", "a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise ShardBusy(f"{path} is being written by another live process -- an orphaned actor from an earlier launch?")
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle

"""E1: process-fork branch-and-run backend (Agent Interface Plan, Milestone
E). Same signature as `keyforge.branching.run_branches`, so search code
written against that stays unchanged when it switches backends.

`os.fork()` duplicates the entire process, including the suspended
generator frames that make `deepcopy` impossible -- so each child holds an
exact, drivable copy of `game` at ANY decision, mid-resolution included,
with flat cost per branch regardless of how deep into the game `game`
already is (replay's own cost grows linearly with prefix length). A
determinized branch resamples hidden info directly on the child's own
(already private, copy-on-write) `game` -- not through `fork_determinized`,
which reconstructs the whole game via `replay()`; that linear-in-prefix-
length cost is exactly what this backend exists to avoid.

**POSIX/WSL/Linux only**: `os.fork()` does not exist on Windows. Guard any
call with `available()`.

**Constraints, from the process-fork contract itself (see the plan):**
- The forking process must never have initialized CUDA (CUDA does not
  survive `fork`) -- satisfied here as everywhere else in this codebase:
  no keyforge/bots/sim module imports `torch`.
- The forking process must be single-threaded. Don't call this from a
  process that has other threads running (e.g. one also serving an
  `InferenceServer`).
"""

from __future__ import annotations

import dataclasses
import os
import pickle
import random
import select
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .enums import Resample


def available() -> bool:
    """Whether this backend can run at all in the current process."""
    return hasattr(os, "fork")


def run_branches_forked(
    game,
    n: int,
    work: Callable[[Any], Any],
    *,
    viewer: Optional[int] = None,
    resample: Resample = Resample.ALL,
    seeds: Optional[Sequence[int]] = None,
) -> List[Any]:
    """Same contract as `keyforge.branching.run_branches`. `work`'s return
    value must be picklable -- it crosses a pipe back to the parent; `game`
    itself never does (and never needs to: each child already has its own
    copy-on-write copy from the fork itself)."""
    if not available():
        raise RuntimeError("the process-fork backend needs os.fork() (POSIX/WSL/Linux only)")
    if seeds is None:
        seeds = [game._branch_seed(i) for i in range(n)]
    opponent = None if viewer is None else 3 - viewer

    # Every pipe is created BEFORE any fork, so every child's fd table is
    # identical at fork time regardless of which iteration created it; each
    # child then closes everything except its own write end. Doing this any
    # other way (e.g. opening pipe i just before forking child i) leaks
    # earlier write ends into later children, which can hang the parent's
    # EOF-based read forever -- a later child would still hold an earlier
    # pipe's write end open even though it never writes to it.
    read_ends: List[int] = []
    write_ends: List[int] = []
    for _ in seeds:
        r, w = os.pipe()
        read_ends.append(r)
        write_ends.append(w)

    pids: List[int] = [0] * n
    for i, seed in enumerate(seeds):
        pid = os.fork()
        if pid == 0:
            for r in read_ends:
                os.close(r)
            for j, w in enumerate(write_ends):
                if j != i:
                    os.close(w)
            _run_child(game, work, viewer, opponent, resample, seed, write_ends[i])
            os._exit(0)  # pragma: no cover -- unreachable: _run_child never returns control here
        pids[i] = pid

    for w in write_ends:
        os.close(w)

    results: List[Any] = [None] * n
    _collect(list(enumerate(read_ends)), results)
    for pid in pids:
        os.waitpid(pid, 0)
    return results


def _run_child(game, work, viewer, opponent, resample, seed, w: int) -> None:
    try:
        if viewer is not None:
            rng = random.Random(seed)
            if resample in (Resample.OWN_DECK, Resample.ALL):
                game._resample_own_deck(viewer, rng)
            if resample in (Resample.OPPONENT_PRIVATE, Resample.ALL):
                game._resample_hidden_pool(opponent, viewer, rng)
            # Without this, every future `event_rng` draw in this branch
            # would exactly match the true game's (both still derive from
            # the same `config.seed` and the same `_rng_counters`) --
            # exactly the future-leakage `fork_determinized` guards against.
            # A new object, not an in-place `.seed = ...`, matching that
            # same method (see its own comment on why).
            game.config = dataclasses.replace(game.config, seed=rng.getrandbits(63))
        payload = pickle.dumps(("ok", work(game)))
    except BaseException as e:  # noqa: BLE001 -- must reach the parent, whatever it is
        payload = pickle.dumps(("error", f"{type(e).__name__}: {e}"))
    with os.fdopen(w, "wb") as f:
        f.write(payload)


def _collect(indexed_read_fds: List[Tuple[int, int]], results: List[Any]) -> None:
    """Reads every child's pipe to EOF via `select()`, rather than one at a
    time in order -- so no child can hang waiting for the parent to drain a
    full pipe buffer while the parent is still busy reading a different
    one."""
    buffers: Dict[int, bytearray] = {fd: bytearray() for _, fd in indexed_read_fds}
    pending = {fd for _, fd in indexed_read_fds}
    while pending:
        ready, _, _ = select.select(list(pending), [], [])
        for fd in ready:
            chunk = os.read(fd, 65536)
            if chunk:
                buffers[fd].extend(chunk)
            else:
                pending.discard(fd)
                os.close(fd)
    for i, fd in indexed_read_fds:
        status, payload = pickle.loads(bytes(buffers[fd]))
        if status == "error":
            raise RuntimeError(f"branch {i} failed in a forked child: {payload}")
        results[i] = payload

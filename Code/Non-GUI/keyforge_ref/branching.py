"""Branch-and-run: the portable search primitive (Agent Interface Plan,
Milestones D and E).

`run_branches(game, n, work, ...)` runs `work(branch)` on each of `n`
branches of `game` and returns the results, in branch order. This
(replay-backed) implementation runs every branch in-process; Milestone E's
process-fork and snapshot-copy backends implement the exact same signature
faster, so search code written against `run_branches` needs no change to
switch backends later.

Search agents should be written against this "branch-and-run" form rather
than "branch-and-hold" (`Game.fork()`/`fork_determinized()` returning a
`Game` the caller keeps and drives itself): only the replay and
snapshot-copy backends can serve branch-and-hold at all, since a
process-fork's branch lives in a different, short-lived process and can
only hand back a result, not itself.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence

from .enums import Resample


def run_branches(
    game,
    n: int,
    work: Callable[[Any], Any],
    *,
    viewer: Optional[int] = None,
    resample: Resample = Resample.ALL,
    seeds: Optional[Sequence[int]] = None,
) -> List[Any]:
    """Runs `work(branch)` on `n` independent branches of `game` and
    returns the results in order. `viewer`/`resample` are forwarded to
    `Game.fork_determinized`; omitting `viewer` gives exact (privileged)
    branches via `Game.fork()` instead."""
    branches = game.fork_many(n, viewer=viewer, resample=resample, seeds=seeds)
    return [work(branch) for branch in branches]

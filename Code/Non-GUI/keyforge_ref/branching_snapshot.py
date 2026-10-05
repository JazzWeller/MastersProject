"""E2: snapshot-copy branch-and-run backend (Agent Interface Plan, Milestone
E). Same signature as `keyforge.branching.run_branches`, so search code
written against that stays unchanged when it switches backends -- with one
extra restriction `run_branches`/E1 don't have: `game` must be at a boundary
decision (`enums.BOUNDARY_KINDS`) or finished; `Game.copy()` raises
otherwise (see its own docstring for why -- a suspended generator frame
mid-resolution isn't something a hand-written copy can reconstruct).

Any platform (unlike E1, which needs `os.fork()`): `Game.copy()` is plain
Python. Whether it's actually faster than replay depends on how deep into
the game `game` already is -- replay's own cost grows with prefix length,
copy's doesn't, but copy has a larger constant factor. Measure with
`tools/bench_engine.py` before switching this on for real search; the plan's
own default is off pending that measurement.
"""

from __future__ import annotations

import dataclasses
import random
from typing import Any, Callable, List, Optional, Sequence

from .enums import Resample


def run_branches_snapshot(
    game,
    n: int,
    work: Callable[[Any], Any],
    *,
    viewer: Optional[int] = None,
    resample: Resample = Resample.ALL,
    seeds: Optional[Sequence[int]] = None,
) -> List[Any]:
    """Same contract as `keyforge.branching.run_branches`. Raises
    `ValueError` (via `Game.copy()`) if `game` is not at a boundary
    decision or finished."""
    if seeds is None:
        seeds = [game._branch_seed(i) for i in range(n)]
    results = []
    for seed in seeds:
        branch = game.copy()
        if viewer is not None:
            opponent = 3 - viewer
            rng = random.Random(seed)
            if resample in (Resample.OWN_DECK, Resample.ALL):
                branch._resample_own_deck(viewer, rng)
            if resample in (Resample.OPPONENT_PRIVATE, Resample.ALL):
                branch._resample_hidden_pool(opponent, viewer, rng)
            branch.config = dataclasses.replace(branch.config, seed=rng.getrandbits(63))
        results.append(work(branch))
    return results

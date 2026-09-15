"""CurrentQueue: tiers used to order queued effects within a resolution frame.

A "frame" of queued items is represented simply as a list handed to
`Game._resolve_batch`; nested frames (created when resolving one item queues
further items) fall naturally out of Python's call stack, since resolving an
item is a generator function that may itself call `_resolve_batch` again
before returning. That gives the same "inner frame resolves completely
before the outer frame continues" behaviour described in the plan, without
needing a separate explicit stack data structure.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# Tiers, in resolution order.
BEFORE_FIGHT = 0
FIGHTING = 1
AFTER = 2  # after fight / after reap
DESTROYED = 3
END = 4

_sequence = itertools.count()


@dataclass
class QueueItem:
    tier: int
    run: Callable  # callable() -> generator, the effect to resolve
    batch_id: Optional[int] = None
    description: str = ""
    sequence: int = field(default_factory=lambda: next(_sequence))


def new_batch_id() -> int:
    return next(_sequence)

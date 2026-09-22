"""Structured event log."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional

_BOTH_PLAYERS = frozenset({1, 2})


@dataclass
class LogEvent:
    kind: str
    data: Dict[str, Any] = field(default_factory=dict)
    # Which players are entitled to this event's content -- both by default.
    # A handful of zone-transition events name a card that was sitting in a
    # hidden zone (hand, deck, or a player's own archive) immediately before
    # the event; those are logged with `visible_to` narrowed to whoever
    # already legitimately knows that card (almost always just its owner),
    # so a player's own view (`view.build_view`'s `log_tail`) can filter
    # entries down to what they're actually entitled to see. See
    # Code/AGENT_INTERFACE_PLAN.md, Milestone C ("first, the leak").
    visible_to: FrozenSet[int] = field(default_factory=lambda: _BOTH_PLAYERS)

    def __repr__(self):
        return f"{self.kind}({self.data})"


class GameLog:
    def __init__(self):
        self.events: List[LogEvent] = []

    def add(self, kind: str, visible_to: Optional[Any] = None, **data) -> None:
        entitled = _BOTH_PLAYERS if visible_to is None else frozenset(visible_to)
        self.events.append(LogEvent(kind, data, visible_to=entitled))

    def tail(self, n: int) -> List[LogEvent]:
        return self.events[-n:]

    def visible_to(self, pid: int) -> List[LogEvent]:
        """Every event `pid` is entitled to see, engine-log order preserved."""
        return [e for e in self.events if pid in e.visible_to]

"""Structured event log."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class LogEvent:
    kind: str
    data: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return f"{self.kind}({self.data})"


class GameLog:
    def __init__(self):
        self.events: List[LogEvent] = []

    def add(self, kind: str, **data) -> None:
        self.events.append(LogEvent(kind, data))

    def tail(self, n: int) -> List[LogEvent]:
        return self.events[-n:]

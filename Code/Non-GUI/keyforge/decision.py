"""Decision objects: what the engine needs from a player next."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from .enums import DecisionKind


@dataclass
class Decision:
    player: int
    kind: DecisionKind
    prompt: str
    options: List[Any] = field(default_factory=list)
    min_n: int = 1
    max_n: int = 1

    def validate(self, choice) -> bool:
        if self.kind in (DecisionKind.CHOOSE_CARDS, DecisionKind.ORDER_EFFECTS):
            if not isinstance(choice, list):
                return False
            if not (self.min_n <= len(choice) <= self.max_n):
                return False
            if self.kind == DecisionKind.ORDER_EFFECTS:
                same_multiset = sorted(id(c) for c in choice) == sorted(id(o) for o in self.options)
                return same_multiset and len(choice) == len(self.options)
            for c in choice:
                if c not in self.options:
                    return False
            return True
        else:
            return choice in self.options

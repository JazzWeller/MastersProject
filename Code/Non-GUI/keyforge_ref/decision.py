"""Decision objects: what the engine needs from a player next."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from .enums import Affects, DecisionIntent, DecisionKind


@dataclass(slots=True)
class Decision:
    player: int
    kind: DecisionKind
    prompt: str
    options: List[Any] = field(default_factory=list)
    min_n: int = 1
    max_n: int = 1
    # Provenance (Agent Interface Plan, Milestone B): which card asked, what
    # the decision is for, whose cards it draws from, and whether declining
    # is legal in spirit (not just via min_n=0 -- a YES_NO "may" decision is
    # optional even though both its options are always present). All default
    # to None/False so every existing call site (and the GUI's rendering)
    # is unaffected by the fields' addition.
    source_card: Optional[Any] = None
    intent: Optional[DecisionIntent] = None
    affects: Optional[Affects] = None
    optional: bool = False

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

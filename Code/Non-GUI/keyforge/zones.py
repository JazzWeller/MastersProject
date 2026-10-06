"""Zone containers.

Every zone of a game knows what it is (`zone`: `(kind, player id)`) and
reports each card that enters or leaves it to the game's journal
(keyforge/journal.py, Agent Observation Plan O1) -- `attach(journal, kind,
pid)` wires that up. A zone built on its own (a test's) has no journal and
behaves exactly the same.
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional

from .cards.card import Card
from .keyed_random import portable_shuffle


class _Zone:
    _j = None
    zone = None

    def attach(self, journal, kind: str, pid: int) -> None:
        self._j = journal
        self.zone = (kind, pid)


class Deck(_Zone):
    def __init__(self, cards: Optional[List[Card]] = None):
        self._cards = deque(cards or [])

    def __len__(self):
        return len(self._cards)

    def is_empty(self) -> bool:
        return len(self._cards) == 0

    def draw_top(self) -> Optional[Card]:
        if not self._cards:
            return None
        card = self._cards.popleft()
        if self._j is not None:
            self._j.leave(card, self.zone, "draw_top", "top")
        return card

    def peek_top(self) -> Optional[Card]:
        if not self._cards:
            return None
        return self._cards[0]

    def put_on_top(self, card: Card) -> None:
        self._cards.appendleft(card)
        if self._j is not None:
            self._j.enter(card, self.zone, "put_on_top", "top")

    def put_on_bottom(self, card: Card) -> None:
        self._cards.append(card)
        if self._j is not None:
            self._j.enter(card, self.zone, "put_on_bottom", "bottom")

    def put_all(self, cards: List[Card], op: str = "add") -> None:
        """Adds `cards` at the bottom, in order (a deal, or a pile turned
        into a deck before its shuffle)."""
        self._cards.extend(cards)
        if self._j is not None:
            if op == "deal":
                self._j.deal(cards, self.zone)
            else:
                for c in cards:
                    self._j.enter(c, self.zone, op)

    def shuffle_in(self, cards: List[Card], rng) -> None:
        self._cards.extend(cards)
        if self._j is not None:
            for c in cards:
                self._j.enter(c, self.zone, "shuffle_in")
        as_list = list(self._cards)
        portable_shuffle(rng, as_list)
        self._cards = deque(as_list)
        if self._j is not None:
            self._j.shuffle(self.zone[1])

    def shuffle(self, rng) -> None:
        as_list = list(self._cards)
        portable_shuffle(rng, as_list)
        self._cards = deque(as_list)
        if self._j is not None:
            self._j.shuffle(self.zone[1])

    def remove(self, card: Card) -> bool:
        try:
            self._cards.remove(card)
        except ValueError:
            return False
        if self._j is not None:
            self._j.leave(card, self.zone, "remove")
        return True

    def take_all(self) -> List[Card]:
        cards = list(self._cards)
        self._cards = deque()
        if self._j is not None:
            for c in cards:
                self._j.leave(c, self.zone, "take_all")
        return cards

    def cards(self) -> List[Card]:
        return list(self._cards)


class _Pile(_Zone):
    """A zone that is an ordered list of cards: hand, discard, archive,
    purged."""

    def __init__(self):
        self._cards: List[Card] = []

    def __len__(self):
        return len(self._cards)

    def is_empty(self) -> bool:
        return len(self._cards) == 0

    def add(self, card: Card) -> None:
        self._cards.append(card)
        if self._j is not None:
            self._j.enter(card, self.zone, "add")

    def remove(self, card: Card) -> bool:
        try:
            self._cards.remove(card)
        except ValueError:
            return False
        if self._j is not None:
            self._j.leave(card, self.zone, "remove")
        return True

    def take_all(self) -> List[Card]:
        cards = self._cards
        self._cards = []
        if self._j is not None:
            for c in cards:
                self._j.leave(c, self.zone, "take_all")
        return cards

    def cards(self) -> List[Card]:
        return list(self._cards)


class DiscardPile(_Pile):
    def push(self, card: Card) -> None:
        self._cards.append(card)
        if self._j is not None:
            self._j.enter(card, self.zone, "push")

    def pop(self) -> Optional[Card]:
        if not self._cards:
            return None
        card = self._cards.pop()
        if self._j is not None:
            self._j.leave(card, self.zone, "pop")
        return card


class Hand(_Pile):
    pass


class Archive(_Pile):
    pass


class PurgedZone(_Pile):
    pass


class PlayArea:
    _j = None
    pid = None
    _bl = _ar = None  # its two zones, ("battleline", pid) and ("artifacts", pid)

    def __init__(self):
        self.creatures: List[Card] = []
        self.artifacts: List[Card] = []

    def attach(self, journal, kind: str, pid: int) -> None:
        self._j = journal
        self.pid = pid
        self._bl = ("battleline", pid)
        self._ar = ("artifacts", pid)

    def all_cards(self) -> List[Card]:
        return list(self.creatures) + list(self.artifacts)

    def add_creature(self, card: Card, flank: Optional[str] = None) -> None:
        if flank == "left":
            self.creatures.insert(0, card)
        else:
            self.creatures.append(card)
        if self._j is not None:
            self._j.enter(card, self._bl, "add_creature", flank or "right")

    def add_artifact(self, card: Card) -> None:
        self.artifacts.append(card)
        if self._j is not None:
            self._j.enter(card, self._ar, "add_artifact")

    def remove(self, card: Card) -> bool:
        if card in self.creatures:
            self.creatures.remove(card)
            if self._j is not None:
                self._j.leave(card, self._bl, "remove")
            return True
        if card in self.artifacts:
            self.artifacts.remove(card)
            if self._j is not None:
                self._j.leave(card, self._ar, "remove")
            return True
        return False

    def neighbors(self, card: Card):
        if card not in self.creatures:
            return []
        i = self.creatures.index(card)
        result = []
        if i > 0:
            result.append(self.creatures[i - 1])
        if i < len(self.creatures) - 1:
            result.append(self.creatures[i + 1])
        return result

    def swap(self, a: Card, b: Card) -> bool:
        """Swaps the battleline positions of `a` and `b` (Sanctum Guardian).
        Both must already be in this play area; a creature "swapped" with
        itself is a legal no-op (an effect may allow a creature to move
        "anywhere in the battleline, including where it already is")."""
        if a is b:
            return a in self.creatures
        if a not in self.creatures or b not in self.creatures:
            return False
        i, j = self.creatures.index(a), self.creatures.index(b)
        self.creatures[i], self.creatures[j] = self.creatures[j], self.creatures[i]
        if self._j is not None:
            self._j.note(a, self._bl, "swap", b.instance_id)
        return True

    def is_flank(self, card: Card) -> bool:
        if getattr(card, "forced_flank", False):
            return True  # Spectral Tunneler: considered a flank creature for the rest of the turn
        if card not in self.creatures:
            return False
        if len(self.creatures) == 1:
            return True
        return card is self.creatures[0] or card is self.creatures[-1]

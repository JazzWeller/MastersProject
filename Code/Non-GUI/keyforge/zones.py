"""Zone containers."""

from __future__ import annotations

from collections import deque
from typing import List, Optional

from .cards.card import Card


class Deck:
    def __init__(self, cards: Optional[List[Card]] = None):
        self._cards = deque(cards or [])

    def __len__(self):
        return len(self._cards)

    def is_empty(self) -> bool:
        return len(self._cards) == 0

    def draw_top(self) -> Optional[Card]:
        if not self._cards:
            return None
        return self._cards.popleft()

    def peek_top(self) -> Optional[Card]:
        if not self._cards:
            return None
        return self._cards[0]

    def put_on_top(self, card: Card) -> None:
        self._cards.appendleft(card)

    def put_on_bottom(self, card: Card) -> None:
        self._cards.append(card)

    def shuffle_in(self, cards: List[Card], rng) -> None:
        self._cards.extend(cards)
        as_list = list(self._cards)
        rng.shuffle(as_list)
        self._cards = deque(as_list)

    def shuffle(self, rng) -> None:
        as_list = list(self._cards)
        rng.shuffle(as_list)
        self._cards = deque(as_list)

    def remove(self, card: Card) -> bool:
        try:
            self._cards.remove(card)
            return True
        except ValueError:
            return False

    def cards(self) -> List[Card]:
        return list(self._cards)


class DiscardPile:
    def __init__(self):
        self._cards: List[Card] = []

    def __len__(self):
        return len(self._cards)

    def is_empty(self) -> bool:
        return len(self._cards) == 0

    def push(self, card: Card) -> None:
        self._cards.append(card)

    def pop(self) -> Optional[Card]:
        if not self._cards:
            return None
        return self._cards.pop()

    def remove(self, card: Card) -> bool:
        try:
            self._cards.remove(card)
            return True
        except ValueError:
            return False

    def take_all(self) -> List[Card]:
        cards = self._cards
        self._cards = []
        return cards

    def cards(self) -> List[Card]:
        return list(self._cards)


class Hand:
    def __init__(self):
        self._cards: List[Card] = []

    def __len__(self):
        return len(self._cards)

    def add(self, card: Card) -> None:
        self._cards.append(card)

    def remove(self, card: Card) -> bool:
        try:
            self._cards.remove(card)
            return True
        except ValueError:
            return False

    def take_all(self) -> List[Card]:
        cards = self._cards
        self._cards = []
        return cards

    def cards(self) -> List[Card]:
        return list(self._cards)


class Archive:
    def __init__(self):
        self._cards: List[Card] = []

    def __len__(self):
        return len(self._cards)

    def add(self, card: Card) -> None:
        self._cards.append(card)

    def remove(self, card: Card) -> bool:
        try:
            self._cards.remove(card)
            return True
        except ValueError:
            return False

    def take_all(self) -> List[Card]:
        cards = self._cards
        self._cards = []
        return cards

    def cards(self) -> List[Card]:
        return list(self._cards)


class PurgedZone:
    def __init__(self):
        self._cards: List[Card] = []

    def __len__(self):
        return len(self._cards)

    def add(self, card: Card) -> None:
        self._cards.append(card)

    def remove(self, card: Card) -> bool:
        try:
            self._cards.remove(card)
            return True
        except ValueError:
            return False

    def cards(self) -> List[Card]:
        return list(self._cards)


class PlayArea:
    def __init__(self):
        self.creatures: List[Card] = []
        self.artifacts: List[Card] = []

    def all_cards(self) -> List[Card]:
        return list(self.creatures) + list(self.artifacts)

    def add_creature(self, card: Card, flank: Optional[str] = None) -> None:
        if flank == "left":
            self.creatures.insert(0, card)
        else:
            self.creatures.append(card)

    def add_artifact(self, card: Card) -> None:
        self.artifacts.append(card)

    def remove(self, card: Card) -> bool:
        if card in self.creatures:
            self.creatures.remove(card)
            return True
        if card in self.artifacts:
            self.artifacts.remove(card)
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

    def is_flank(self, card: Card) -> bool:
        if getattr(card, "forced_flank", False):
            return True  # Spectral Tunneler: considered a flank creature for the rest of the turn
        if card not in self.creatures:
            return False
        if len(self.creatures) == 1:
            return True
        return card is self.creatures[0] or card is self.creatures[-1]

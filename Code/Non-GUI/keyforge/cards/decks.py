"""The `Deck` model: validation, JSON storage, and random legal decks
(Code/PHASE_2_PLAN.md Milestone D), plus `AllianceDeck` (Milestone F): a
deck whose three house pods are each copied from a different saved deck.

Bundled presets live as JSON files in `Code/Non-GUI/decks/*.json` and are
loaded into `DECKS` at import time (mirroring how `cards/card_data.py` loads
`cota_pool.json`). `build_deck` accepts a preset name, a path to a deck JSON
file, or a `Deck` object directly -- `GameConfig.decks` may hold any mix of
the three. `AllianceDeck` is a `Deck` subclass, so it works everywhere a
`Deck` does (the engine plays it exactly like any other deck) -- the only
addition is `sources`, which names of each pod's origin deck for display."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Union

from ..enums import House
from .card import Card
from .card_data import CARD_DEFS, get_card_def

PODS_PER_DECK = 3
CARDS_PER_POD = 12
ALL_HOUSES = tuple(House)

_PRESET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "decks")

_NAMES_BY_HOUSE: Dict[House, List[str]] = {}
for _name, _cdef in CARD_DEFS.items():
    _NAMES_BY_HOUSE.setdefault(_cdef.house, []).append(_name)


@dataclass
class Deck:
    """A playable deck: exactly one 12-card pod per house, for 3 distinct
    houses out of the 7."""

    name: str
    pods: Dict[House, List[str]] = field(default_factory=dict)

    def houses(self) -> List[House]:
        return sorted(self.pods.keys(), key=lambda h: h.value)

    def all_card_names(self) -> List[str]:
        names: List[str] = []
        for house in self.houses():
            names.extend(self.pods[house])
        return names

    def validate(self) -> None:
        errors = validate_deck(self)
        if errors:
            raise ValueError(f"Invalid deck {self.name!r}: {'; '.join(errors)}")


@dataclass
class AllianceDeck(Deck):
    """A `Deck` assembled by taking one house's pod from each of up to 3
    different saved decks (Milestone F). Engine-wise it's just a `Deck` --
    `sources` (house -> the origin deck's display name) exists only so the
    GUI can label it "Alliance" and show where each pod came from. Pods are
    copied by value at build time, so later edits to a source deck never
    change an alliance deck already built from it."""

    sources: Dict[House, str] = field(default_factory=dict)


def build_alliance_deck(name: str, house_sources: Dict[House, "DeckSource"]) -> AllianceDeck:
    """`house_sources` maps each identity house to a deck source (preset
    name, file path, or `Deck`) to take that house's pod from."""
    pods: Dict[House, List[str]] = {}
    sources: Dict[House, str] = {}
    for house, source in house_sources.items():
        deck = resolve_deck(source)
        pods[house] = list(deck.pods.get(house, []))
        sources[house] = deck.name
    return AllianceDeck(name=name, pods=pods, sources=sources)


def validate_deck(deck: Deck) -> List[str]:
    """Human-readable problems with `deck`, or [] if it's legal: exactly 3
    distinct houses (of the 7), 12 cards per pod, every name in the pool and
    of the right house. Duplicate card names within a pod are legal, as in
    real KeyForge."""
    errors: List[str] = []
    houses = set(deck.pods.keys())
    if len(houses) != PODS_PER_DECK:
        errors.append(f"must have exactly {PODS_PER_DECK} distinct houses, got {sorted(h.value for h in houses)}")
    for house, names in deck.pods.items():
        if len(names) != CARDS_PER_POD:
            errors.append(f"{house.value} pod has {len(names)} cards, needs {CARDS_PER_POD}")
        for name in names:
            cdef = CARD_DEFS.get(name)
            if cdef is None:
                errors.append(f"{name!r} is not in the card pool")
            elif cdef.house != house:
                errors.append(f"{name!r} is a {cdef.house.value} card, not {house.value}")
    return errors


def random_deck(rng: random.Random, name: str = "Random") -> Deck:
    """A random legal deck: 3 random distinct houses, then 12 random cards
    (with replacement) per house."""
    houses = rng.sample(ALL_HOUSES, PODS_PER_DECK)
    pods = {house: [rng.choice(_NAMES_BY_HOUSE[house]) for _ in range(CARDS_PER_POD)] for house in houses}
    return Deck(name=name, pods=pods)


def deck_to_dict(deck: Deck) -> dict:
    data = {"name": deck.name, "pods": {house.value: list(names) for house, names in deck.pods.items()}}
    if isinstance(deck, AllianceDeck):
        data["sources"] = {house.value: label for house, label in deck.sources.items()}
    return data


def deck_from_dict(data: dict) -> Deck:
    pods = {House(house_name): list(names) for house_name, names in data["pods"].items()}
    if "sources" in data:
        sources = {House(house_name): label for house_name, label in data["sources"].items()}
        return AllianceDeck(name=data["name"], pods=pods, sources=sources)
    return Deck(name=data["name"], pods=pods)


def save_deck_json(deck: Deck, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(deck_to_dict(deck), f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_deck_json(path: str) -> Deck:
    with open(path, "r", encoding="utf-8") as f:
        return deck_from_dict(json.load(f))


def _load_presets() -> Dict[str, Deck]:
    presets: Dict[str, Deck] = {}
    if os.path.isdir(_PRESET_DIR):
        for fname in sorted(os.listdir(_PRESET_DIR)):
            if fname.endswith(".json"):
                deck = load_deck_json(os.path.join(_PRESET_DIR, fname))
                presets[deck.name.lower()] = deck
    return presets


DECKS: Dict[str, Deck] = _load_presets()

DeckSource = Union[str, Deck]


def resolve_deck(source: DeckSource) -> Deck:
    """`source` may be a `Deck` object, a bundled preset name, or a path to
    a deck JSON file (in that order of precedence)."""
    if isinstance(source, Deck):
        return source
    key = str(source).lower()
    if key in DECKS:
        return DECKS[key]
    return load_deck_json(source)


def build_deck(source: DeckSource, owner: int) -> List[Card]:
    deck = resolve_deck(source)
    return [Card(get_card_def(name), owner) for name in deck.all_card_names()]


def deck_label(source: DeckSource) -> str:
    """A short display name for a deck source (preset name, file path, or a
    `Deck` object directly) -- for anywhere a deck needs to show up as
    plain text (UI labels, history/database TEXT columns), never for
    resolving actual cards. Tags an `AllianceDeck` so it reads as one
    wherever a deck's name is shown (the menu, history, ...)."""
    if isinstance(source, AllianceDeck):
        return f"{source.name} (Alliance)"
    return source.name if isinstance(source, Deck) else str(source)

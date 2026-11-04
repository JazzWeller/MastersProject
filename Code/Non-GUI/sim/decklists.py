"""Real decks for training and evaluation: the lists `tools/import_decks.py`
writes. A list's `train` and `test` splits never share a deck, so `test` is the
held-out measure of how a network does on decks it never played."""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Tuple

from keyforge.cards.decks import Deck, deck_from_dict

DECKLIST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "keyforge", "cards", "data", "decklists")
_LOADED: Dict[str, Dict[str, List[Deck]]] = {}


def decklist(name: str, split: str) -> List[Deck]:
    """The list's decks in `split`, in the file's order."""
    if name not in _LOADED:
        with open(os.path.join(DECKLIST_DIR, f"{name}.json"), encoding="utf-8") as f:
            rows = json.load(f)["decks"]
        loaded: Dict[str, List[Deck]] = {"train": [], "test": []}
        for row in rows:
            loaded[row["split"]].append(deck_from_dict(row))
        _LOADED[name] = loaded
    return _LOADED[name][split]


def pair_for(name: str, split: str, k: int) -> Tuple[Deck, Deck]:
    """The k-th evaluation pairing: a fixed shuffle of the split taken two at a
    time, so that consecutive pairings use every deck once before any repeats."""
    decks = decklist(name, split)
    order = list(range(len(decks)))
    random.Random(f"pairs|{name}|{split}").shuffle(order)
    return decks[order[2 * k % len(order)]], decks[order[(2 * k + 1) % len(order)]]

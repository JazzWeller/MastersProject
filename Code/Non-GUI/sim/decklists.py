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
_SAS: Dict[str, Dict[str, float]] = {}  # list -> deck name -> SAS rating (Decks of KeyForge)


def decklist(name: str, split: str) -> List[Deck]:
    """The list's decks in `split`, in the file's order."""
    if name not in _LOADED:
        with open(os.path.join(DECKLIST_DIR, f"{name}.json"), encoding="utf-8") as f:
            rows = json.load(f)["decks"]
        loaded: Dict[str, List[Deck]] = {"train": [], "test": []}
        for row in rows:
            loaded[row["split"]].append(deck_from_dict(row))
        _LOADED[name] = loaded
        _SAS[name] = {row["name"]: float(row["sas"]) for row in rows if row.get("sas") is not None}
    return _LOADED[name][split]


def pair_for(name: str, split: str, k: int) -> Tuple[Deck, Deck]:
    """The k-th evaluation pairing: a fixed shuffle of the split taken two at a
    time, so that consecutive pairings use every deck once before any repeats."""
    decks = decklist(name, split)
    order = list(range(len(decks)))
    random.Random(f"pairs|{name}|{split}").shuffle(order)
    return decks[order[2 * k % len(order)]], decks[order[(2 * k + 1) % len(order)]]


def sas_of(name: str, deck_name: str) -> float:
    """A deck's SAS rating, by its name (names are unique within a list)."""
    decklist(name, "train")
    return _SAS[name][deck_name]


def pair_in_gap(name: str, split: str, k: int, lo: float, hi: float) -> Tuple[Deck, Deck]:
    """The k-th evaluation pairing among the split's pairs whose SAS ratings differ by at least `lo` and less than
    `hi`: a fixed shuffle of those pairs, so every player pairing (and a resumed run) meets the same decks."""
    key = (name, split, lo, hi)
    if key not in _GAP_PAIRS:
        decks = decklist(name, split)
        sas = _SAS[name]
        pairs = [(i, j) for i in range(len(decks)) for j in range(i + 1, len(decks))
                 if lo <= abs(sas[decks[i].name] - sas[decks[j].name]) < hi]
        if not pairs:
            raise ValueError(f"no {split} pairs in {name} with a SAS gap in [{lo}, {hi})")
        random.Random(f"gap-pairs|{name}|{split}|{lo}|{hi}").shuffle(pairs)
        _GAP_PAIRS[key] = pairs
    decks = decklist(name, split)
    i, j = _GAP_PAIRS[key][k % len(_GAP_PAIRS[key])]
    return decks[i], decks[j]


_GAP_PAIRS: Dict[tuple, List[Tuple[int, int]]] = {}

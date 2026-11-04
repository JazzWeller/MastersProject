"""Turns a Decks of KeyForge CSV export into a deck list the runs can use.

    python -m tools.import_decks "phase3_combined.csv" phase3

writes `keyforge/cards/data/decklists/<name>.json`: each deck's Master Vault id,
name, SAS rating, split and cards, and nothing else from the export. A card in
a house it isn't printed in is a maverick (`keyforge/cards/decks.py`). Refuses a
deck the engine can't build, rather than dropping it unnoticed.

The split is a function of each deck's id alone, taken per 10-point SAS band so
that train and test cover the same range of strength: a held-out deck stays held
out however the export was sorted, and whatever else is added later.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import re
from typing import Dict, List

from keyforge.cards.decks import Deck, validate_deck
from keyforge.enums import House
from sim.decklists import DECKLIST_DIR

TEST_FRACTION = 1 / 6
CREDIT = "Deck data from Decks of KeyForge, https://decksofkeyforge.com"
_DECK_ID = re.compile(r"/(?:decks|deck-details)/([0-9a-f-]{36})")


def band(sas: float) -> str:
    return "0-50" if sas <= 50 else f"{int((sas - 1) // 10) * 10 + 1}-{int((sas - 1) // 10) * 10 + 10}"


def read_rows(path: str) -> List[dict]:
    """The export's rows as decks, in file order; raises on anything the engine can't build."""
    decks: List[dict] = []
    seen_ids, seen_lists = set(), set()
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            found = _DECK_ID.search(r["Master Vault Link"])
            if found is None:
                raise ValueError(f"{r['Name']!r}: no deck id in {r['Master Vault Link']!r}")
            houses = [h.strip() for h in r["Houses"].split("|")]
            pods: Dict[str, List[str]] = {h: r[f"House {i} Cards"].split("|") for i, h in enumerate(houses, 1)}
            errors = validate_deck(Deck(name=r["Name"], pods={House(h): names for h, names in pods.items()}))
            if errors:
                raise ValueError(f"{r['Name']!r}: {'; '.join(errors)}")
            cards = tuple(sorted((h, n) for h, names in pods.items() for n in names))
            if found.group(1) in seen_ids or cards in seen_lists:
                continue  # the same deck twice (overlapping exports)
            seen_ids.add(found.group(1))
            seen_lists.add(cards)
            decks.append({"id": found.group(1), "name": r["Name"], "sas": float(r["Sas Rating"]), "pods": pods})
    return decks


def split(decks: List[dict]) -> None:
    by_band = collections.defaultdict(list)
    for d in decks:
        by_band[band(d["sas"])].append(d)
    for members in by_band.values():
        members.sort(key=lambda d: hashlib.sha256(d["id"].encode()).hexdigest())
        n_test = round(len(members) * TEST_FRACTION)
        for i, d in enumerate(members):
            d["split"] = "test" if i < n_test else "train"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv")
    parser.add_argument("name")
    args = parser.parse_args()
    decks = read_rows(args.csv)
    split(decks)
    decks.sort(key=lambda d: d["id"])
    os.makedirs(DECKLIST_DIR, exist_ok=True)
    out = os.path.join(DECKLIST_DIR, f"{args.name}.json")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"credit": CREDIT, "decks": decks}, f, ensure_ascii=False, indent=0)
        f.write("\n")
    print(f"{out}: {len(decks)} decks, {collections.Counter(d['split'] for d in decks)}")


if __name__ == "__main__":
    main()

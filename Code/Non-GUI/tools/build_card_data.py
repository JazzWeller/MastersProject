#!/usr/bin/env python3
"""Generates `keyforge/cards/data/cota_pool.json` from `Code/PHASE_2_CARD_POOL.md`
(Dis/Logos/Shadows) and `Code/PHASE_3_CARD_POOL.md` (Brobnar/Mars/Sanctum/Untamed) --
the researched card tables: keyteki CotA.json text cross-checked against the
official Master Vault API, with official errata applied -- see each file's
header for sourcing. No network access at runtime or at build time: this
script only reads the already-researched markdown tables.

Run from `Code/Non-GUI`: `python -m tools.build_card_data`
"""

from __future__ import annotations

import json
import os
import re

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_NON_GUI_DIR = os.path.dirname(_THIS_DIR)
_CODE_DIR = os.path.dirname(_NON_GUI_DIR)
OUT_JSON = os.path.join(_NON_GUI_DIR, "keyforge", "cards", "data", "cota_pool.json")

# Each pool file lists its houses as "## <House>" headers. Phase 2's table has
# no Armor column (no pool card in it has armor); Phase 3's table adds one
# between Pow and AE. Column presence is detected from each file's own header
# row, so either layout parses correctly.
POOL_FILES = [
    (os.path.join(_CODE_DIR, "PHASE_2_CARD_POOL.md"), {"Dis": 54, "Logos": 53, "Shadows": 52}),
    (os.path.join(_CODE_DIR, "PHASE_3_CARD_POOL.md"), {"Brobnar": 52, "Mars": 52, "Sanctum": 55, "Untamed": 52}),
]

_COLUMN_NAMES = {
    "#": "number",
    "Card": "name",
    "Type": "type",
    "Pow": "power",
    "Armor": "armor",
    "Æ": "aember_on_play",
    "Traits": "traits",
    "Keywords": "keywords",
    "Canonical text": "text",
    "Errata": "errata",
    "Art": "image",
    "P1": "phase1",
}


def _split_row(line: str):
    line = line.strip()
    assert line.startswith("|") and line.endswith("|"), f"not a table row: {line!r}"
    cells = line[1:-1].split("|")
    # The source tables use a literal vertical-tab (U+000B) as a soft line
    # break within a cell, for cards whose text spans multiple printed
    # lines (e.g. a keyword reminder line followed by the next ability).
    return [c.strip().replace("\x0b", "\n") for c in cells]


def _parse_list(cell: str, capitalize: bool) -> list:
    if not cell:
        return []
    parts = [p.strip() for p in cell.split(",") if p.strip()]
    return [p.capitalize() if capitalize else p.lower() for p in parts]


def _parse_int(cell: str):
    cell = cell.strip()
    return int(cell) if cell else None


def _clean_text(s: str) -> str:
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    return s.strip()


def _parse_art(cell: str) -> str:
    m = re.search(r"`([^`]+)`", cell)
    assert m, f"couldn't find a backtick-quoted art path in {cell!r}"
    return m.group(1)


def parse_pool(md_path: str) -> list:
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cards = []
    current_house = None
    columns = None  # list of field names, in this file's column order
    for raw_line in lines:
        line = raw_line.rstrip("\n")
        stripped = line.strip()
        if stripped.startswith("## "):
            current_house = stripped[3:].strip()
            columns = None
            continue
        if not stripped.startswith("|") or current_house is None:
            continue
        cells = _split_row(stripped)
        if columns is None:
            # This must be the header row for the table that just started.
            assert cells[0] == "#", f"expected header row after '## {current_house}', got {cells!r}"
            columns = [_COLUMN_NAMES[c] for c in cells]
            continue
        if set(c.strip("-: ") for c in cells) == {""}:
            continue  # the |---|---|... separator row
        row = dict(zip(columns, cells))
        if not row["number"].isdigit():
            continue  # stray non-data row

        cards.append(
            {
                "number": int(row["number"]),
                "name": row["name"],
                "house": current_house,
                "type": row["type"].strip().capitalize(),
                "power": _parse_int(row["power"]),
                "armor": _parse_int(row.get("armor", "")) or 0,
                "aember_on_play": _parse_int(row["aember_on_play"]) or 0,
                "traits": _parse_list(row["traits"], capitalize=True),
                "keywords": _parse_list(row["keywords"], capitalize=False),
                "text": _clean_text(row["text"]),
                "errata": _clean_text(row["errata"]) if row["errata"] else None,
                "image": _parse_art(row["image"]),
                "phase1": row["phase1"].strip() == "*",
            }
        )
    return cards


def main() -> None:
    cards = []
    for md_path, expected in POOL_FILES:
        file_cards = parse_pool(md_path)
        by_house = {}
        for c in file_cards:
            by_house.setdefault(c["house"], []).append(c)
        for house, count in expected.items():
            actual = len(by_house.get(house, []))
            assert actual == count, f"{os.path.basename(md_path)} {house}: parsed {actual} cards, expected {count}"
        assert len(file_cards) == sum(expected.values()), (
            f"{os.path.basename(md_path)}: parsed {len(file_cards)} cards, expected {sum(expected.values())}"
        )
        cards.extend(file_cards)

    expected_total = sum(sum(exp.values()) for _, exp in POOL_FILES)
    assert len(cards) == expected_total, f"parsed {len(cards)} cards total, expected {expected_total}"
    names = [c["name"] for c in cards]
    assert len(names) == len(set(names)), "duplicate card names in the pool"

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8", newline="\n") as f:
        json.dump(cards, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Wrote {len(cards)} cards to {OUT_JSON}")


if __name__ == "__main__":
    main()

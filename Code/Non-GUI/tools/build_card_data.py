#!/usr/bin/env python3
"""Generates `keyforge/cards/data/cota_pool.json` from `Code/PHASE_2_CARD_POOL.md`
(the researched card table: keyteki CotA.json text cross-checked against the
official Master Vault API, with official errata applied -- see that file's
header for sourcing). No network access at runtime or at build time: this
script only reads the already-researched markdown table.

Run from `Code/Non-GUI`: `python -m tools.build_card_data`
"""

from __future__ import annotations

import json
import os
import re

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_NON_GUI_DIR = os.path.dirname(_THIS_DIR)
_CODE_DIR = os.path.dirname(_NON_GUI_DIR)
POOL_MD = os.path.join(_CODE_DIR, "PHASE_2_CARD_POOL.md")
OUT_JSON = os.path.join(_NON_GUI_DIR, "keyforge", "cards", "data", "cota_pool.json")

_HOUSE_HEADERS = {"## Dis": "Dis", "## Logos": "Logos", "## Shadows": "Shadows"}
_EXPECTED_COLUMNS = 11  # #, Card, Type, Pow, AE, Traits, Keywords, Canonical text, Errata, Art, P1


def _split_row(line: str):
    line = line.strip()
    assert line.startswith("|") and line.endswith("|"), f"not a table row: {line!r}"
    cells = line[1:-1].split("|")
    # The source table uses a literal vertical-tab (U+000B) as a soft line
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
    for raw_line in lines:
        line = raw_line.rstrip("\n")
        stripped = line.strip()
        if stripped in _HOUSE_HEADERS:
            current_house = _HOUSE_HEADERS[stripped]
            continue
        if not stripped.startswith("|") or current_house is None:
            continue
        if stripped.startswith("|---") or "---" in stripped.split("|")[1]:
            continue
        cells = _split_row(stripped)
        if len(cells) != _EXPECTED_COLUMNS:
            continue  # not a data row (e.g. a stray '|' in prose, or the header row)
        number_cell = cells[0]
        if not number_cell.isdigit():
            continue  # header row ("#", "Card", ...)

        number, name, ctype, pow_, ae, traits, keywords, text, errata, art, p1 = cells
        cards.append(
            {
                "number": int(number),
                "name": name,
                "house": current_house,
                "type": ctype.strip().capitalize(),
                "power": _parse_int(pow_),
                "armor": 0,  # confirmed in PHASE_2_CARD_POOL.md notes: no pool card has armor
                "aember_on_play": _parse_int(ae) or 0,
                "traits": _parse_list(traits, capitalize=True),
                "keywords": _parse_list(keywords, capitalize=False),
                "text": _clean_text(text),
                "errata": _clean_text(errata) if errata else None,
                "image": _parse_art(art),
                "phase1": p1.strip() == "*",
            }
        )
    return cards


def main() -> None:
    cards = parse_pool(POOL_MD)
    by_house = {}
    for c in cards:
        by_house.setdefault(c["house"], []).append(c)
    expected = {"Dis": 54, "Logos": 53, "Shadows": 52}
    for house, count in expected.items():
        actual = len(by_house.get(house, []))
        assert actual == count, f"{house}: parsed {actual} cards, expected {count}"
    assert len(cards) == 159, f"parsed {len(cards)} cards total, expected 159"
    names = [c["name"] for c in cards]
    assert len(names) == len(set(names)), "duplicate card names in the pool"

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8", newline="\n") as f:
        json.dump(cards, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Wrote {len(cards)} cards to {OUT_JSON}")


if __name__ == "__main__":
    main()

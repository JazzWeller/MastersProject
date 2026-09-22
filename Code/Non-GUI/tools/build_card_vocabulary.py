#!/usr/bin/env python3
"""Updates `keyforge/cards/data/card_vocabulary.json` (Agent Interface Plan,
Milestone F): every card name in the current pool gets a stable integer id.

Append-only by construction: existing entries are never touched, and any
card name found in `CARD_DEFS` but not yet in the vocabulary file is
appended in sorted-name order, starting at the current max id + 1. Safe to
run repeatedly -- a pool with no new cards leaves the file byte-identical.

Run from `Code/Non-GUI`: `python -m tools.build_card_vocabulary`
"""

from __future__ import annotations

import json

from keyforge.cards.card_data import CARD_DEFS
from keyforge.cards.vocabulary import VOCAB_JSON_PATH, _load


def build() -> int:
    """Returns the number of newly-added entries."""
    vocab = _load()
    next_id = (max(vocab.values()) + 1) if vocab else 0
    missing = sorted(name for name in CARD_DEFS if name not in vocab)
    for name in missing:
        vocab[name] = next_id
        next_id += 1
    with open(VOCAB_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False, sort_keys=False)
        f.write("\n")
    return len(missing)


def main() -> None:
    added = build()
    print(f"card_vocabulary.json: {added} new card(s) added.")


if __name__ == "__main__":
    main()

"""Versioned, append-only card vocabulary: card name -> a stable integer id
for fixed-vocabulary encoding heads (Agent Interface Plan, Milestone F).

Loaded from `cards/data/card_vocabulary.json` (built and updated by
`tools/build_card_vocabulary.py`), a flat `{name: id}` mapping. New cards
are always appended at the current max id + 1 -- an id, once assigned, is
never reused or reassigned, so a network trained against an older
vocabulary file stays loadable after a later card pool adds more cards.
`VOCAB_HASH` (a hash of the file's own bytes) should be stamped into every
checkpoint alongside `keyforge.version.ENGINE_VERSION`, the same way a
replay record is stamped, so a checkpoint trained against a stale
vocabulary is caught outright instead of silently mis-decoding ids.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, Optional

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
VOCAB_JSON_PATH = os.path.join(_DATA_DIR, "card_vocabulary.json")


def _load() -> Dict[str, int]:
    if not os.path.isfile(VOCAB_JSON_PATH):
        return {}
    with open(VOCAB_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _hash() -> str:
    if not os.path.isfile(VOCAB_JSON_PATH):
        return hashlib.sha256(b"").hexdigest()
    with open(VOCAB_JSON_PATH, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


CARD_VOCAB: Dict[str, int] = _load()
CARD_VOCAB_REVERSE: Dict[int, str] = {v: k for k, v in CARD_VOCAB.items()}
VOCAB_HASH: str = _hash()


def card_id(name: str) -> int:
    """Raises `KeyError` for a card not yet in the vocabulary -- run
    `python -m tools.build_card_vocabulary` after adding new cards to the
    pool, before training or encoding against them."""
    return CARD_VOCAB[name]


def card_name(vocab_id: int) -> Optional[str]:
    return CARD_VOCAB_REVERSE.get(vocab_id)

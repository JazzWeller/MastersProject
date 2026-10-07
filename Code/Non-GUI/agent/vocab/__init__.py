"""The v2 vocabularies (Agent Observation Plan, Milestone O5): append-only
lists that replace v1's hashing (traits, mode names, trigger events,
effect variables and kinds, log event kinds, ability kinds, cleanup
operations, zone codes, the resolution stack's routines, sites, roles and
operations, journal ops, destined zones, verbs).

Each is `agent/vocab/<name>.json`: {"name", "version", "entries"}. An
entry's id is its index + 1 (0 is padding / none), so appending never
moves an id. `tools/build_vocab_v2.py` builds and extends them from the
card data and the engine's source; `VOCAB_V2_HASH` goes into every v2
checkpoint and shard.

**An unknown entry raises** (`Vocab.id`): a new card or engine change must
extend the vocabulary; nothing is silently mapped to "other".
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List

_DIR = os.path.dirname(os.path.abspath(__file__))

NAMES = (
    "traits", "keywords", "modes", "events", "effect_variables", "effect_kinds", "effect_ops", "log_kinds",
    "ability_kinds", "cleanup_ops", "zone_kinds", "routines", "sites", "roles", "op_names", "journal_ops",
    "destined_zones", "verbs", "steps", "option_tags",
)


def op_name(op) -> str:
    """A reachable operation's entry in `op_names` (the compiler's static
    table, keyforge/resolution.py): its kind and callee."""
    return op[0] if op[0] in ("loop", "decision") else f"{op[0]}:{op[1]}"


class UnknownVocabEntry(KeyError):
    pass


class Vocab:
    __slots__ = ("name", "entries", "_ids")

    def __init__(self, name: str, entries: List[str]):
        self.name = name
        self.entries = list(entries)
        self._ids: Dict[str, int] = {e: i + 1 for i, e in enumerate(self.entries)}

    def __len__(self):
        return len(self.entries) + 1  # with padding

    def id(self, entry) -> int:
        if entry is None:
            return 0
        i = self._ids.get(entry)
        if i is None:
            raise UnknownVocabEntry(f"{self.name}: {entry!r} is not in the vocabulary -- extend it "
                                    f"(python -m tools.build_vocab_v2)")
        return i

    def get(self, entry, default: int = 0) -> int:
        return self._ids.get(entry, default)


def path(name: str) -> str:
    return os.path.join(_DIR, f"{name}.json")


def load(name: str) -> Vocab:
    with open(path(name), encoding="utf-8") as f:
        data = json.load(f)
    return Vocab(name, data["entries"])


def load_all() -> Dict[str, Vocab]:
    return {n: load(n) for n in NAMES if os.path.exists(path(n))}


def vocab_hash() -> str:
    h = hashlib.sha256()
    for n in NAMES:
        p = path(n)
        if os.path.exists(p):
            with open(p, "rb") as f:
                h.update(n.encode())
                h.update(f.read())
    return h.hexdigest()

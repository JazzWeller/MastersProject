"""Static card meaning for v2 (Agent Observation Plan, Milestone O5): three
representations, all built, of what each card *does* -- a pure function
of the card registry, computed once.

- **Attributes** (`attribute_row`): v1's static row, with traits from the
  trait vocabulary (multi-hot) instead of crc32 buckets.
- **Parsed text** (`text_rows`): each sentence of the card's text as an
  ability row -- (trigger, verb, amount, scope, condition flags) --
  pooled by a small set encoder on the network side.
- **Engine signatures** (`signature_row`): an `ast` analysis of the card's
  effect functions in `effects/named/*.py` (and the helpers they call in
  the same module): which `steps.*`, `game.*` and `generic.*` calls they
  make (multi-hot over the `steps` vocabulary), the lasting effects they
  create (class and variable / event / kind), the engine state attributes
  they write (the coverage registry's names), and the sum of the literal
  numbers passed. Exact about what the engine does.
- **Text embedding** (`text_embedding`): a pretrained sentence encoder's
  embedding of the text, read from `agent/vocab/text_embedding.json` only
  if that table exists (`tools/build_text_embedding.py` makes it; the
  dependency and the model download are optional).
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import re
import textwrap
from array import array
from typing import Dict, List, Optional, Tuple

from keyforge.cards.card_data import CARD_DEFS
from keyforge.cards.vocabulary import CARD_VOCAB

from .features_v1 import static_row as _v1_static_row
from .spec import STATIC
from .vocab import load

TRIGGERS = ("play", "reap", "fight", "action", "omni", "destroyed", "before fight", "after fight", "after reap",
            "leaves play", "each time", "at the start of", "at the end of", "while", "this creature", "your",
            "constant")
VERBS = ("destroy", "deal", "damage", "capture", "steal", "gain", "lose", "draw", "discard", "archive", "purge",
         "ready", "exhaust", "stun", "heal", "return", "shuffle", "play", "use", "fight", "reap", "forge", "key",
         "control", "power", "swap", "move", "search", "reveal", "look", "put", "choose", "cannot", "may", "enrage",
         "ward", "armor", "elusive", "skirmish", "taunt", "poison", "assault", "hazardous")
SCOPES = ("each creature", "each enemy creature", "each friendly creature", "an enemy creature",
          "a friendly creature", "a creature", "each artifact", "an artifact", "your opponent", "you", "each player",
          "this creature", "its neighbors", "a flank creature", "the most powerful", "the least powerful", "other")
CONDITIONS = ("if", "for each", "instead", "unless", "may", "random", "cannot", "this turn", "next turn")

_EMBED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocab", "text_embedding.json")
_TRAITS = load("traits")
_STEPS = load("steps")

TEXT_ROW_WIDTH = 4 + len(CONDITIONS)  # trigger id, verb id, amount, scope id, condition flags


def attribute_row(cdef) -> array:
    """v1's static row with the hashed trait buckets replaced by a trait
    multi-hot (vocabulary order)."""
    v1 = _v1_static_row(cdef)
    lo, w = STATIC.span("traits")
    row = array("f", list(v1[:lo]) + list(v1[lo + w:]))
    traits = array("f", bytes(4 * len(_TRAITS.entries)))
    for t in cdef.tags:
        traits[_TRAITS.id(t) - 1] = 1.0
    row.extend(traits)
    return row


_SENTENCE = re.compile(r"(?<=[.!])\s+")
_NUMBER = re.compile(r"(\d+)")


def text_rows(cdef) -> List[Tuple[int, int, float, int, Tuple[int, ...]]]:
    """The card's text as ability rows: (trigger id, verb id, amount, scope
    id, condition flags), ids 1-based into TRIGGERS / VERBS / SCOPES (0 =
    none)."""
    text = (cdef.text or "").replace("Æ", " aember").replace("�", " aember")
    rows = []
    trigger = 0
    for sentence in _SENTENCE.split(text):
        low = sentence.lower().strip()
        if not low:
            continue
        head = low.split(":", 1)[0] if ":" in low[:40] else ""
        for part in re.split(r"/", head):
            part = part.strip()
            if part in TRIGGERS:
                trigger = TRIGGERS.index(part) + 1
        if not head:
            for k, t in enumerate(TRIGGERS):
                if low.startswith(t):
                    trigger = k + 1
                    break
        verb = 0
        for k, v in enumerate(VERBS):
            if re.search(r"\b" + re.escape(v), low):
                verb = k + 1
                break
        m = _NUMBER.search(low)
        amount = float(m.group(1)) if m else 0.0
        scope = 0
        for k, sc in enumerate(SCOPES):
            if sc in low:
                scope = k + 1
                break
        conditions = tuple(int(re.search(r"\b" + re.escape(c) + r"\b", low) is not None) for c in CONDITIONS)
        rows.append((trigger, verb, amount, scope, conditions))
    return rows


def _effect_functions(cdef):
    hooks = (cdef.on_play, cdef.on_reap, cdef.on_fight, cdef.on_action, cdef.on_omni, cdef.on_destroyed,
             cdef.on_destroyed_fighting, cdef.on_before_fight, cdef.register_passive, cdef.unregister_passive)
    return [h for h in hooks if h is not None]


_EFFECT_CLASSES = {"DurationEffect": ("effect_variables", 4), "TriggerEffect": ("events", 2),
                   "InsteadEffect": ("effect_kinds", 2), "ModifierEffect": ("effect_kinds", 2)}
_EFFECT_VOCABS = {n: load(n) for n in ("effect_variables", "events", "effect_kinds")}


def _state_attributes() -> List[str]:
    from .state_registry import REGISTRY

    names = set()
    for cls in ("Game", "Player", "Card", "CreatureType"):
        names.update(REGISTRY[cls])
    return sorted(names)


_ATTRS = _state_attributes()
_ATTR_INDEX = {a: i for i, a in enumerate(_ATTRS)}


def _calls(fn, seen) -> Tuple[set, float]:
    """The steps/game/generic calls `fn` makes (and, one module deep, the
    module's own helpers it calls), the lasting effects it creates (class,
    and its variable / event / kind), and the literal numbers passed."""
    names, amount = set(), 0.0
    code = getattr(fn, "__code__", None)
    if code is None or code in seen:
        return names, amount
    seen.add(code)
    try:
        src = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError):
        return names, amount
    module = inspect.getmodule(fn)
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Attribute) and t.attr in _ATTR_INDEX:
                    names.add(("attr", t.attr))
            continue
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute)
                and node.value.attr == "extra_triggers" and isinstance(node.slice, ast.Constant)):
            names.add(("events", node.slice.value))  # host.extra_triggers["after_reap"]
            continue
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if module is not None:
            # a module function handed over rather than called (an ability
            # granted to a host): what it does counts too
            for arg in node.args:
                if isinstance(arg, ast.Name):
                    helper = getattr(module, arg.id, None)
                    if callable(helper) and getattr(helper, "__module__", None) == module.__name__:
                        n2, a2 = _calls(helper, seen)
                        names |= n2
                        amount += a2
        if (isinstance(f, ast.Attribute) and f.attr in ("setdefault", "get") and isinstance(f.value, ast.Attribute)
                and f.value.attr == "extra_triggers" and node.args and isinstance(node.args[0], ast.Constant)):
            # an ability granted to the host (an upgrade): its hook, as an event
            names.add(("events", node.args[0].value))
            continue
        if isinstance(f, ast.Name) and f.id in _EFFECT_CLASSES:
            names.add(("class", f.id))
            vocab, index = _EFFECT_CLASSES[f.id]
            if len(node.args) > index and isinstance(node.args[index], ast.Constant):
                names.add((vocab, node.args[index].value))
            continue
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in ("steps", "game", "generic"):
            key = f"{f.value.id}.{f.attr}"
            if key in _STEPS._ids:
                names.add(key)
            for a in node.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, (int, float)) and not isinstance(a.value, bool):
                    amount += a.value
        elif isinstance(f, ast.Name) and module is not None:
            helper = getattr(module, f.id, None)
            if callable(helper) and getattr(helper, "__module__", None) == module.__name__:
                n2, a2 = _calls(helper, seen)
                names |= n2
                amount += a2
    return names, amount


SIGNATURE_WIDTH = (len(_STEPS.entries) + len(_EFFECT_CLASSES)
                   + sum(len(v.entries) for v in _EFFECT_VOCABS.values()) + len(_ATTRS) + 1)


def signature_row(cdef) -> array:
    """Multi-hot over the `steps` vocabulary, the effect classes created and
    their variables / events / kinds, then the literal amount sum."""
    names, amount = set(), 0.0
    seen = set()
    for fn in _effect_functions(cdef):
        n, a = _calls(fn, seen)
        names |= n
        amount += a
    row = array("f", bytes(4 * SIGNATURE_WIDTH))
    classes = list(_EFFECT_CLASSES)
    base_cls = len(_STEPS.entries)
    base_vocab = base_cls + len(classes)
    offsets = {}
    at = base_vocab
    for vname, v in _EFFECT_VOCABS.items():
        offsets[vname] = at
        at += len(v.entries)
    for key in names:
        if isinstance(key, tuple):
            kind, value = key
            if kind == "class":
                row[base_cls + classes.index(value)] = 1.0
            elif kind == "attr":
                row[at + _ATTR_INDEX[value]] = 1.0
            elif isinstance(value, str) and value in _EFFECT_VOCABS[kind]._ids:
                row[offsets[kind] + _EFFECT_VOCABS[kind].id(value) - 1] = 1.0
        else:
            row[_STEPS.id(key) - 1] = 1.0
    row[SIGNATURE_WIDTH - 1] = amount / 10.0
    return row


def text_embedding() -> Optional[Dict[str, List[float]]]:
    """name -> embedding, if the table has been built; else None."""
    if not os.path.exists(_EMBED_PATH):
        return None
    with open(_EMBED_PATH, encoding="utf-8") as f:
        return json.load(f)["embeddings"]


_TABLES = None


def tables():
    """(attributes, text rows, signatures, embedding or None), each indexed
    by card vocabulary id (row 0 unused)."""
    global _TABLES
    if _TABLES is None:
        size = max(CARD_VOCAB.values()) + 1
        attrs: List[Optional[array]] = [None] * size
        texts: List[list] = [[] for _ in range(size)]
        sigs: List[Optional[array]] = [None] * size
        for name, vid in CARD_VOCAB.items():
            cdef = CARD_DEFS.get(name)
            if cdef is None:
                continue
            attrs[vid] = attribute_row(cdef)
            texts[vid] = text_rows(cdef)
            sigs[vid] = signature_row(cdef)
        _TABLES = (attrs, texts, sigs, text_embedding())
    return _TABLES

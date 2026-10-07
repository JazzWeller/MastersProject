"""Serializing a game at any decision (Agent Observation Plan, Part R, R6):
`Game.snapshot() -> bytes` and `Game.restore(bytes) -> Game`.

Canonical, versioned and pickle-free. A snapshot is JSON: a stamp (format
version, engine version, rules hash), the root object, and a table of every
object the game reaches, each written once and referred to by its index
(`{"$": i}`), so whatever is shared stays shared -- a list a frame holds and
the zone it came from, a cell a frame and a closure both hold. Objects are
numbered in the order a fixed traversal from the game first meets them, so
two games in the same state give the same bytes.

What can appear is a closed list (`_CLASSES` and the special cases in
`_Writer.value`); anything else is an error (I5), never a silent omission:

- plain values (None, bool, int, float, str) as themselves; tuples,
  frozensets and ranges as values; enum members by class and value; a card
  definition by its name; a module-level function by module and qualified
  name; the frame-slot sentinel;
- lists, dicts (in insertion order, keys of any type), sets, deques and
  defaultdicts as objects;
- the engine's own classes, by their attributes;
- a nested function or lambda as the path to its code object plus its
  defaults and closure cells; a bound method as its function and `self`;
- a compiled frame as its routine id, `pc` and locals; the machine as its
  stack; a `random.Random` by its state.

Restoring needs the same rules hash: a frame's `pc` is only meaningful for
the routine it was compiled into. A native game can be snapshot only at a
boundary decision, as it can be copied only there.
"""

from __future__ import annotations

import collections
import enum
import importlib
import itertools
import json
import random
import types
from typing import Any, Dict, List, Tuple

from . import vm
from .actions import DiscardCard, EndTurn, Fight, PlayCard, Reap, UseAction, UseOmni
from .cards import decks as _decks
from .cards.card import ActionType, ArtifactType, Card, CardDef, CreatureType, UpgradeType
from .cards.card_data import get_card_def
from .config import GameConfig
from .decision import Decision
from .effects.effect_object import ActiveEffectList, DurationEffect, InsteadEffect, ModifierEffect, TriggerEffect
from .journal import Journal
from .log import GameLog, LogEvent
from .player import Player
from .version import ENGINE_VERSION, RULES_HASH
from .zones import Archive, Deck, DiscardPile, Hand, PlayArea, PurgedZone

SNAPSHOT_VERSION = 1

# Objects written by their attributes: name -> class.
_CLASSES: Dict[str, type] = {c.__name__: c for c in (
    Card, CreatureType, UpgradeType, ArtifactType, ActionType, Player, Deck, Hand, DiscardPile, Archive, PurgedZone,
    PlayArea, ActiveEffectList, DurationEffect, TriggerEffect, InsteadEffect, ModifierEffect, GameLog, LogEvent, Decision,
    PlayCard, DiscardCard, UseAction, UseOmni, Reap, Fight, EndTurn, GameConfig, Journal,
)}
_CLASSES["decks.Deck"] = _decks.Deck
_CLASSES["decks.AllianceDeck"] = _decks.AllianceDeck
_NAME_OF = {c: n for n, c in _CLASSES.items()}

# Attributes derived from others, rebuilt on restore rather than written:
# `ActiveEffectList`'s index keeps emptied buckets in one game and not in its
# copy, which would make equal states give different bytes.
_DERIVED = {ActiveEffectList: ("_duration_by_key",)}

# Game attributes a snapshot leaves out: caches, rebuilt on demand.
_GAME_SKIPPED = frozenset({"_fork_snapshot", "_container_index", "_driver", "_machine", "_projectors", "_trackers", "_trackers2", "_chance", "_history",
                           # this object's own submission history, not game state (see Game.copy)
                           "choice_log"})


class SnapshotError(ValueError):
    pass


def _attrs(obj) -> List[Tuple[str, Any]]:
    """An object's attributes, in a fixed order: its `__slots__` (all
    classes of its MRO), then its `__dict__`."""
    out = []
    for cls in reversed(type(obj).__mro__):
        for name in cls.__dict__.get("__slots__", ()):
            if name in ("__dict__", "__weakref__"):
                continue
            if hasattr(obj, name):
                out.append((name, getattr(obj, name)))
    d = getattr(obj, "__dict__", None)
    if d is not None:
        out.extend(sorted(d.items()))  # by name: how an object was built doesn't show
    return out


# ------------------------------------------------------------- code refs ----


def _code_path(code: types.CodeType, module_name: str) -> List[Any]:
    """`[module, top-level qualname, (name, first line), ...]`: how to find
    a nested function's code object again -- the top-level function by
    name, then each enclosing code object's constants."""
    qual = code.co_qualname if hasattr(code, "co_qualname") else code.co_name
    parts = qual.split(".<locals>.")
    module = importlib.import_module(module_name)
    top = _resolve_attr(module, parts[0])
    chain = []
    cur = getattr(top, "__func__", top).__code__
    for i, name in enumerate(parts[1:]):
        last = i == len(parts) - 2
        cands = [c for c in cur.co_consts if isinstance(c, types.CodeType) and c.co_name == name.split(".")[-1]]
        if last:
            cands = [c for c in cands if c is code]
        if len(cands) != 1:
            raise SnapshotError(f"can't locate code {qual!r} in {module_name}")
        chain.append([cands[0].co_name, cands[0].co_firstlineno])
        cur = cands[0]
    return [module_name, parts[0]] + chain


def _code_from_path(path: List[Any]) -> types.CodeType:
    module = importlib.import_module(path[0])
    top = _resolve_attr(module, path[1])
    code = getattr(top, "__func__", top).__code__
    for name, line in path[2:]:
        cands = [c for c in code.co_consts if isinstance(c, types.CodeType) and c.co_name == name and c.co_firstlineno == line]
        if len(cands) != 1:
            raise SnapshotError(f"no nested code {name!r} at line {line} under {path[:2]}")
        code = cands[0]
    return code


def _resolve_attr(module, qualname: str):
    obj = module
    for part in qualname.split("."):
        obj = obj.__dict__[part] if hasattr(obj, "__dict__") and part in obj.__dict__ else getattr(obj, part)
    return obj


def _is_top_level(fn) -> bool:
    try:
        module = importlib.import_module(fn.__module__)
        return getattr(_resolve_attr(module, fn.__qualname__), "__func__", None) is fn or _resolve_attr(module, fn.__qualname__) is fn
    except Exception:  # noqa: BLE001 -- anything unresolvable is not a top-level function
        return False


# ----------------------------------------------------------------- write ----


class _Writer:
    def __init__(self):
        self.table: List[Any] = []
        self.index: Dict[int, int] = {}
        self.keep: List[Any] = []  # keeps every written object alive, so ids stay unique

    def ref(self, obj, build) -> dict:
        i = self.index.get(id(obj))
        if i is None:
            i = len(self.table)
            self.index[id(obj)] = i
            self.keep.append(obj)
            self.table.append(None)
            self.table[i] = build()
        return {"$": i}

    def value(self, x) -> Any:
        t = type(x)
        if x is None or t in (bool, int, str):
            return x
        if t is float:
            return {"f": repr(x)}
        if t is tuple:
            return {"t": [self.value(v) for v in x]}
        if t is frozenset:
            return {"fs": sorted((self.value(v) for v in x), key=lambda v: json.dumps(v, sort_keys=True))}
        if t is range:
            return {"r": [x.start, x.stop, x.step]}
        if isinstance(x, enum.Enum):
            return {"e": f"{t.__module__}:{t.__qualname__}", "v": self.value(x.value)}
        if t is CardDef:
            return {"cd": x.name}
        if x is vm.UNBOUND:
            return {"u": 0}
        if t is types.FunctionType:
            if x.__closure__ is None and not x.__defaults__ and not x.__kwdefaults__ and _is_top_level(x):
                return {"fn": [x.__module__, x.__qualname__]}
            return self.ref(x, lambda: {"k": "function", "code": _code_path(x.__code__, x.__module__),
                                        "globals": x.__module__,
                                        "closure": [self.value(c) for c in (x.__closure__ or ())],
                                        "defaults": self.value(x.__defaults__) if x.__defaults__ else None,
                                        "kwdefaults": self.value(x.__kwdefaults__) if x.__kwdefaults__ else None})
        if t is types.MethodType:
            return {"m": [self.value(x.__func__), self.value(x.__self__)]}
        if t is types.CellType:
            def cell():
                try:
                    return {"k": "cell", "v": self.value(x.cell_contents)}
                except ValueError:
                    return {"k": "cell"}
            return self.ref(x, cell)
        if t is list:
            return self.ref(x, lambda: {"k": "list", "v": [self.value(v) for v in x]})
        if t is dict:
            return self.ref(x, lambda: {"k": "dict", "v": [[self.value(k), self.value(v)] for k, v in x.items()]})
        if t is collections.defaultdict:
            if x.default_factory is not list:
                raise SnapshotError(f"a defaultdict of {x.default_factory!r}")
            return self.ref(x, lambda: {"k": "defaultdict_list", "v": [[self.value(k), self.value(v)] for k, v in x.items()]})
        if t is set:
            return self.ref(x, lambda: {"k": "set", "v": sorted((self.value(v) for v in x), key=lambda v: json.dumps(v, sort_keys=True))})
        if t is collections.deque:
            return self.ref(x, lambda: {"k": "deque", "v": [self.value(v) for v in x]})
        if t is random.Random:
            return self.ref(x, lambda: {"k": "random", "v": self.value(x.getstate())})
        if t is itertools.count:
            # its next value, from its repr ("count(73)"): Python 3.14 drops
            # itertools' pickling support
            text = repr(x)
            if not (text.startswith("count(") and text.endswith(")") and "," not in text):
                raise SnapshotError(f"an itertools.count with a step: {text}")
            return {"count": int(text[len("count("):-1])}
        if t is vm.Frame:
            return self.ref(x, lambda: {"k": "frame", "routine": x.routine.rid, "pc": x.pc, "L": [self.value(v) for v in x.L]})
        if t is vm.Machine:
            if not x.copyable:
                raise SnapshotError("code that isn't compiled is suspended (a NativeFrame)")
            return self.ref(x, lambda: {"k": "machine", "stack": [self.value(f) for f in x.stack]})
        name = _NAME_OF.get(t)
        if name is not None:
            skip = _DERIVED.get(t, ())
            return self.ref(x, lambda: {"k": "obj", "c": name, "a": [[k, self.value(v)] for k, v in _attrs(x) if k not in skip]})
        from .game import Game

        if t is Game:
            return self.ref(x, lambda: {"k": "game", "a": [[k, self.value(v)] for k, v in sorted(vars(x).items()) if k not in _GAME_SKIPPED],
                                        "machine": self.value(x._machine) if x._machine is not None else None})
        raise SnapshotError(f"can't snapshot a {t.__module__}.{t.__qualname__}: {x!r}")


def snapshot(game) -> bytes:
    if not game.is_over and game.execution != "compiled" and not game.copy_anywhere:
        raise SnapshotError("a native game can only be snapshot at a boundary decision (or run it compiled)")
    game.journal.settle()  # its decision records in data form, not live decisions
    w = _Writer()
    root = w.value(game)
    doc = {"snapshot_version": SNAPSHOT_VERSION, "engine_version": ENGINE_VERSION, "rules_hash": RULES_HASH,
           "root": root, "objects": w.table}
    return json.dumps(doc, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


# ------------------------------------------------------------------ read ----


class _Reader:
    def __init__(self, table: List[dict]):
        self.table = table
        self.objs: List[Any] = [None] * len(table)
        self.built = [False] * len(table)

    def shell(self, i: int):
        """Phase 1: an empty object of the right type for every entry that
        isn't a function (a function needs its cells, which exist after
        this phase)."""
        rec = self.table[i]
        k = rec["k"]
        if k == "list":
            return []
        if k == "dict":
            return {}
        if k == "defaultdict_list":
            return collections.defaultdict(list)
        if k == "set":
            return set()
        if k == "deque":
            return collections.deque()
        if k == "cell":
            return types.CellType()
        if k == "random":
            return random.Random()
        if k == "frame":
            return vm.Frame(vm.ROUTINES_BY_ID[rec["routine"]], [])
        if k == "machine":
            return vm.Machine()
        if k == "obj":
            cls = _CLASSES[rec["c"]]
            return object.__new__(cls)
        if k == "game":
            from .game import Game

            return Game.__new__(Game)
        if k == "function":
            return None
        raise SnapshotError(f"unknown snapshot record kind {k!r}")

    def value(self, v):
        if v is None or isinstance(v, (bool, int, str)):
            return v
        if isinstance(v, dict):
            if "$" in v:
                obj = self.objs[v["$"]]
                if obj is None and self.table[v["$"]]["k"] == "function":
                    obj = self._function(v["$"])
                return obj
            if "f" in v:
                return float(v["f"])
            if "t" in v:
                return tuple(self.value(x) for x in v["t"])
            if "fs" in v:
                return frozenset(self.value(x) for x in v["fs"])
            if "r" in v:
                return range(*v["r"])
            if "e" in v:
                mod, _, qual = v["e"].partition(":")
                return _resolve_attr(importlib.import_module(mod), qual)(self.value(v["v"]))
            if "cd" in v:
                return get_card_def(v["cd"])
            if "u" in v:
                return vm.UNBOUND
            if "fn" in v:
                mod, qual = v["fn"]
                obj = _resolve_attr(importlib.import_module(mod), qual)
                return getattr(obj, "__func__", obj)
            if "m" in v:
                fn, self_ = (self.value(x) for x in v["m"])
                return types.MethodType(fn, self_)
            if "count" in v:
                return itertools.count(v["count"])
        raise SnapshotError(f"unreadable snapshot value {v!r}")

    def _function(self, i: int):
        rec = self.table[i]
        code = _code_from_path(rec["code"])
        closure = tuple(self.value(c) for c in rec["closure"]) or None
        defaults = self.value(rec["defaults"]) if rec["defaults"] is not None else None
        fn = types.FunctionType(code, importlib.import_module(rec["globals"]).__dict__, code.co_name, defaults, closure)
        fn.__qualname__ = code.co_qualname if hasattr(code, "co_qualname") else code.co_name
        if rec["kwdefaults"] is not None:
            fn.__kwdefaults__ = self.value(rec["kwdefaults"])
        self.objs[i] = fn
        return fn

    def fill(self, i: int):
        rec, obj = self.table[i], self.objs[i]
        k = rec["k"]
        if k == "list":
            obj.extend(self.value(x) for x in rec["v"])
        elif k in ("dict", "defaultdict_list"):
            for kk, vv in rec["v"]:
                obj[self.value(kk)] = self.value(vv)
        elif k == "set":
            obj.update(self.value(x) for x in rec["v"])
        elif k == "deque":
            obj.extend(self.value(x) for x in rec["v"])
        elif k == "cell":
            if "v" in rec:
                obj.cell_contents = self.value(rec["v"])
        elif k == "random":
            state = self.value(rec["v"])
            obj.setstate(state)
        elif k == "frame":
            obj.pc = rec["pc"]
            obj.L = [self.value(x) for x in rec["L"]]
        elif k == "machine":
            obj.stack = [self.value(x) for x in rec["stack"]]
        elif k == "obj":
            for name, v in rec["a"]:
                object.__setattr__(obj, name, self.value(v))

        elif k == "game":
            for name, v in rec["a"]:
                setattr(obj, name, self.value(v))
            obj._fork_snapshot = None
            obj.choice_log = []
            obj._machine = self.value(rec["machine"]) if rec["machine"] is not None else None


def restore(data: bytes):
    doc = json.loads(data.decode("utf-8"))
    if doc.get("snapshot_version") != SNAPSHOT_VERSION:
        raise SnapshotError(f"snapshot format {doc.get('snapshot_version')!r}, this engine reads {SNAPSHOT_VERSION}")
    if doc.get("engine_version") != ENGINE_VERSION or doc.get("rules_hash") != RULES_HASH:
        raise SnapshotError("snapshot was made by a different engine (version or rules hash): its frames don't fit this one")
    vm.load_compiled()
    r = _Reader(doc["objects"])
    for i in range(len(r.table)):
        r.objs[i] = r.shell(i)
    for i, rec in enumerate(r.table):
        if rec["k"] == "function" and r.objs[i] is None:
            r._function(i)
    for i in range(len(r.table)):
        r.fill(i)
    for obj in r.objs:  # derived state, once everything it derives from is filled
        if type(obj) is ActiveEffectList:
            obj._duration_by_key = collections.defaultdict(list)
            for e in obj.duration_effects:
                obj._duration_by_key[(e.variable, e.player_affected)].append(e)
    game = r.value(doc["root"])
    from .game import Game

    if game.is_over:
        game._driver = iter(())
    elif game.execution == "compiled":
        game._driver = vm.MachineDriver(game._machine)
    else:
        pending = game.pending_decision
        game._driver = game._resume()
        game.pending_decision = next(game._driver)
        if Game is not None and pending is not None and pending.kind != game.pending_decision.kind:
            raise SnapshotError("a native game restored to a different decision")
    return game

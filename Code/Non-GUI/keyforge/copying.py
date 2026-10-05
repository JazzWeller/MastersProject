"""Copying what `Game.copy()`'s hand-written data copy can't know the shape
of (Agent Observation Plan, Part R, R1 and R6): the locals of suspended
frames, closures stored in lasting effects, and the pending decision.

`Remapper(old_game, new_game, memo)` maps every object reachable from those
to its counterpart in the copy, by identity, like `copy.deepcopy` with a
memo -- but seeded with the copy's own engine objects, so that:

- a card, player, zone, effect or the game itself maps to the copy's own
  object, never to a second copy of it;
- an engine-owned container a frame holds (a live battleline list, a
  player's `CardsPlayed` dict, the log's event list) maps to the copy's
  container, so a frame that reads it live still does;
- anything else (a frame's own lists and dicts, an event dict shared by the
  frames of a trigger chain, a cell shared by a frame and the closure it
  made) is copied once and shared exactly as it was;
- immutable values (ints, strings, enums, card definitions, plain
  functions) are shared.

An object of a type it doesn't know is an error (I5), never a shallow
reference into the original game.
"""

from __future__ import annotations

import dataclasses
import random
import types
from collections import deque
from enum import Enum
from typing import Any, Dict

from . import vm
from .actions import DiscardCard, EndTurn, Fight, PlayCard, Reap, UseAction, UseOmni
from .cards.card import Card, CardDef, TypeObject
from .decision import Decision
from .effects.effect_object import EffectObject
from .log import GameLog, LogEvent

_ATOMS = (type(None), bool, int, float, complex, str, bytes, range, type, types.ModuleType, types.BuiltinFunctionType,
          types.CodeType, CardDef, LogEvent, vm.Routine, type(vm.UNBOUND))
_ACTIONS = (DiscardCard, Fight, PlayCard, Reap, UseAction, UseOmni)
# Types whose values are shared, never copied: `_ATOMS`, plus each enum class
# as it is first met.
_ATOM_TYPES = set(_ATOMS)


class UncopyableState(TypeError):
    pass


def container_index(game) -> Dict[int, tuple]:
    """id() of each of `game`'s own containers -> where it lives (a key of
    `_RESOLVE` and its arguments). Cached on the game until it changes: a
    search copies the same root once per simulation, and builds this once."""
    stamp = (len(game.choice_record), len(game.log.events), id(game.pending_decision))
    cached = game.__dict__.get("_container_index")
    if cached is not None and cached[0] == stamp:
        return cached[1]
    idx: Dict[int, tuple] = {}
    for pid, p in game.players.items():
        idx[id(p.all_cards)] = ("all_cards", pid)
        for attr in ("hand", "discard", "archive", "purged", "deck"):
            z = getattr(p, attr)
            idx[id(z)] = ("zone", pid, attr)
            idx[id(z._cards)] = ("zone_cards", pid, attr)
        idx[id(p.play_area)] = ("play_area", pid)
        idx[id(p.play_area.creatures)] = ("creatures", pid)
        idx[id(p.play_area.artifacts)] = ("artifacts", pid)
        for attr in ("CardsPlayed", "used_this_turn", "ExtraHousePlayable", "hand_revealed_to"):
            idx[id(getattr(p, attr))] = ("player_attr", pid, attr)
    for iid, c in game._cards_by_id.items():
        idx[id(c.under_cards)] = ("under", iid)
        idx[id(c.extra_triggers)] = ("extra", iid)
        for k, v in c.extra_triggers.items():
            idx[id(v)] = ("extra_list", iid, k)
        ups = getattr(c.type_object, "upgrades", None)
        if ups is not None:
            idx[id(ups)] = ("upgrades", iid)
    idx[id(game.log)] = ("log",)
    idx[id(game.log.events)] = ("log_events",)
    for attr in ("choice_log", "choice_record", "_rng_counters", "_temp_control", "_end_of_turn_cleanups",
                 "_redirected_hits", "_player_houses_cache", "_cards_by_id"):
        idx[id(getattr(game, attr))] = ("game_attr", attr)
    if game.result is not None:
        idx[id(game.result)] = ("game_attr", "result")
    game.__dict__["_container_index"] = (stamp, idx)
    return idx


_RESOLVE = {
    "all_cards": lambda g, pid: g.players[pid].all_cards,
    "zone": lambda g, pid, attr: getattr(g.players[pid], attr),
    "zone_cards": lambda g, pid, attr: getattr(g.players[pid], attr)._cards,
    "play_area": lambda g, pid: g.players[pid].play_area,
    "creatures": lambda g, pid: g.players[pid].play_area.creatures,
    "artifacts": lambda g, pid: g.players[pid].play_area.artifacts,
    "player_attr": lambda g, pid, attr: getattr(g.players[pid], attr),
    "under": lambda g, iid: g._cards_by_id[iid].under_cards,
    "extra": lambda g, iid: g._cards_by_id[iid].extra_triggers,
    "extra_list": lambda g, iid, k: g._cards_by_id[iid].extra_triggers[k],
    "upgrades": lambda g, iid: g._cards_by_id[iid].type_object.upgrades,
    "log": lambda g: g.log,
    "log_events": lambda g: g.log.events,
    "game_attr": lambda g, attr: getattr(g, attr),
}


class Remapper:
    def __init__(self, old_game, new_game, memo: Dict[int, Any]):
        self.old = old_game
        self.new = new_game
        self.memo = memo
        self._index = None

    def _engine_container(self, x):
        """The copy's counterpart of `x` if `x` is one of the old game's own
        containers (a zone's list, a battleline, a player's dict, the log's
        event list, ...), else None."""
        index = self._index
        if index is None:
            index = self._index = container_index(self.old)
        where = index.get(id(x))
        if where is None:
            return None
        return _RESOLVE[where[0]](self.new, *where[1:])

    def seed_effects(self) -> None:
        """Pairs the effect lists, once the copy has them."""
        oe, ne = self.old.active_effects, self.new.active_effects
        self.memo[id(oe)] = ne
        for attr in ("duration_effects", "trigger_effects", "instead_effects", "modifier_effects"):
            self.memo[id(getattr(oe, attr))] = getattr(ne, attr)

    # ------------------------------------------------------------ remap ----

    def remap(self, x):
        t = type(x)
        if t in _ATOM_TYPES:
            return x
        if isinstance(x, Enum):
            _ATOM_TYPES.add(t)  # an enum class: its members are shared
            return x
        memo = self.memo
        hit = memo.get(id(x))
        if hit is not None:
            return hit
        if t is list:
            hit = self._engine_container(x)
            if hit is not None:
                memo[id(x)] = hit
                return hit
            out = []
            memo[id(x)] = out
            out.extend(self.remap(v) for v in x)
            return out
        if t is tuple:
            items = tuple(self.remap(v) for v in x)
            return x if all(a is b for a, b in zip(items, x)) else items
        if t is dict:
            hit = self._engine_container(x)
            if hit is not None:
                memo[id(x)] = hit
                return hit
            out = {}
            memo[id(x)] = out
            for k, v in x.items():
                out[self.remap(k)] = self.remap(v)
            return out
        if t is frozenset:
            items = [self.remap(v) for v in x]
            return x if all(a is b for a, b in zip(items, x)) else frozenset(items)
        if t is set:
            hit = self._engine_container(x)
            if hit is not None:
                memo[id(x)] = hit
                return hit
            out = set(self.remap(v) for v in x)
            memo[id(x)] = out
            return out
        if t is deque:
            hit = self._engine_container(x)
            if hit is not None:
                memo[id(x)] = hit
                return hit
            out = deque()
            memo[id(x)] = out
            out.extend(self.remap(v) for v in x)
            return out
        if t is types.FunctionType:
            return self._function(x)
        if t is types.MethodType:
            out = types.MethodType(self.remap(x.__func__), self.remap(x.__self__))
            memo[id(x)] = out
            return out
        if t is types.CellType:
            out = types.CellType()
            memo[id(x)] = out
            try:
                out.cell_contents = self.remap(x.cell_contents)
            except ValueError:
                pass  # an empty cell stays empty
            return out
        if t is vm.Frame:
            return self.frame(x)
        if t is Decision:
            out = Decision(x.player, x.kind, x.prompt)
            memo[id(x)] = out
            out.options = self.remap(x.options)
            out.min_n, out.max_n = x.min_n, x.max_n
            out.source_card = self.remap(x.source_card)
            out.intent, out.affects, out.optional = x.intent, x.affects, x.optional
            return out
        if t in _ACTIONS:
            out = t(self.remap(x.card))
            memo[id(x)] = out
            return out
        if t is EndTurn:
            return x
        if isinstance(x, EffectObject):
            out = t.__new__(t)
            memo[id(x)] = out
            for k, v in vars(x).items():
                setattr(out, k, self.remap(v))
            return out
        if t is random.Random:
            out = random.Random()
            out.setstate(x.getstate())
            memo[id(x)] = out
            return out
        hit = self._engine_container(x)  # a zone, a play area, the log
        if hit is not None:
            memo[id(x)] = hit
            return hit
        if isinstance(x, (Card, TypeObject, GameLog)):
            raise UncopyableState(f"{t.__name__} {x!r} is not one of this game's own objects")
        if dataclasses.is_dataclass(x) and not isinstance(x, type):
            out = object.__new__(t)
            memo[id(x)] = out
            for f in dataclasses.fields(x):
                object.__setattr__(out, f.name, self.remap(getattr(x, f.name)))
            return out
        raise UncopyableState(f"can't copy a {t.__module__}.{t.__qualname__} held in game state: {x!r}")

    def _function(self, fn):
        if fn.__closure__ is None and not fn.__defaults__ and not fn.__kwdefaults__:
            return fn
        closure = tuple(self.remap(c) for c in fn.__closure__) if fn.__closure__ else None
        defaults = self.remap(fn.__defaults__) if fn.__defaults__ else fn.__defaults__
        if closure is not None and all(a is b for a, b in zip(closure, fn.__closure__)) and defaults is fn.__defaults__ and not fn.__kwdefaults__:
            return fn
        out = types.FunctionType(fn.__code__, fn.__globals__, fn.__name__, defaults, closure)
        out.__qualname__ = fn.__qualname__
        if fn.__kwdefaults__:
            out.__kwdefaults__ = self.remap(fn.__kwdefaults__)
        if fn.__dict__:
            attrs = {k: v for k, v in fn.__dict__.items() if k != vm.STEP_ATTR}
            if attrs:
                out.__dict__.update(self.remap(attrs))
        self.memo[id(fn)] = out
        return out

    def frame(self, fr):
        if type(fr) is not vm.Frame:
            raise UncopyableState(f"can't copy a {type(fr).__name__}: code that isn't compiled is suspended")
        xs = fr.routine.exc_slot
        if xs is not None and fr.L[xs] is not None:
            raise UncopyableState("can't copy a frame with an exception in flight")
        out = vm.Frame(fr.routine, [])
        self.memo[id(fr)] = out
        remap = self.remap
        memo = self.memo
        atoms = _ATOM_TYPES
        L = []  # the slot list itself is the frame's own
        for v in fr.L:
            if type(v) in atoms:
                L.append(v)
            else:
                hit = memo.get(id(v))
                L.append(hit if hit is not None else remap(v))
        out.L = L
        out.pc = fr.pc
        return out

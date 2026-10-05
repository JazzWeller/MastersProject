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


class UncopyableState(TypeError):
    pass


class Remapper:
    def __init__(self, old_game, new_game, memo: Dict[int, Any]):
        self.old = old_game
        self.new = new_game
        self.memo = memo
        self._containers_done = False

    # Engine-owned containers, paired up only once something needs one.
    def _seed_containers(self) -> None:
        if self._containers_done:
            return
        self._containers_done = True
        memo = self.memo
        old, new = self.old, self.new

        def pair(a, b):
            if id(a) not in memo:
                memo[id(a)] = b

        for pid, op in old.players.items():
            np = new.players[pid]
            pair(op.all_cards, np.all_cards)
            for attr in ("hand", "discard", "archive", "purged"):
                oz, nz = getattr(op, attr), getattr(np, attr)
                pair(oz, nz)
                pair(oz._cards, nz._cards)
            pair(op.deck, np.deck)
            pair(op.deck._cards, np.deck._cards)
            pair(op.play_area, np.play_area)
            pair(op.play_area.creatures, np.play_area.creatures)
            pair(op.play_area.artifacts, np.play_area.artifacts)
            for attr in ("CardsPlayed", "used_this_turn", "ExtraHousePlayable", "hand_revealed_to"):
                pair(getattr(op, attr), getattr(np, attr))
        for iid, oc in old._cards_by_id.items():
            nc = new._cards_by_id[iid]
            pair(oc.under_cards, nc.under_cards)
            pair(oc.extra_triggers, nc.extra_triggers)
            for k, v in oc.extra_triggers.items():
                pair(v, nc.extra_triggers[k])
            ups = getattr(oc.type_object, "upgrades", None)
            if ups is not None:
                pair(ups, nc.type_object.upgrades)
        pair(old.log, new.log)
        pair(old.log.events, new.log.events)
        for attr in ("choice_log", "choice_record", "_rng_counters", "_temp_control", "_end_of_turn_cleanups",
                     "_redirected_hits", "_player_houses_cache", "_cards_by_id"):
            pair(getattr(old, attr), getattr(new, attr))
        if old.result is not None and new.result is not None:
            pair(old.result, new.result)

    def seed_effects(self) -> None:
        """Pairs the effect lists, once the copy has them."""
        oe, ne = self.old.active_effects, self.new.active_effects
        self.memo[id(oe)] = ne
        for attr in ("duration_effects", "trigger_effects", "instead_effects", "modifier_effects"):
            self.memo[id(getattr(oe, attr))] = getattr(ne, attr)

    # ------------------------------------------------------------ remap ----

    def remap(self, x):
        t = type(x)
        if t in _ATOMS or isinstance(x, Enum):
            return x
        memo = self.memo
        hit = memo.get(id(x))
        if hit is not None:
            return hit
        if t is list:
            self._seed_containers()
            hit = memo.get(id(x))
            if hit is not None:
                return hit
            out = []
            memo[id(x)] = out
            out.extend(self.remap(v) for v in x)
            return out
        if t is tuple:
            items = tuple(self.remap(v) for v in x)
            return x if all(a is b for a, b in zip(items, x)) else items
        if t is dict:
            self._seed_containers()
            hit = memo.get(id(x))
            if hit is not None:
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
            self._seed_containers()
            hit = memo.get(id(x))
            if hit is not None:
                return hit
            out = set(self.remap(v) for v in x)
            memo[id(x)] = out
            return out
        if t is deque:
            self._seed_containers()
            hit = memo.get(id(x))
            if hit is not None:
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
        out.L = [remap(v) for v in fr.L]  # the slot list itself is the frame's own
        out.pc = fr.pc
        return out

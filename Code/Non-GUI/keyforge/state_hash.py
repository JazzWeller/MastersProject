"""Canonical, JSON-safe representations of engine state, for `Game.state_hash()`.

Split out of `game.py` because canonicalizing everything that can appear in
play -- `Card`s, the seven `CHOOSE_ACTION` option dataclasses, `TriggerEffect`
objects and ad hoc tuples used as `ORDER_EFFECTS` options, closures held by
`DurationEffect`/upgrade-granted abilities -- is a decent chunk of logic on
its own, and keeping it together makes the "what does the hash actually
cover" question answerable by reading one file.

The hash is exact (Agent Observation Plan, Part R, R1): a callable held in
state -- a lasting effect's value, condition or handler, an upgrade-granted
ability -- is hashed as what it is, its code's qualified name plus the
values it closes over and its defaults; pending end-of-turn cleanups are
hashed by content. (Before Part R both were approximations: a fixed
`"<callable>"` marker, and the cleanups' count.)
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any

from .actions import DiscardCard, EndTurn, Fight, PlayCard, Reap, UseAction, UseOmni
from .cards.card import Card, CreatureType, UpgradeType
from .effects.effect_object import EffectObject, TriggerEffect
from .version import ENGINE_VERSION

import types as _types

_ACTION_CARD_TYPES = (PlayCard, DiscardCard, UseAction, UseOmni, Reap, Fight)

_EXCLUDED_CARD_ATTRS = frozenset({"card_def", "type_object"})
_EXCLUDED_PLAYER_ATTRS = frozenset({"deck", "discard", "hand", "archive", "purged", "play_area", "all_cards"})


def canonicalize(value: Any) -> Any:
    """A JSON-safe, order-preserving representation of `value`. Cards become
    `{"__card__": instance_id}` (stable per-game per Milestone A, so two
    independent replays of the same choices agree); enums become their
    `.value`; sets become a `repr`-sorted list (order-independent, and
    immune to hash randomization since it never sorts by Python's own
    hash); anything else callable collapses to a fixed marker."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Card):
        return {"__card__": value.instance_id}
    if isinstance(value, EffectObject):
        # A handful of cards stash a back-reference to their own registered
        # effect object as an ad hoc attribute (Ember Imp); it's the same
        # object already canonicalized in full via `active_effects`, so a
        # reference here only needs to identify it, not repeat its content.
        return {
            "__effect__": type(value).__name__,
            "source": value.source_card.instance_id if value.source_card is not None else None,
            "controller": value.controller,
        }
    if isinstance(value, (list, tuple)):
        return [canonicalize(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((canonicalize(v) for v in value), key=repr)
    if isinstance(value, dict):
        return {str(k): canonicalize(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, _types.FunctionType):
        return canonical_function(value)
    if isinstance(value, _types.MethodType):
        return {"__method__": canonical_function(value.__func__), "self": canonicalize(value.__self__)}
    t = type(value).__name__
    if t == "Player":
        return {"__player__": value.id}
    if t == "Game":
        return {"__game__": None}
    if callable(value):
        return {"__callable__": f"{type(value).__module__}.{type(value).__qualname__}"}
    return str(value)


def canonical_function(fn) -> dict:
    """A function as data: its code's module and qualified name, the values
    its closure cells hold (by free-variable name) and its defaults. Two
    closures made by the same code over the same values are equal; any
    difference in what they capture shows."""
    out = {"__fn__": f"{fn.__module__}:{fn.__qualname__}"}
    if fn.__closure__:
        cells = {}
        for name, cell in zip(fn.__code__.co_freevars, fn.__closure__):
            try:
                cells[name] = canonicalize(cell.cell_contents)
            except ValueError:
                cells[name] = {"__empty_cell__": None}
        out["cells"] = cells
    if fn.__defaults__:
        out["defaults"] = canonicalize(fn.__defaults__)
    if fn.__kwdefaults__:
        out["kwdefaults"] = canonicalize(fn.__kwdefaults__)
    return out


def _canonicalize_option(option: Any) -> Any:
    if isinstance(option, _ACTION_CARD_TYPES):
        return {"type": type(option).__name__, "card": option.card.instance_id}
    if isinstance(option, EndTurn):
        return {"type": "EndTurn"}
    if isinstance(option, TriggerEffect):
        return {
            "type": "TriggerEffect",
            "source": option.source_card.instance_id if option.source_card is not None else None,
            "controller": option.controller,
            "event": option.event,
        }
    return canonicalize(option)


def canonicalize_decision(decision) -> Any:
    if decision is None:
        return None
    return {
        "player": decision.player,
        "kind": decision.kind.name,
        "min_n": decision.min_n,
        "max_n": decision.max_n,
        "options": [_canonicalize_option(o) for o in decision.options],
    }


def canonical_type_object_state(type_object) -> dict:
    if isinstance(type_object, CreatureType):
        return {
            "kind": "creature",
            "base_power": type_object.base_power,
            "base_armor": type_object.base_armor,
            "damage": type_object.damage,
            "armor_used_this_turn": type_object.armor_used_this_turn,
            "upgrades": [u.instance_id for u in type_object.upgrades],
        }
    if isinstance(type_object, UpgradeType):
        return {"kind": "upgrade", "host": type_object.host.instance_id if type_object.host is not None else None}
    return {"kind": "other"}  # ActionType / ArtifactType carry no extra mutable state


def canonical_card_state(card: Card) -> dict:
    # `card.__slots__` instead of `vars(card)` -- Card has no `__dict__`
    # (Milestone L: __slots__, for less memory and cheaper copies), but its
    # own `__slots__` tuple is exactly the same "every attribute name" list
    # vars() used to give for free.
    data = {}
    for k in Card.__slots__:
        if k in _EXCLUDED_CARD_ATTRS:
            continue
        value = getattr(card, k)
        if k == "_ember_imp_effect" and value is None:
            # Only Ember Imp ever sets this -- skipping it here when unset
            # matches vars()'s old behavior exactly (absent for every other
            # card, rather than a newly-uniform `None` on all of them, which
            # would silently change every card's canonical state and, with
            # it, every recorded state_hash).
            continue
        data[k] = canonicalize(value)
    data["type_object"] = canonical_type_object_state(card.type_object)
    return data


def canonical_player_state(player) -> dict:
    data = {k: canonicalize(v) for k, v in vars(player).items() if k not in _EXCLUDED_PLAYER_ATTRS}
    data["deck"] = [canonical_card_state(c) for c in player.deck.cards()]
    data["hand"] = [canonical_card_state(c) for c in player.hand.cards()]
    data["discard"] = [canonical_card_state(c) for c in player.discard.cards()]
    data["archive"] = [canonical_card_state(c) for c in player.archive.cards()]
    data["purged"] = [canonical_card_state(c) for c in player.purged.cards()]
    data["creatures"] = [canonical_card_state(c) for c in player.play_area.creatures]
    data["artifacts"] = [canonical_card_state(c) for c in player.play_area.artifacts]
    return data


def canonical_effects_state(active_effects) -> dict:
    def source_iid(e):
        return e.source_card.instance_id if e.source_card is not None else None

    return {
        "duration": [
            {
                "source": source_iid(e),
                "controller": e.controller,
                "remaining": e.remaining_duration,
                "player_affected": e.player_affected,
                "variable": e.variable,
                "op": e.op,
                "value": canonicalize(e.value),
                "conditional": canonicalize(e.conditional),
            }
            for e in active_effects.duration_effects
        ],
        "trigger": [
            {"source": source_iid(e), "controller": e.controller, "event": e.event, "remaining": e.remaining_duration,
             "handler": canonicalize(e.handler)}
            for e in active_effects.trigger_effects
        ],
        "instead": [
            {"source": source_iid(e), "controller": e.controller, "kind": e.kind, "handler": canonicalize(e.handler)}
            for e in active_effects.instead_effects
        ],
        "modifier": [
            {"source": source_iid(e), "controller": e.controller, "kind": e.kind, "handler": canonicalize(e.handler)}
            for e in active_effects.modifier_effects
        ],
    }


def canonical_log_state(log) -> list:
    return [{"kind": e.kind, "data": canonicalize(e.data)} for e in log.events]


def canonical_temp_control(temp_control: dict) -> list:
    return [
        [source_iid, [[c.instance_id, orig_pid] for c, orig_pid in entries]]
        for source_iid, entries in sorted(temp_control.items())
    ]


def canonical_rng_counters(counters: dict) -> list:
    return sorted(([player, kind, count] for (player, kind), count in counters.items()), key=lambda row: (str(row[0]), row[1]))


def canonical_game_state(game) -> dict:
    return {
        "engine_version": ENGINE_VERSION,
        "turn_number": game.turn_number,
        "active_player_id": game.active_player_id,
        "first_player": game._first_player,
        "is_over": game.is_over,
        "result": canonicalize(game.result),
        "elusive_suppressed": game._elusive_suppressed,
        "players": {str(pid): canonical_player_state(p) for pid, p in sorted(game.players.items())},
        "active_effects": canonical_effects_state(game.active_effects),
        "pending_cleanups": canonicalize(game._end_of_turn_cleanups),
        "temp_control": canonical_temp_control(game._temp_control),
        "log": canonical_log_state(game.log),
        "rng_counters": canonical_rng_counters(game._rng_counters),
        "pending_decision": canonicalize_decision(game.pending_decision),
    }


def compute_state_hash(game) -> str:
    payload = json.dumps(canonical_game_state(game), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

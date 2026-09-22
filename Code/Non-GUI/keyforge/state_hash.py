"""Canonical, JSON-safe representations of engine state, for `Game.state_hash()`.

Split out of `game.py` because canonicalizing everything that can appear in
play -- `Card`s, the seven `CHOOSE_ACTION` option dataclasses, `TriggerEffect`
objects and ad hoc tuples used as `ORDER_EFFECTS` options, closures held by
`DurationEffect`/upgrade-granted abilities -- is a decent chunk of logic on
its own, and keeping it together makes the "what does the hash actually
cover" question answerable by reading one file.

Two known, accepted approximations, both inherent to the engine holding
closures as live state (see Code/AGENT_INTERFACE_PLAN.md's "Verified engine
facts"): a `DurationEffect.value`/`ModifierEffect.handler`/etc. that is
callable collapses to a fixed `"<callable>"` marker rather than comparing
code identity, and pending `_end_of_turn_cleanups` are hashed by count only,
not content. Two independently-replayed games that reached the same state by
the same choices register the same closures by construction, so this does
not weaken fork-equivalence checking in practice -- it only means the hash
can't distinguish games that (incorrectly) differ *solely* in an effect's
code identity, which isn't a thing that can happen at all outside a plugin
architecture this engine doesn't have.
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
    if callable(value):
        return "<callable>"
    return str(value)


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
    data = {k: canonicalize(v) for k, v in vars(card).items() if k not in _EXCLUDED_CARD_ATTRS}
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
                "conditional": e.conditional is not None,
            }
            for e in active_effects.duration_effects
        ],
        "trigger": [
            {"source": source_iid(e), "controller": e.controller, "event": e.event, "remaining": e.remaining_duration}
            for e in active_effects.trigger_effects
        ],
        "instead": [
            {"source": source_iid(e), "controller": e.controller, "kind": e.kind} for e in active_effects.instead_effects
        ],
        "modifier": [
            {"source": source_iid(e), "controller": e.controller, "kind": e.kind} for e in active_effects.modifier_effects
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
        "pending_cleanups": len(game._end_of_turn_cleanups),
        "temp_control": canonical_temp_control(game._temp_control),
        "log": canonical_log_state(game.log),
        "rng_counters": canonical_rng_counters(game._rng_counters),
        "pending_decision": canonicalize_decision(game.pending_decision),
    }


def compute_state_hash(game) -> str:
    payload = json.dumps(canonical_game_state(game), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

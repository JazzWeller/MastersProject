"""Stable option keys and feature vectors, for every option type the engine
produces (Agent Interface Plan, Milestone F).

`option_key(decision, option)` gives a hashable, canonical identity for one
option -- for fixed-vocabulary policy heads and information-set /
transposition-table keys. `option_features(decision, option)` gives a
JSON-safe feature dict for the same option -- for pointer/embedding heads.
Both exist because which one an agent uses is that agent's own design
choice, not something this module should decide for it.
"""

from __future__ import annotations

from typing import Any

from .actions import DiscardCard, EndTurn, Fight, PlayCard, Reap, UseAction, UseOmni
from .cards.card import Card
from .effects.effect_object import TriggerEffect
from .enums import House
from .state_hash import canonical_card_state, canonicalize

_ACTION_KINDS = {
    PlayCard: "play",
    DiscardCard: "discard",
    UseAction: "use_action",
    UseOmni: "use_omni",
    Reap: "reap",
    Fight: "fight",
}


def option_key(decision, option: Any) -> tuple:
    """A stable, hashable key for `option`. Two different options within
    the same decision never collide (card instance ids are unique per
    game, per Milestone A), and the same logical option keys identically
    across independent replays of the same choices."""
    kind = _ACTION_KINDS.get(type(option))
    if kind is not None:
        return (kind, option.card.instance_id)
    if isinstance(option, EndTurn):
        return ("end_turn",)
    if isinstance(option, Card):
        return ("card", option.instance_id)
    if isinstance(option, TriggerEffect):
        return ("trigger", option.source_card.instance_id if option.source_card is not None else None, option.event)
    if isinstance(option, House):
        return ("house", option.value)
    if isinstance(option, bool):
        return ("bool", option)
    if isinstance(option, (int, float, str)):
        return ("value", option)
    if isinstance(option, (list, tuple)):
        return ("seq", tuple(option_key(decision, o) for o in option))
    if callable(option):
        return ("callable",)
    return ("opaque", str(option))


def option_features(decision, option: Any) -> dict:
    """A JSON-safe feature dict for `option`. A `Card` option gets its full
    public state (state_hash.canonical_card_state -- safe here because the
    engine only ever offers a player options it's already entitled to see);
    every other option type gets a small, typed dict."""
    kind = _ACTION_KINDS.get(type(option))
    if kind is not None:
        return {"kind": kind, "card": option.card.instance_id}
    if isinstance(option, EndTurn):
        return {"kind": "end_turn"}
    if isinstance(option, Card):
        return {"kind": "card", **canonical_card_state(option)}
    if isinstance(option, TriggerEffect):
        return {
            "kind": "trigger",
            "source": option.source_card.instance_id if option.source_card is not None else None,
            "controller": option.controller,
            "event": option.event,
        }
    if isinstance(option, House):
        return {"kind": "house", "value": option.value}
    return {"kind": "value", "value": canonicalize(option)}

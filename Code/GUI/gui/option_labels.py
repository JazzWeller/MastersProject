"""Human-readable text for a Decision's options, and for game-log events.

This is the one place that knows how to turn any of the engine's option
types (Card, an actions.py dataclass, a House, a bool, a bare "left"/
"right"/"reap"/"fight"/"action"/"effect"/"check" string, an int pile id, a
TriggerEffect, or a destroy-pipeline tuple) into a label a player can read.
"""

from __future__ import annotations

from typing import Any, Optional

from keyforge.actions import DiscardCard, EndTurn, Fight, PlayCard, Reap, UseAction, UseOmni
from keyforge.cards.card import Card
from keyforge.effects.effect_object import TriggerEffect
from keyforge.enums import House


def _house_label(house: House) -> str:
    return house.value


def describe_option(opt: Any, decision=None, view=None) -> str:
    if isinstance(opt, EndTurn):
        return "End Turn"
    if isinstance(opt, PlayCard):
        return f"Play {opt.card.name}"
    if isinstance(opt, DiscardCard):
        return f"Discard {opt.card.name}"
    if isinstance(opt, UseAction):
        return f"Action: {opt.card.name}"
    if isinstance(opt, UseOmni):
        return f"Omni: {opt.card.name}"
    if isinstance(opt, Reap):
        return f"Reap with {opt.card.name}"
    if isinstance(opt, Fight):
        return f"Fight with {opt.card.name}"
    if isinstance(opt, Card):
        return opt.name
    if isinstance(opt, House):
        return _house_label(opt)
    if isinstance(opt, bool):
        return "Yes" if opt else "No"
    if isinstance(opt, TriggerEffect):
        src = opt.source_card.name if opt.source_card is not None else "Effect"
        return f"{src} (triggered effect)"
    if isinstance(opt, tuple) and len(opt) == 3 and opt[0] == "extra":
        _, card, _fn = opt
        return f"{card.name} (upgrade effect)"
    if opt == "reap":
        return "Reap"
    if opt == "fight":
        return "Fight"
    if opt == "action":
        return "Use Action"
    if opt == "effect":
        return "Play effect"
    if opt == "check":
        return "Play triggers"
    if opt == "left":
        return "Left flank"
    if opt == "right":
        return "Right flank"
    if isinstance(opt, int) and view is not None:
        me = view.viewer
        if opt == me:
            return f"Your discard pile ({len(view.players[opt].discard)})"
        return f"Opponent's discard pile ({len(view.players[opt].discard)})"
    if isinstance(opt, int):
        return f"Player {opt}"
    return str(opt)


# ------------------------------------------------------------ log events ----

_HIDDEN_ZONE_EVENTS = {"archive"}  # events that name a card moving into a hidden zone


def describe_log_event(event, viewer: int) -> Optional[str]:
    """A short sentence for the log panel. Redacts a card name when the
    event moves a card into a zone that's hidden from `viewer`."""
    k = event.kind
    d = event.data

    def who(pid):
        return "You" if pid == viewer else "Your opponent"

    def whose(pid):
        return "your" if pid == viewer else "their"

    def card_name(default="a card"):
        if k in _HIDDEN_ZONE_EVENTS and d.get("owner") is not None and d.get("owner") != viewer:
            return default
        if k in _HIDDEN_ZONE_EVENTS and d.get("player") is not None and d.get("player") != viewer:
            return default
        return d.get("card", default)

    if k == "mulligan":
        return f"{who(d['player'])} mulligan{'s' if d['player'] != viewer else ''} their hand."
    if k == "forge_key":
        return f"{who(d['player'])} forge{'s' if d['player'] != viewer else ''} a key! ({d['keys']}/3)"
    if k == "choose_house":
        return f"{who(d['player'])} choose{'s' if d['player'] != viewer else ''} house {d['house']}."
    if k == "take_archive":
        return f"{who(d['player'])} take{'s' if d['player'] != viewer else ''} {whose(d['player'])} archive into hand."
    if k == "discard_from_hand":
        return f"{who(d['player'])} discard{'s' if d['player'] != viewer else ''} {d['card']}."
    if k == "play_card":
        return f"{who(d['player'])} play{'s' if d['player'] != viewer else ''} {d['card']}."
    if k == "use_action":
        return f"{who(d['player'])} use{'s' if d['player'] != viewer else ''} {d['card']}'s Action."
    if k == "use_omni":
        return f"{who(d['player'])} use{'s' if d['player'] != viewer else ''} {d['card']}'s Omni."
    if k == "reap":
        return f"{who(d['player'])} reap{'s' if d['player'] != viewer else ''} with {d['card']}."
    if k == "fight":
        return f"{d['attacker']} fights {d['target']}."
    if k == "gain":
        return f"{who(d['player'])} gain{'s' if d['player'] != viewer else ''} {d['amount']} Æmber."
    if k == "steal":
        return f"{who(d['frm'])} loses {d['amount']} Æmber to {who(d['to']).lower()}."
    if k == "capture":
        return f"{d['card']} captures {d['amount']} Æmber."
    if k == "draw":
        return f"{who(d['player'])} draw{'s' if d['player'] != viewer else ''} {d['n']} card(s)."
    if k == "reshuffle":
        return f"{who(d['player'])} shuffle{'s' if d['player'] != viewer else ''} their discard pile into their deck."
    if k == "archive":
        return f"{who(d['player'])} archive{'s' if d['player'] != viewer else ''} {card_name()}."
    if k == "discard":
        return f"{d.get('card', 'A card')} is discarded."
    if k == "discard_random":
        return f"{who(d['player'])} discard{'s' if d['player'] != viewer else ''} a card at random."
    if k == "purge":
        return f"{d['card']} is purged."
    if k == "damage":
        return f"{d['card']} takes {d['amount']} damage."
    if k == "heal":
        return f"{d['card']} heals {d['amount']} damage."
    if k == "return_to_hand":
        return f"{d['card']} returns to hand."
    if k == "shuffle_into_deck":
        return f"{d['card']} is shuffled into the deck."
    if k == "ready":
        return f"{d['card']} is readied."
    if k == "exhaust":
        return f"{d['card']} is exhausted."
    if k == "destroyed":
        return f"{d['card']} is destroyed."
    if k == "duration_effect":
        return f"{d['card']}'s effect takes hold."
    if k == "arise":
        return f"Arise returns {d['n']} creature(s) to hand."
    if k == "help_from_future_self":
        return "Help From Future Self finds a Timetraveler." if d.get("found") else "Help From Future Self finds nothing."
    if k == "timetraveler_shuffle":
        return "Timetraveler shuffles itself into the deck."
    return None

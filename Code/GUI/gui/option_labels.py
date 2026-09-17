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


# What a lasting "when you play a card" trigger does, for effect-ordering labels.
_TRIGGER_TEXT = {"Library Access": "draw a card"}


def describe_option_short(opt: Any) -> str:
    """A short verb/label for `opt`, for use next to a card that's already
    shown on screen (the action chooser): "Play", "Discard", "Reap", not
    the full "Play Snudge" -- avoids repeating the card's own name back at
    the player right next to its art."""
    if isinstance(opt, EndTurn):
        return "End Turn"
    if isinstance(opt, PlayCard):
        return "Play"
    if isinstance(opt, DiscardCard):
        return "Discard"
    if isinstance(opt, UseAction):
        return "Use Action"
    if isinstance(opt, UseOmni):
        return "Use Omni"
    if isinstance(opt, Reap):
        return "Reap"
    if isinstance(opt, Fight):
        return "Fight"
    if isinstance(opt, TriggerEffect):
        return "Trigger Effect"
    if isinstance(opt, tuple) and len(opt) == 3 and opt[0] == "extra":
        return "Upgrade Effect"
    return describe_option(opt)


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
    ordering = decision is not None and getattr(decision.kind, "name", "") == "ORDER_EFFECTS"
    if isinstance(opt, Card):
        return f"{opt.name}: its Destroyed effect" if ordering else opt.name
    if isinstance(opt, House):
        return _house_label(opt)
    if isinstance(opt, bool):
        return "Yes" if opt else "No"
    if isinstance(opt, TriggerEffect):
        src = opt.source_card.name if opt.source_card is not None else "Effect"
        what = _TRIGGER_TEXT.get(src, "its triggered effect")
        return f"{src}: {what}"
    if isinstance(opt, tuple) and len(opt) == 3 and opt[0] == "extra":
        _, card, _fn = opt
        return f"{card.name}: effect granted by its upgrade"
    if opt == "reap":
        return "Reap"
    if opt == "fight":
        return "Fight"
    if opt == "action":
        return "Use Action"
    if opt == "effect":
        return "The card's own Play: effect"
    if opt == "check":
        return "Triggers from other cards (e.g. Library Access draws)"
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

_LOG_CATEGORY = {
    "gain": "aember", "steal": "aember", "capture": "aember",
    "forge_key": "key",
    "purge": "purge",
    "damage": "damage", "destroyed": "damage",
    "heal": "heal",
}


def log_event_category(event) -> str:
    """A coarse color bucket for the log panel (aember gold / key gold /
    damage red / purge violet / heal green / neutral) -- purely cosmetic,
    doesn't affect what's shown. See UX_FIX_PLAN.md D6."""
    return _LOG_CATEGORY.get(event.kind, "neutral")


def describe_log_event(event, viewer: int, spectating: bool = False) -> Optional[str]:
    """A short sentence for the log panel. Redacts a card name when the
    event moves a card into a zone that's hidden from `viewer`. When
    `spectating` (bot vs. bot, no human seat), nobody is "you" -- every
    player is addressed in the third person as "Player N" instead."""
    k = event.kind
    d = event.data

    def third_person(pid) -> bool:
        return spectating or pid != viewer

    def who(pid):
        if spectating:
            return f"Player {pid}"
        return "You" if pid == viewer else "Your opponent"

    def whose(pid):
        if spectating:
            return f"Player {pid}'s"
        return "your" if pid == viewer else "their"

    def s(pid):
        return "s" if third_person(pid) else ""

    def card_name(default="a card"):
        if k in _HIDDEN_ZONE_EVENTS and d.get("owner") is not None and d.get("owner") != viewer:
            return default
        if k in _HIDDEN_ZONE_EVENTS and d.get("player") is not None and d.get("player") != viewer:
            return default
        return d.get("card", default)

    if k == "mulligan":
        return f"{who(d['player'])} mulligan{s(d['player'])} {whose(d['player'])} hand."
    if k == "forge_key":
        cost = f" for {d['cost']} Æmber" if d.get("cost") is not None else ""
        return f"{who(d['player'])} forge{s(d['player'])} a key{cost}! ({d['keys']}/3)"
    if k == "choose_house":
        return f"{who(d['player'])} choose{s(d['player'])} house {d['house']}."
    if k == "house_forced":
        src = f" ({d['source']})" if d.get("source") else ""
        return f"{who(d['player'])} must use house {d['house']} this turn{src}."
    if k == "forge_skipped":
        src = f" because of {d['source']}" if d.get("source") else ""
        return f"{who(d['player'])} can't forge a key{src}, despite having {d['aember']} Æmber."
    if k == "put_on_top":
        return f"A card is put on top of {whose(d['player'])} deck."
    if k == "put_on_bottom":
        return f"A card is put on the bottom of {whose(d['player'])} deck."
    if k == "take_archive":
        return f"{who(d['player'])} take{s(d['player'])} {whose(d['player'])} archive into hand."
    if k == "discard_from_hand":
        return f"{who(d['player'])} discard{s(d['player'])} {d['card']}."
    if k == "play_card":
        return f"{who(d['player'])} play{s(d['player'])} {d['card']}."
    if k == "use_action":
        return f"{who(d['player'])} use{s(d['player'])} {d['card']}'s Action."
    if k == "use_omni":
        return f"{who(d['player'])} use{s(d['player'])} {d['card']}'s Omni."
    if k == "reap":
        return f"{who(d['player'])} reap{s(d['player'])} with {d['card']}."
    if k == "fight":
        return f"{d['attacker']} fights {d['target']}."
    if k == "gain":
        return f"{who(d['player'])} gain{s(d['player'])} {d['amount']} Æmber."
    if k == "steal":
        to_txt = who(d["to"]) if spectating else who(d["to"]).lower()
        return f"{who(d['frm'])} lose{s(d['frm'])} {d['amount']} Æmber to {to_txt}."
    if k == "capture":
        return f"{d['card']} captures {d['amount']} Æmber."
    if k == "draw":
        return f"{who(d['player'])} draw{s(d['player'])} {d['n']} card(s)."
    if k == "reshuffle":
        return f"{who(d['player'])} shuffle{s(d['player'])} {whose(d['player'])} discard pile into {whose(d['player'])} deck."
    if k == "archive":
        return f"{who(d['player'])} archive{s(d['player'])} {card_name()}."
    if k == "discard":
        return f"{d.get('card', 'A card')} is discarded."
    if k == "discard_random":
        return f"{who(d['player'])} discard{s(d['player'])} a card at random."
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
        return f"{d['card']}: {_duration_text(d)}."
    if k == "arise":
        return f"Arise returns {d['n']} creature(s) to hand."
    if k == "help_from_future_self":
        return "Help From Future Self finds a Timetraveler." if d.get("found") else "Help From Future Self finds nothing."
    if k == "timetraveler_shuffle":
        return "Timetraveler shuffles itself into the deck."
    return None


_DURATION_TEXT = {
    ("CanPlayActions", False): "the opponent can't play actions next turn",
    ("CanPlayCreatures", False): "the opponent can't play creatures next turn",
    ("CanKeyForge", False): "the opponent can't forge a key next turn",
}


def _duration_text(d) -> str:
    var, op, value = d.get("variable"), d.get("op"), d.get("value")
    if (var, value) in _DURATION_TEXT:
        return _DURATION_TEXT[(var, value)]
    if var == "KeyForgeCost":
        return f"keys cost {op}{value} Æmber for the opponent next turn"
    return f"{var} {op} {value}"


# Events that name a card in a zone that may be hidden from some viewers.
_REDACTABLE = {"archive"}


def log_event_iid(event, viewer: int):
    """The instance id of the card a log line is about, if it's safe to let
    the viewer hover it (i.e. the line isn't redacted for them), else None."""
    d = event.data
    if event.kind in _REDACTABLE and d.get("player") not in (None, viewer):
        return None
    for key in ("iid", "attacker_iid"):
        if d.get(key) is not None:
            return d[key]
    return None

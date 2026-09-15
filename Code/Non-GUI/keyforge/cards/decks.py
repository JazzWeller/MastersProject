"""Fignor and Igor decklists (3 pods x 12 cards each, per the spec)."""

from __future__ import annotations

from typing import List

from .card import Card
from .card_data import get_card_def

FIGNOR_LIST = (
    # Dis
    "Arise",
    "Control the Weak",
    "Creeping Oblivion",
    "Gateway to Dis",
    "Three Fates",
    "Library of the Damned",
    "Dust Imp",
    "Shooler",
    "Snudge",
    "Snudge",
    "Succubus",
    "The Terror",
    # Logos
    "Help From Future Self",
    "Labwork",
    "Phase Shift",
    "Scrambler Storm",
    "Sloppy Labwork",
    "Wild Wormhole",
    "The Howling Pit",
    "Doc Bookton",
    "Mother",
    "Mother",
    "Timetraveler",
    "Titan Mechanic",
    # Shadows
    "Bait and Switch",
    "Booby Trap",
    "Ghostly Hand",
    "Lights Out",
    "Miasma",
    "Miasma",
    "Nerve Blast",
    "Too Much To Protect",
    "Subtle Maul",
    "Noddy the Thief",
    "Silvertooth",
    "Urchin",
)

IGOR_LIST = (
    # Dis
    "Arise",
    "Control the Weak",
    "Control the Weak",
    "Gateway to Dis",
    "Dominator Bauble",
    "Lash of Broken Dreams",
    "Lifeward",
    "Dust Imp",
    "Ember Imp",
    "Guardian Demon",
    "Pit Demon",
    "The Terror",
    # Logos
    "Help From Future Self",
    "Labwork",
    "Library Access",
    "Sloppy Labwork",
    "Wild Wormhole",
    "Wild Wormhole",
    "Wild Wormhole",
    "Library of Babble",
    "Mother",
    "Mother",
    "Quixo the Adventurer",
    "Timetraveler",
    # Shadows
    "Ghostly Hand",
    "Miasma",
    "One Last Job",
    "Oubliette",
    "Pawn Sacrifice",
    "Relentless Whispers",
    "Relentless Whispers",
    "Relentless Whispers",
    "Too Much To Protect",
    "Bad Penny",
    "Old Bruno",
    "Duskrunner",
)

DECKS = {
    "fignor": FIGNOR_LIST,
    "igor": IGOR_LIST,
}


def build_deck(deck_name: str, owner: int) -> List[Card]:
    names = DECKS[deck_name.lower()]
    return [Card(get_card_def(name), owner) for name in names]

"""When mirror augmentation is exact (Agent Training Plan, M7/M11).

Reversing both battlelines and swapping CHOOSE_FLANK's left/right is an
exact symmetry of the game unless something breaks it: a card whose text
names a side, or one of the four engine sites that put a creature on the
*left* flank regardless of choice -- Overlord Greking, Collar of
Subordination and Harland Mindlock (both through the temporary-control
revert) and Spangler Box. None is in Fignor or Igor, so augmentation is
exact there; for any other pool, check first (the learner does).
"""

from __future__ import annotations

import re
from typing import Iterable

from keyforge.cards.card_data import CARD_DEFS
from keyforge.cards.decks import resolve_deck

MIRROR_UNSAFE_CARDS = frozenset({"Overlord Greking", "Collar of Subordination", "Harland Mindlock", "Spangler Box"})
_SIDE_WORDS = re.compile(r"\b(left|right)\b", re.IGNORECASE)


def card_breaks_mirror(name: str) -> bool:
    cdef = CARD_DEFS[name]
    return name in MIRROR_UNSAFE_CARDS or bool(_SIDE_WORDS.search(cdef.text or ""))


def mirror_safe(decks: Iterable) -> bool:
    """True if no card in any of `decks` (preset names, paths or `Deck`s)
    breaks the left/right symmetry."""
    for deck in decks:
        for name in resolve_deck(deck).all_card_names():
            if card_breaks_mirror(name):
                return False
    return True

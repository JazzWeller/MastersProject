"""Regression for the Milestone L `__slots__` migration (Code/
AGENT_INTERFACE_PLAN.md: "__slots__ on Card and the type objects" +
"slots=True on the hot dataclasses"). Two things must both hold for this to
be a real win and not a latent bug generator:

1. No instance actually has a `__dict__` (otherwise `__slots__` bought
   nothing -- a base class without its own `__slots__` silently gives
   every subclass a `__dict__` back).
2. Setting an attribute NOT in `__slots__` raises `AttributeError` -- this
   is the actual enforcement that stops a future card effect from adding a
   new ad hoc dynamic attribute the way Ember Imp's `_ember_imp_effect`
   used to (see keyforge/cards/card.py and keyforge/state_hash.py's own
   comments on that).
"""

import unittest

from keyforge.actions import DiscardCard, EndTurn, Fight, PlayCard, Reap, UseAction, UseOmni
from keyforge.cards.card import ActionType, ArtifactType, Card, CreatureType, TypeObject, UpgradeType
from keyforge.cards.decks import build_deck
from keyforge.decision import Decision
from keyforge.enums import DecisionKind
from keyforge.log import LogEvent


def _a_card() -> Card:
    return build_deck("fignor", 1)[0]


class TestNoInstanceHasADict(unittest.TestCase):
    def test_card_has_no_dict(self):
        self.assertFalse(hasattr(_a_card(), "__dict__"))

    def test_type_objects_have_no_dict(self):
        card = _a_card()
        self.assertFalse(hasattr(card.type_object, "__dict__"))
        for cls, args in ((CreatureType, (5, 0)), (ArtifactType, ()), (UpgradeType, ()), (ActionType, ())):
            self.assertFalse(hasattr(cls(*args), "__dict__"))

    def test_action_dataclasses_have_no_dict(self):
        card = _a_card()
        for instance in (PlayCard(card), DiscardCard(card), UseAction(card), UseOmni(card), Reap(card), Fight(card), EndTurn()):
            self.assertFalse(hasattr(instance, "__dict__"))

    def test_decision_has_no_dict(self):
        d = Decision(player=1, kind=DecisionKind.CHOOSE_ACTION, prompt="x", options=[EndTurn()])
        self.assertFalse(hasattr(d, "__dict__"))

    def test_log_event_has_no_dict(self):
        self.assertFalse(hasattr(LogEvent("gain", {"player": 1, "amount": 1}), "__dict__"))


class TestUndeclaredAttributeIsRejected(unittest.TestCase):
    def test_card_rejects_a_new_ad_hoc_attribute(self):
        card = _a_card()
        with self.assertRaises(AttributeError):
            card.something_nobody_declared = 1

    def test_type_object_rejects_a_new_ad_hoc_attribute(self):
        card = _a_card()
        with self.assertRaises(AttributeError):
            card.type_object.something_nobody_declared = 1

    def test_decision_rejects_a_new_ad_hoc_attribute(self):
        d = Decision(player=1, kind=DecisionKind.CHOOSE_ACTION, prompt="x", options=[EndTurn()])
        with self.assertRaises(AttributeError):
            d.something_nobody_declared = 1


class TestEmberImpAttributeStillWorks(unittest.TestCase):
    """The one attribute that used to be truly ad hoc (never declared in
    __init__) is now a declared slot, defaulting to None -- must still be
    assignable exactly the way dis.py's own effect sets it."""

    def test_declared_and_defaults_to_none_and_is_assignable(self):
        card = _a_card()
        self.assertIsNone(card._ember_imp_effect)
        card._ember_imp_effect = object()
        self.assertIsNotNone(card._ember_imp_effect)


if __name__ == "__main__":
    unittest.main()

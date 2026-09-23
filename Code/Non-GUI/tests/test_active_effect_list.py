"""Regression for the Milestone L `ActiveEffectList` indexing change
(Code/AGENT_INTERFACE_PLAN.md, "index active effects by (player, variable)"):
`duration_effects_for` now reads a `(variable, player_affected)` index
instead of scanning every active effect, kept up to date in `add`,
`remove_from_source`, and `end_of_turn_tick` -- the only three places the
underlying list ever changes. These tests check the index directly rather
than only through whole-game play, so a future change to any of those three
methods that forgets to update it fails here, not three call frames deep in
a card effect.
"""

import unittest

from keyforge.effects.effect_object import INFINITE, ActiveEffectList, DurationEffect, TriggerEffect


def _duration(variable="CanPlayCards", player=1, duration=INFINITE, value=True, source=None):
    return DurationEffect(source, player, duration, player, variable, "=", value)


class TestActiveEffectListIndexing(unittest.TestCase):
    def test_add_makes_the_effect_findable_by_its_own_key_only(self):
        effects = ActiveEffectList()
        e = _duration(variable="CanPlayCards", player=1)
        effects.add(e)
        self.assertEqual(effects.duration_effects_for("CanPlayCards", 1), [e])
        self.assertEqual(effects.duration_effects_for("CanPlayCards", 2), [])
        self.assertEqual(effects.duration_effects_for("CanPlayActions", 1), [])

    def test_a_second_effect_on_a_different_key_does_not_leak_into_the_first(self):
        effects = ActiveEffectList()
        e1 = _duration(variable="CanPlayCards", player=1)
        e2 = _duration(variable="CanPlayActions", player=2)
        effects.add(e1)
        effects.add(e2)
        self.assertEqual(effects.duration_effects_for("CanPlayCards", 1), [e1])
        self.assertEqual(effects.duration_effects_for("CanPlayActions", 2), [e2])

    def test_remove_from_source_drops_it_from_the_index(self):
        effects = ActiveEffectList()
        card = object()
        e = _duration(variable="CanPlayCards", player=1, source=card)
        effects.add(e)
        effects.remove_from_source(card)
        self.assertEqual(effects.duration_effects_for("CanPlayCards", 1), [])
        self.assertEqual(effects.duration_effects, [])

    def test_remove_from_source_only_drops_effects_from_that_card(self):
        effects = ActiveEffectList()
        card_a, card_b = object(), object()
        e_a = _duration(variable="CanPlayCards", player=1, source=card_a)
        e_b = _duration(variable="CanPlayCards", player=1, source=card_b)
        effects.add(e_a)
        effects.add(e_b)
        effects.remove_from_source(card_a)
        self.assertEqual(effects.duration_effects_for("CanPlayCards", 1), [e_b])

    def test_end_of_turn_tick_removes_expired_effects_from_the_index(self):
        effects = ActiveEffectList()
        e = _duration(variable="CanPlayCards", player=1, duration=1)
        effects.add(e)
        effects.end_of_turn_tick()
        self.assertEqual(effects.duration_effects_for("CanPlayCards", 1), [])
        self.assertEqual(effects.duration_effects, [])

    def test_end_of_turn_tick_keeps_unexpired_effects_in_the_index(self):
        effects = ActiveEffectList()
        e = _duration(variable="CanPlayCards", player=1, duration=2)
        effects.add(e)
        effects.end_of_turn_tick()
        self.assertEqual(effects.duration_effects_for("CanPlayCards", 1), [e])
        self.assertEqual(e.remaining_duration, 1, "tick() must run exactly once per effect per call")

    def test_infinite_duration_effects_survive_ticking_and_stay_indexed(self):
        effects = ActiveEffectList()
        e = _duration(variable="CanPlayCards", player=1, duration=INFINITE)
        effects.add(e)
        for _ in range(5):
            effects.end_of_turn_tick()
        self.assertEqual(effects.duration_effects_for("CanPlayCards", 1), [e])

    def test_trigger_effects_are_unaffected_by_the_duration_index(self):
        effects = ActiveEffectList()
        card = object()
        t = TriggerEffect(card, 1, "key_forged", lambda game, event: (yield))
        effects.add(t)
        self.assertEqual(effects.triggers_for("key_forged"), [t])
        effects.remove_from_source(card)
        self.assertEqual(effects.triggers_for("key_forged"), [])


if __name__ == "__main__":
    unittest.main()

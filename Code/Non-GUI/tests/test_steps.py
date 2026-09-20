"""Direct tests for effects/steps.py's armor absorption in deal_damage
(Code/PHASE_3_PLAN.md Milestone F: the "armor never absorbs more than is
available" guarantee, pinned down here instead of as a fuzz invariant --
see test_cards_sanctum.TestSanctumUpgrades for why a conditional armor
bonus makes that unsound to check after the fact across a whole game)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.helpers import new_game, put_creature

from keyforge.effects import steps


class TestDealDamageArmor(unittest.TestCase):
    def test_absorbs_up_to_available_armor_then_the_rest_gets_through(self):
        game = new_game()
        bulwark = put_creature(game, 1, "Bulwark")  # power 4, armor 2
        steps.deal_damage(game, bulwark, 5)
        self.assertEqual(bulwark.type_object.armor_used_this_turn, 2)
        self.assertEqual(bulwark.type_object.damage, 3)

    def test_armor_used_accumulates_and_caps_across_multiple_hits(self):
        game = new_game()
        bulwark = put_creature(game, 1, "Bulwark")  # armor 2
        steps.deal_damage(game, bulwark, 1)  # 1 absorbed, 0 through
        self.assertEqual(bulwark.type_object.armor_used_this_turn, 1)
        self.assertEqual(bulwark.type_object.damage, 0)
        steps.deal_damage(game, bulwark, 3)  # only 1 armor left -> 1 absorbed, 2 through
        self.assertEqual(bulwark.type_object.armor_used_this_turn, 2)
        self.assertEqual(bulwark.type_object.damage, 2)
        steps.deal_damage(game, bulwark, 2)  # armor exhausted -> all of it through
        self.assertEqual(bulwark.type_object.armor_used_this_turn, 2)
        self.assertEqual(bulwark.type_object.damage, 4)

    def test_ignore_armor_bypasses_it_entirely(self):
        game = new_game()
        bulwark = put_creature(game, 1, "Bulwark")
        steps.deal_damage(game, bulwark, 3, ignore_armor=True)
        self.assertEqual(bulwark.type_object.armor_used_this_turn, 0)
        self.assertEqual(bulwark.type_object.damage, 3)

    def test_zero_or_negative_damage_is_a_no_op(self):
        game = new_game()
        bulwark = put_creature(game, 1, "Bulwark")
        self.assertIsNone(steps.deal_damage(game, bulwark, 0))
        self.assertEqual(bulwark.type_object.damage, 0)
        self.assertEqual(bulwark.type_object.armor_used_this_turn, 0)


if __name__ == "__main__":
    unittest.main()

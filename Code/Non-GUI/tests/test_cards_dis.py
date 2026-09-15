import unittest

from helpers import hand_card, make_card, new_game, put_artifact, put_creature, run_hook

from keyforge.enums import House
from keyforge.effects import named


class TestDis(unittest.TestCase):
    def test_arise(self):
        game = new_game()
        p1 = game.players[1]
        dust_imp = make_card("Dust Imp", 1)
        p1.discard.push(dust_imp)
        card = make_card("Arise", 1)
        run_hook(game, named.arise, card, answers=[House.DIS])
        self.assertIn(dust_imp, p1.hand.cards())
        self.assertNotIn(dust_imp, p1.discard.cards())
        self.assertEqual(p1.chains, 1)

    def test_control_the_weak(self):
        game = new_game()
        card = make_card("Control the Weak", 1)
        run_hook(game, named.control_the_weak, card, answers=[House.SHADOWS])
        self.assertEqual(game.players[2].get_house_selection(game), House.SHADOWS)

    def test_creeping_oblivion(self):
        game = new_game()
        p2 = game.players[2]
        c1, c2, c3 = (make_card("Old Bruno", 2) for _ in range(3))
        p2.discard.push(c1)
        p2.discard.push(c2)
        p2.discard.push(c3)
        card = make_card("Creeping Oblivion", 1)
        # Only player 2's discard pile is non-empty, so the pile choice is
        # forced and doesn't consume an answer -- only the card selection does.
        run_hook(game, named.creeping_oblivion, card, answers=[[c1, c2]])
        self.assertIn(c1, p2.purged.cards())
        self.assertIn(c2, p2.purged.cards())
        self.assertIn(c3, p2.discard.cards())

    def test_dominator_bauble_uses_a_friendly_creature(self):
        game = new_game()
        p1 = game.players[1]
        dust_imp = put_creature(game, 1, "Dust Imp", exhausted=False)
        card = make_card("Dominator Bauble", 1)
        before = p1.aember
        run_hook(game, named.dominator_bauble, card, answers=[[dust_imp]])
        self.assertEqual(p1.aember, before + 1)  # reap gained 1 aember
        self.assertTrue(dust_imp.Exhausted)

    def test_dust_imp_destroyed_gains_aember(self):
        game = new_game()
        p1 = game.players[1]
        dust_imp = put_creature(game, 1, "Dust Imp")
        before = p1.aember
        gen = game.destroy_cards([dust_imp])
        try:
            next(gen)
        except StopIteration:
            pass
        self.assertEqual(p1.aember, before + 2)

    def test_ember_imp_limits_enemy_card_plays(self):
        game = new_game()
        ember_imp = make_card("Ember Imp", 1)
        ember_imp.card_def.register_passive(game, ember_imp)
        self.assertEqual(game.players[2].get_card_played_limit(game), 2)
        ember_imp.card_def.unregister_passive(game, ember_imp)
        self.assertIsNone(game.players[2].get_card_played_limit(game))

    def test_gateway_to_dis_destroys_all_creatures(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        a = put_creature(game, 1, "Pit Demon")
        b = put_creature(game, 2, "Doc Bookton")
        card = make_card("Gateway to Dis", 1)
        run_hook(game, named.gateway_to_dis, card)
        self.assertNotIn(a, p1.play_area.creatures)
        self.assertNotIn(b, p2.play_area.creatures)
        self.assertIn(a, p1.discard.cards())
        self.assertIn(b, p2.discard.cards())
        self.assertEqual(p1.chains, 3)

    def test_guardian_demon_heals_and_redistributes_damage(self):
        game = new_game()
        p1 = game.players[1]
        damaged = put_creature(game, 1, "Doc Bookton")  # power 5
        damaged.type_object.damage = 2
        target = put_creature(game, 1, "Dust Imp")  # power 2, undamaged
        card = make_card("Guardian Demon", 1)
        run_hook(game, named.guardian_demon, card, answers=[[damaged], [target]])
        self.assertEqual(damaged.type_object.damage, 0)
        # 2 damage to a 2-power creature destroys it (damage resets on leaving play)
        self.assertIn(target, p1.discard.cards())
        self.assertNotIn(target, p1.play_area.creatures)

    def test_lash_of_broken_dreams_raises_enemy_key_cost(self):
        game = new_game()
        card = put_artifact(game, 1, "Lash of Broken Dreams", exhausted=False)
        base_cost = game.players[2].get_key_forge_cost(game)
        run_hook(game, card.card_def.on_action, card)
        self.assertEqual(game.players[2].get_key_forge_cost(game), base_cost + 3)

    def test_library_of_the_damned_archives(self):
        game = new_game()
        p1 = game.players[1]
        c = hand_card(game, 1, "Urchin")
        card = put_artifact(game, 1, "Library of the Damned", exhausted=False)
        run_hook(game, card.card_def.on_action, card, answers=[[c]])
        self.assertIn(c, p1.archive.cards())

    def test_lifeward_omni_sacrifices_and_locks_enemy_creatures(self):
        game = new_game()
        p1 = game.players[1]
        card = put_artifact(game, 1, "Lifeward", exhausted=False)
        run_hook(game, named.lifeward_omni, card)
        self.assertNotIn(card, p1.play_area.artifacts)
        self.assertIn(card, p1.discard.cards())
        self.assertFalse(game.players[2].get_can_play_creatures(game))

    def test_pit_demon_steal_action(self):
        game = new_game()
        game.players[2].aember = 3
        card = put_creature(game, 1, "Pit Demon", exhausted=False)
        run_hook(game, card.card_def.on_action, card)
        self.assertEqual(game.players[1].aember, 1)
        self.assertEqual(game.players[2].aember, 2)

    def test_shooler_steals_when_enemy_has_four_or_more(self):
        game = new_game()
        game.players[2].aember = 4
        card = put_creature(game, 1, "Shooler")
        run_hook(game, named.shooler_play, card)
        self.assertEqual(game.players[1].aember, 1)
        self.assertEqual(game.players[2].aember, 3)

    def test_shooler_no_steal_below_four(self):
        game = new_game()
        game.players[2].aember = 3
        card = put_creature(game, 1, "Shooler")
        run_hook(game, named.shooler_play, card)
        self.assertEqual(game.players[1].aember, 0)

    def test_snudge_returns_artifact_or_flank_creature(self):
        game = new_game()
        p2 = game.players[2]
        artifact = put_artifact(game, 2, "Subtle Maul")
        card = make_card("Snudge", 1)
        run_hook(game, named.snudge, card, answers=[[artifact]])
        self.assertIn(artifact, p2.hand.cards())
        self.assertNotIn(artifact, p2.play_area.artifacts)

    def test_succubus_lowers_enemy_draw_limit(self):
        game = new_game()
        succubus = make_card("Succubus", 1)
        succubus.card_def.register_passive(game, succubus)
        self.assertEqual(game.players[2].get_draw_up_to_limit(game), 5)

    def test_the_terror_gains_when_enemy_has_none(self):
        game = new_game()
        game.players[2].aember = 0
        card = put_creature(game, 1, "The Terror")
        run_hook(game, named.the_terror_play, card)
        self.assertEqual(game.players[1].aember, 2)

    def test_the_terror_no_gain_if_enemy_has_aember(self):
        game = new_game()
        game.players[2].aember = 1
        card = put_creature(game, 1, "The Terror")
        run_hook(game, named.the_terror_play, card)
        self.assertEqual(game.players[1].aember, 0)

    def test_three_fates_destroys_three_most_powerful(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        c1 = put_creature(game, 1, "Titan Mechanic")  # power 6
        c2 = put_creature(game, 2, "Doc Bookton")  # power 5
        c3 = put_creature(game, 2, "The Terror")  # power 5
        c4 = put_creature(game, 1, "Dust Imp")  # power 2, should survive
        card = make_card("Three Fates", 1)
        run_hook(game, named.three_fates, card)
        self.assertIn(c1, p1.discard.cards())
        self.assertIn(c2, p2.discard.cards())
        self.assertIn(c3, p2.discard.cards())
        self.assertIn(c4, p1.play_area.creatures)


if __name__ == "__main__":
    unittest.main()

import unittest

from helpers import drive, hand_card, make_card, new_game, put_artifact, put_creature, run_hook

from keyforge.effects import named


class TestShadows(unittest.TestCase):
    def test_bad_penny_destroyed_goes_to_hand(self):
        game = new_game()
        p1 = game.players[1]
        card = put_creature(game, 1, "Bad Penny")
        gen = game.destroy_cards([card])
        drive(gen)
        self.assertIn(card, p1.hand.cards())
        self.assertNotIn(card, p1.discard.cards())

    def test_bait_and_switch_repeats_while_behind(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        p2.aember = 3
        p1.aember = 0
        card = make_card("Bait and Switch", 1)
        run_hook(game, named.bait_and_switch, card)
        # steals 1 at a time until p1 is no longer strictly behind: 0->1 (1<2,cont)
        # ->2 (2<1? no) stop. p1 ends with 2, p2 with 1.
        self.assertEqual(p1.aember, 2)
        self.assertEqual(p2.aember, 1)

    def test_bait_and_switch_always_steals_once(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        p2.aember = 5
        p1.aember = 5
        card = make_card("Bait and Switch", 1)
        run_hook(game, named.bait_and_switch, card)
        self.assertEqual(p1.aember, 6)
        self.assertEqual(p2.aember, 4)

    def test_booby_trap_hits_center_and_neighbors(self):
        game = new_game()
        p1 = game.players[1]
        left = put_creature(game, 1, "Old Bruno")  # power 3
        center = put_creature(game, 1, "Doc Bookton")  # power 5
        right = put_creature(game, 1, "Titan Mechanic")  # power 6
        card = make_card("Booby Trap", 1)
        run_hook(game, named.booby_trap, card, answers=[[center]])
        self.assertEqual(center.type_object.damage, 4)
        self.assertEqual(left.type_object.damage, 2)
        self.assertEqual(right.type_object.damage, 2)

    def test_duskrunner_grants_after_reap_steal(self):
        game = new_game()
        p1 = game.players[1]
        p2 = game.players[2]
        p2.aember = 2
        host = put_creature(game, 1, "Pit Demon")
        upgrade = make_card("Duskrunner", 1)
        upgrade.type_object.host = host
        host.type_object.upgrades.append(upgrade)
        upgrade.card_def.register_passive(game, upgrade)
        drive(game._reap(1, host))
        self.assertEqual(p1.aember, 1 + 1)  # reap's own 1 + Duskrunner's steal 1
        self.assertEqual(p2.aember, 1)
        upgrade.card_def.unregister_passive(game, upgrade)
        self.assertEqual(host.extra_triggers["after_reap"], [])

    def test_ghostly_hand_steals_when_enemy_has_exactly_one(self):
        game = new_game()
        game.players[2].aember = 1
        card = make_card("Ghostly Hand", 1)
        run_hook(game, named.ghostly_hand, card)
        self.assertEqual(game.players[1].aember, 1)
        self.assertEqual(game.players[2].aember, 0)

    def test_ghostly_hand_no_steal_otherwise(self):
        game = new_game()
        game.players[2].aember = 2
        card = make_card("Ghostly Hand", 1)
        run_hook(game, named.ghostly_hand, card)
        self.assertEqual(game.players[1].aember, 0)

    def test_lights_out_returns_up_to_two(self):
        game = new_game()
        p2 = game.players[2]
        c1 = put_creature(game, 2, "Urchin")
        c2 = put_creature(game, 2, "Old Bruno")
        card = make_card("Lights Out", 1)
        run_hook(game, named.lights_out, card, answers=[[c1, c2]])
        self.assertIn(c1, p2.hand.cards())
        self.assertIn(c2, p2.hand.cards())
        self.assertEqual(len(p2.play_area.creatures), 0)

    def test_miasma_blocks_enemy_key_forging(self):
        game = new_game()
        card = make_card("Miasma", 1)
        run_hook(game, card.card_def.on_play, card)
        self.assertFalse(game.players[2].get_can_key_forge(game))

    def test_nerve_blast_steals_then_damages(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        p2.aember = 1
        target = put_creature(game, 2, "Old Bruno")
        card = make_card("Nerve Blast", 1)
        run_hook(game, named.nerve_blast, card, answers=[[target]])
        self.assertEqual(p1.aember, 1)
        self.assertEqual(p2.aember, 0)
        self.assertEqual(target.type_object.damage, 2)

    def test_nerve_blast_no_damage_if_steal_fails(self):
        game = new_game()
        p2 = game.players[2]
        p2.aember = 0
        target = put_creature(game, 2, "Old Bruno")
        card = make_card("Nerve Blast", 1)
        run_hook(game, named.nerve_blast, card)
        self.assertEqual(target.type_object.damage, 0)

    def test_noddy_elusive_and_steal_action(self):
        game = new_game()
        game.players[2].aember = 2
        noddy = put_creature(game, 1, "Noddy the Thief", exhausted=False)
        self.assertTrue(noddy.Elusive)
        run_hook(game, noddy.card_def.on_action, noddy)
        self.assertEqual(game.players[1].aember, 1)

    def test_old_bruno_captures_and_is_elusive(self):
        game = new_game()
        game.players[2].aember = 5
        bruno = put_creature(game, 1, "Old Bruno")
        self.assertTrue(bruno.Elusive)
        run_hook(game, bruno.card_def.on_play, bruno)
        self.assertEqual(bruno.aember_captured, 3)
        self.assertEqual(game.players[2].aember, 2)

    def test_one_last_job_purges_shadows_creatures_and_steals(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        p2.aember = 5
        s1 = put_creature(game, 1, "Urchin")  # Shadows
        s2 = put_creature(game, 1, "Old Bruno")  # Shadows
        non_shadows = put_creature(game, 1, "Doc Bookton")  # Logos, unaffected
        card = make_card("One Last Job", 1)
        run_hook(game, named.one_last_job, card)
        self.assertIn(s1, p1.purged.cards())
        self.assertIn(s2, p1.purged.cards())
        self.assertIn(non_shadows, p1.play_area.creatures)
        self.assertEqual(p1.aember, 2)
        self.assertEqual(p2.aember, 3)

    def test_oubliette_purges_low_power_creature(self):
        game = new_game()
        p2 = game.players[2]
        low = put_creature(game, 2, "Urchin")  # power 1
        high = put_creature(game, 2, "Titan Mechanic")  # power 6
        card = make_card("Oubliette", 1)
        run_hook(game, named.oubliette, card, answers=[[low]])
        self.assertIn(low, p2.purged.cards())
        self.assertIn(high, p2.play_area.creatures)

    def test_pawn_sacrifice(self):
        game = new_game()
        p1 = game.players[1]
        victim = put_creature(game, 1, "Urchin")
        friendly = put_creature(game, 1, "Titan Mechanic")  # power 6, survives 3 damage
        enemy = put_creature(game, 2, "Doc Bookton")  # power 5, survives 3 damage
        card = make_card("Pawn Sacrifice", 1)
        run_hook(game, named.pawn_sacrifice, card, answers=[[victim], [friendly, enemy]])
        self.assertIn(victim, p1.discard.cards())
        self.assertEqual(friendly.type_object.damage, 3)
        self.assertEqual(enemy.type_object.damage, 3)

    def test_pawn_sacrifice_no_effect_without_a_creature(self):
        game = new_game()
        p1 = game.players[1]
        card = make_card("Pawn Sacrifice", 1)
        run_hook(game, named.pawn_sacrifice, card)  # no creatures: should be a no-op
        self.assertEqual(len(p1.discard), 0)

    def test_relentless_whispers_steals_on_destroy(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        p2.aember = 2
        target = put_creature(game, 2, "Urchin")  # power 1, dies to 2 damage
        card = make_card("Relentless Whispers", 1)
        run_hook(game, named.relentless_whispers, card, answers=[[target]])
        self.assertIn(target, p2.discard.cards())
        self.assertEqual(p1.aember, 1)
        self.assertEqual(p2.aember, 1)

    def test_relentless_whispers_no_steal_if_survives(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        p2.aember = 2
        target = put_creature(game, 2, "Doc Bookton")  # power 5, survives
        card = make_card("Relentless Whispers", 1)
        run_hook(game, named.relentless_whispers, card, answers=[[target]])
        self.assertIn(target, p2.play_area.creatures)
        self.assertEqual(p1.aember, 0)

    def test_silvertooth_readies_itself(self):
        game = new_game()
        card = put_creature(game, 1, "Silvertooth", exhausted=True)
        run_hook(game, card.card_def.on_play, card)
        self.assertFalse(card.Exhausted)

    def test_subtle_maul_discards_random_from_enemy(self):
        game = new_game()
        p2 = game.players[2]
        hand_card(game, 2, "Urchin")
        card = put_artifact(game, 1, "Subtle Maul", exhausted=False)
        before = len(p2.hand)
        run_hook(game, card.card_def.on_action, card)
        self.assertEqual(len(p2.hand), before - 1)
        self.assertEqual(len(p2.discard), 1)

    def test_too_much_to_protect(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        p2.aember = 9
        card = make_card("Too Much To Protect", 1)
        run_hook(game, named.too_much_to_protect, card)
        self.assertEqual(p1.aember, 3)
        self.assertEqual(p2.aember, 6)

    def test_too_much_to_protect_no_effect_below_six(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        p2.aember = 4
        card = make_card("Too Much To Protect", 1)
        run_hook(game, named.too_much_to_protect, card)
        self.assertEqual(p1.aember, 0)
        self.assertEqual(p2.aember, 4)

    def test_urchin_elusive_and_steal_on_play(self):
        game = new_game()
        game.players[2].aember = 3
        urchin = put_creature(game, 1, "Urchin")
        self.assertTrue(urchin.Elusive)
        run_hook(game, urchin.card_def.on_play, urchin)
        self.assertEqual(game.players[1].aember, 1)


if __name__ == "__main__":
    unittest.main()

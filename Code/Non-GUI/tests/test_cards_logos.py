import unittest

from helpers import drive, hand_card, make_card, new_game, put_artifact, put_creature, run_hook

from keyforge.effects import named
from keyforge.enums import House


class TestLogos(unittest.TestCase):
    def test_doc_bookton_after_reap_draws(self):
        game = new_game()
        p1 = game.players[1]
        card = put_creature(game, 1, "Doc Bookton", exhausted=False, can_be_used=True)
        before = len(p1.hand)
        run_hook(game, card.card_def.on_reap, card)
        self.assertEqual(len(p1.hand), before + 1)

    def test_help_from_future_self_finds_timetraveler_in_deck(self):
        game = new_game()
        p1 = game.players[1]
        tt = make_card("Timetraveler", 1)
        p1.deck.put_on_top(tt)
        card = make_card("Help From Future Self", 1)
        run_hook(game, named.help_from_future_self, card)
        self.assertIn(tt, p1.hand.cards())
        self.assertNotIn(tt, p1.deck.cards())

    def test_help_from_future_self_finds_timetraveler_in_discard(self):
        game = new_game()
        p1 = game.players[1]
        tt = make_card("Timetraveler", 1)
        p1.discard.push(tt)
        other = make_card("Urchin", 1)
        p1.discard.push(other)
        card = make_card("Help From Future Self", 1)
        run_hook(game, named.help_from_future_self, card)
        self.assertIn(tt, p1.hand.cards())
        self.assertIn(other, p1.deck.cards())
        self.assertEqual(len(p1.discard), 0)

    def test_labwork_archives_a_card(self):
        game = new_game()
        p1 = game.players[1]
        c = hand_card(game, 1, "Urchin")
        card = make_card("Labwork", 1)
        run_hook(game, card.card_def.on_play, card, answers=[[c]])
        self.assertIn(c, p1.archive.cards())

    def test_library_access_purges_self_and_registers_trigger(self):
        game = new_game()
        p1 = game.players[1]
        card = make_card("Library Access", 1)
        run_hook(game, named.library_access_play, card)
        self.assertIn(card, p1.purged.cards())
        triggers = game.active_effects.triggers_for("card_played")
        self.assertEqual(len(triggers), 1)
        self.assertIs(triggers[0].source_card, card)

    def test_library_access_trigger_draws_on_next_play(self):
        game = new_game()
        p1 = game.players[1]
        card = make_card("Library Access", 1)
        run_hook(game, named.library_access_play, card)
        other = make_card("Urchin", 1)
        before = len(p1.hand)
        drive(game._run_play_trigger_check(other))
        self.assertEqual(len(p1.hand), before + 1)

    def test_library_of_babble_draws(self):
        game = new_game()
        p1 = game.players[1]
        card = put_artifact(game, 1, "Library of Babble", exhausted=False)
        before = len(p1.hand)
        run_hook(game, card.card_def.on_action, card)
        self.assertEqual(len(p1.hand), before + 1)

    def test_mother_raises_own_draw_limit(self):
        game = new_game()
        mother = make_card("Mother", 1)
        mother.card_def.register_passive(game, mother)
        self.assertEqual(game.players[1].get_draw_up_to_limit(game), 7)
        mother.card_def.unregister_passive(game, mother)
        self.assertEqual(game.players[1].get_draw_up_to_limit(game), 6)

    def test_phase_shift_grants_non_logos_allowance(self):
        game = new_game()
        card = make_card("Phase Shift", 1)
        run_hook(game, card.card_def.on_play, card)
        self.assertEqual(game.players[1].NonLogosCardsPlayable, 1)

    def test_quixo_skirmish_and_after_fight_draw(self):
        game = new_game()
        p1 = game.players[1]
        quixo = put_creature(game, 1, "Quixo the Adventurer")
        self.assertTrue(quixo.Skirmish)
        before = len(p1.hand)
        run_hook(game, quixo.card_def.on_fight, quixo)
        self.assertEqual(len(p1.hand), before + 1)
        quixo.card_def.unregister_passive(game, quixo)
        self.assertFalse(quixo.Skirmish)

    def test_scrambler_storm_disables_enemy_actions(self):
        game = new_game()
        card = make_card("Scrambler Storm", 1)
        run_hook(game, card.card_def.on_play, card)
        self.assertFalse(game.players[2].get_can_play_actions(game))

    def test_sloppy_labwork_archives_then_discards(self):
        game = new_game()
        p1 = game.players[1]
        c1 = hand_card(game, 1, "Urchin")
        c2 = hand_card(game, 1, "Old Bruno")
        card = make_card("Sloppy Labwork", 1)
        run_hook(game, named.sloppy_labwork, card, answers=[[c1], [c2]])
        self.assertIn(c1, p1.archive.cards())
        self.assertIn(c2, p1.discard.cards())

    def test_the_howling_pit_raises_both_draw_up_to_limits(self):
        # Follows the printed card ("refills their hand to 1 additional
        # card"), not the spec's CardDrawModifier -- approved divergence.
        game = new_game()
        pit = make_card("The Howling Pit", 1)
        pit.card_def.register_passive(game, pit)
        self.assertEqual(game.players[1].get_draw_up_to_limit(game), 7)
        self.assertEqual(game.players[2].get_draw_up_to_limit(game), 7)
        self.assertEqual(game.players[1].get_card_draw_modifier(game), 0)

    def test_timetraveler_play_draws_two_and_action_shuffles_self(self):
        game = new_game()
        p1 = game.players[1]
        tt = put_creature(game, 1, "Timetraveler", exhausted=False)
        before = len(p1.hand)
        run_hook(game, tt.card_def.on_play, tt)
        self.assertEqual(len(p1.hand), before + 2)
        run_hook(game, tt.card_def.on_action, tt)
        self.assertNotIn(tt, p1.play_area.creatures)
        self.assertIn(tt, p1.deck.cards())

    def test_titan_mechanic_lowers_key_cost_only_on_flank(self):
        game = new_game()
        p1 = game.players[1]
        put_creature(game, 1, "Dust Imp")  # filler so Titan Mechanic can be non-flank
        titan = put_creature(game, 1, "Titan Mechanic")
        put_creature(game, 1, "Old Bruno")
        base = p1.get_key_forge_cost(game)
        # Titan Mechanic is in the center (not a flank) with two neighbors
        self.assertFalse(p1.play_area.is_flank(titan))
        self.assertEqual(p1.get_key_forge_cost(game), base)
        p1.play_area.remove(titan)
        p1.play_area.add_creature(titan, flank="left")
        self.assertTrue(p1.play_area.is_flank(titan))
        self.assertEqual(p1.get_key_forge_cost(game), base - 1)

    def test_wild_wormhole_plays_top_card_ignoring_house(self):
        game = new_game()
        p1 = game.players[1]
        p1.selected_house = House.LOGOS  # Urchin is Shadows: this proves house is ignored
        top = make_card("Urchin", 1)
        p1.deck.put_on_top(top)
        game.players[2].aember = 1
        card = make_card("Wild Wormhole", 1)
        run_hook(game, named.wild_wormhole, card)
        self.assertIn(top, p1.play_area.creatures)
        self.assertEqual(p1.aember, 1)  # stole 1 aember from Urchin's play effect

    def test_wild_wormhole_returns_uncastable_card_to_top(self):
        game = new_game()
        p1 = game.players[1]
        p1.CanPlayCreatures = False
        top = make_card("Urchin", 1)
        p1.deck.put_on_top(top)
        card = make_card("Wild Wormhole", 1)
        run_hook(game, named.wild_wormhole, card)
        self.assertNotIn(top, p1.play_area.creatures)
        self.assertIs(p1.deck.peek_top(), top)


if __name__ == "__main__":
    unittest.main()

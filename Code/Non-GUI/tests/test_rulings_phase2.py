"""MRB 18.3 FAQ rulings for the Phase 2 pool (Code/PHASE_2_CARD_RULINGS.md)
that no per-card test already pins down. Each test names the ruling it
checks; the rulings doc lists every ruling and where it is tested."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.helpers import drive, make_card, new_game, put_creature, run_hook

from keyforge.actions import PlayCard
from keyforge.cards.card import Card
from keyforge.cards.card_data import get_card_def
from keyforge.config import GameConfig
from keyforge.effects.named import brobnar, dis, logos, shadows
from keyforge.enums import DecisionKind, House
from keyforge.game import Game
from keyforge.zones import Deck


def _deck_card(game, pid, name):
    card = Card(get_card_def(name), pid)
    card.controller = pid
    game._cards_by_id[card.instance_id] = card
    return card


class TestShadowSelfRulings(unittest.TestCase):
    def test_special_delivery_destroys_shadow_self_but_does_not_purge_it(self):
        # Shadow Self FAQ: the redirected damage destroys Shadow Self, but
        # "that creature" is the chosen flank creature, so nothing is purged.
        game = new_game()
        flank = put_creature(game, 2, "Charette")
        shadow = put_creature(game, 2, "Shadow Self")  # power 9
        put_creature(game, 2, "Doc Bookton")
        shadow.type_object.damage = 6
        delivery = make_card("Special Delivery", 1)
        game.players[1].play_area.add_artifact(delivery)
        run_hook(game, shadows.special_delivery, delivery, answers=[[flank]])
        self.assertIn(flank, game.players[2].play_area.creatures)
        self.assertEqual(flank.type_object.damage, 0)
        self.assertIn(shadow, game.players[2].discard.cards())
        self.assertNotIn(shadow, game.players[2].purged.cards())

    def test_seeker_needle_redirect_destroys_shadow_self_without_the_gain(self):
        # Same ruling for "If this damage destroys that creature, gain 1".
        game = new_game()
        target = put_creature(game, 2, "Urchin")  # power 1
        shadow = put_creature(game, 2, "Shadow Self")
        shadow.type_object.damage = 8
        run_hook(game, shadows.seeker_needle, make_card("Seeker Needle", 1), answers=[[target]])
        self.assertIn(target, game.players[2].play_area.creatures)
        self.assertIn(shadow, game.players[2].discard.cards())
        self.assertEqual(game.players[1].aember, 0)

    def test_armor_prevents_before_the_redirect(self):
        # Armor FAQ: Raiding Knight (armor 2) next to Shadow Self, attacked by
        # a 4-power creature -> Shadow Self takes 2, the Knight takes none.
        game = new_game()
        knight = put_creature(game, 2, "Raiding Knight")
        shadow = put_creature(game, 2, "Shadow Self")
        attacker = put_creature(game, 1, "Charette")  # power 4
        drive(game._fight(1, attacker), answers=[[knight]])
        self.assertEqual(knight.type_object.damage, 0)
        self.assertEqual(shadow.type_object.damage, 2)
        self.assertNotIn(attacker, game.players[1].play_area.creatures)  # knight's power 4

    def test_bulwark_armor_does_not_protect_shadow_self_from_redirected_damage(self):
        # Shadow Self FAQ (Gluttony vs Bulwark): 6 damage, 2 prevented by
        # Bulwark's armor, 4 placed on Shadow Self -- Shadow Self's own
        # armor from Bulwark doesn't apply to damage already past armor.
        game = new_game()
        bulwark = put_creature(game, 2, "Bulwark")
        shadow = put_creature(game, 2, "Shadow Self")
        attacker = put_creature(game, 1, "Titan Mechanic")  # power 6
        drive(game._fight(1, attacker), answers=[[bulwark]])
        self.assertEqual(bulwark.type_object.damage, 0)
        self.assertEqual(shadow.type_object.damage, 4)


class TestDisRulings(unittest.TestCase):
    def test_key_hammer_unforges_only_one_of_two_keys(self):
        game = new_game()
        p2 = game.players[2]
        p2.keys = 2
        for keys in (1, 2):
            game.log.add("forge_key", player=2, keys=keys, cost=6, turn=game.turn_number - 1)
        run_hook(game, dis.key_hammer, make_card("Key Hammer", 1))
        self.assertEqual(p2.keys, 1)

    def test_magda_the_rat_aember_on_creatures_moves_before_her_steal(self):
        # Leaves Play FAQ: Gateway to Dis with 2 captured on my creature and
        # the opponent's Magda in play, both pools empty -> the 2 goes to
        # the opponent first, then Magda's leaves-play steal takes it back.
        game = new_game()
        mine = put_creature(game, 1, "Charette")
        mine.aember_captured = 2
        magda = make_card("Magda the Rat", 2)
        game.players[2].play_area.add_creature(magda)
        game._cards_by_id[magda.instance_id] = magda
        run_hook(game, shadows.magda_the_rat_play, magda)  # both pools empty: steals nothing
        run_hook(game, dis.gateway_to_dis, make_card("Gateway to Dis", 1))
        self.assertEqual(game.players[1].aember, 2)
        self.assertEqual(game.players[2].aember, 0)


class TestTolasRulings(unittest.TestCase):
    def test_a_bad_penny_returned_to_hand_still_counts_as_destroyed(self):
        game = new_game()
        put_creature(game, 1, "Tolas")
        penny = put_creature(game, 2, "Bad Penny")
        drive(game.destroy_cards([penny]))
        self.assertIn(penny, game.players[2].hand.cards())
        self.assertEqual(game.players[1].aember, 1)

    def test_tolas_taken_by_overlord_greking_does_not_trigger_on_itself(self):
        game = new_game()
        greking = put_creature(game, 1, "Overlord Greking")  # power 7
        tolas = put_creature(game, 2, "Tolas")  # power 1, elusive
        tolas.fought_this_turn = True  # elusive already spent this turn
        drive(game._fight(1, greking), answers=[[tolas]])
        self.assertIn(tolas, game.players[1].play_area.creatures)
        self.assertEqual(game.players[1].aember, 0)
        self.assertEqual(game.players[2].aember, 0)


class TestLogosRulings(unittest.TestCase):
    def test_phase_shift_lets_the_first_player_play_another_card(self):
        # First Turn Rule FAQ: card effects modify the rule; Phase Shift's
        # allowance lets the first player play a second (non-Logos) card.
        game = Game(GameConfig(decks=("igor", "igor"), first_player=1, seed=1))
        game.submit(False)
        game.submit(False)
        while game.pending_decision.kind != DecisionKind.CHOOSE_HOUSE:
            game.submit(game.pending_decision.options[0])
        phase_shift = _deck_card(game, 1, "Phase Shift")
        urchin = _deck_card(game, 1, "Urchin")
        logos_card = _deck_card(game, 1, "Batdrone")
        for c in (phase_shift, urchin, logos_card):
            game.players[1].hand.add(c)
        game.submit(House.LOGOS)
        while game.pending_decision.kind != DecisionKind.CHOOSE_ACTION:
            game.submit(game.pending_decision.options[0])
        game.submit(PlayCard(phase_shift))
        while game.pending_decision.kind != DecisionKind.CHOOSE_ACTION:
            game.submit(game.pending_decision.options[0])
        plays = [o.card for o in game.pending_decision.options if isinstance(o, PlayCard)]
        self.assertIn(urchin, plays)
        self.assertTrue(all(c.house != House.LOGOS for c in plays))  # only what the allowance covers
        self.assertIsNone(game.why_not_playable(1, urchin))
        self.assertIsNotNone(game.why_not_playable(1, logos_card))
        game.submit(PlayCard(urchin))
        while game.pending_decision.kind != DecisionKind.CHOOSE_ACTION:
            game.submit(game.pending_decision.options[0])
        self.assertFalse(any(isinstance(o, PlayCard) for o in game.pending_decision.options))

    def test_neutron_shark_that_destroys_itself_does_not_repeat(self):
        game = new_game()
        player = game.players[1]
        shark = put_creature(game, 1, "Neutron Shark")
        e1 = put_creature(game, 2, "Batdrone")
        e2 = put_creature(game, 2, "Dr. Escotera")
        player.deck = Deck([_deck_card(game, 1, "Charette")] + player.deck.cards())  # non-Logos top card
        run_hook(game, logos.neutron_shark, shark, answers=[[e1], [shark]])
        self.assertNotIn(e1, game.players[2].play_area.creatures)
        self.assertIn(e2, game.players[2].play_area.creatures)

    def test_two_phase_shifts_allow_two_non_logos_cards(self):
        game = new_game()
        for _ in range(2):
            run_hook(game, logos.phase_shift, make_card("Phase Shift", 1))
        self.assertEqual(game.players[1].get_non_logos_cards_playable(game), 2)

    def test_wild_wormhole_playing_a_non_logos_card_uses_phase_shifts_allowance(self):
        game = new_game()
        player = game.players[1]
        player.selected_house = House.LOGOS
        run_hook(game, logos.phase_shift, make_card("Phase Shift", 1))
        top = _deck_card(game, 1, "Urchin")  # Shadows action-free creature
        player.deck = Deck([top] + player.deck.cards())
        run_hook(game, logos.wild_wormhole, make_card("Wild Wormhole", 1))
        self.assertIn(top, player.play_area.creatures)
        self.assertEqual(player.get_non_logos_cards_playable(game), 0)

    def test_reverse_time_turns_the_deck_over_onto_the_discard_pile(self):
        game = new_game()
        player = game.players[1]
        deck_order = player.deck.cards()  # top first
        run_hook(game, logos.reverse_time, make_card("Reverse Time", 1))
        # The old top card of the deck is now the bottom of the discard pile.
        self.assertEqual(player.discard.cards(), deck_order)

    def test_replicator_on_an_enemy_sequis_captures_from_its_owners_pool(self):
        # "As If It Were" FAQ: using the opponent's Sequis as if you
        # controlled it captures 1 from your opponent's pool.
        game = new_game()
        replicator = put_creature(game, 1, "Replicator")
        sequis = put_creature(game, 2, "Sequis")
        game.players[1].aember = 3
        game.players[2].aember = 3
        run_hook(game, logos.replicator, replicator, answers=[[sequis]])
        self.assertEqual(game.players[2].aember, 2)
        self.assertEqual(game.players[1].aember, 3)
        self.assertEqual(sequis.aember_captured, 1)
        self.assertEqual(sequis.controller, 2)

    def test_replicator_on_an_enemy_sanctum_guardian_moves_nothing(self):
        # Moving Creatures FAQ: a creature can't move to the other player's
        # battleline without a control change, so the swap does nothing.
        game = new_game()
        replicator = put_creature(game, 1, "Replicator")
        mine = put_creature(game, 1, "Charette")
        guardian = put_creature(game, 2, "Sanctum Guardian")
        theirs = put_creature(game, 2, "Doc Bookton")
        before = (list(game.players[1].play_area.creatures), list(game.players[2].play_area.creatures))
        run_hook(game, logos.replicator, replicator, answers=[[guardian]])
        self.assertEqual(game.players[1].play_area.creatures, before[0])
        self.assertEqual(game.players[2].play_area.creatures, before[1])
        self.assertEqual(guardian.controller, 2)
        self.assertIn(mine, game.players[1].play_area.creatures)
        self.assertIn(theirs, game.players[2].play_area.creatures)


class TestShadowsRulings(unittest.TestCase):
    def test_bulleteye_killed_by_assault_was_not_destroyed_in_a_fight(self):
        # The Warchest FAQ: assault destruction happens before the fight.
        game = new_game()
        bear = put_creature(game, 1, "Ancient Bear")  # assault 2
        bulleteye = put_creature(game, 2, "Bulleteye")  # power 2
        drive(game._fight(1, bear), answers=[[bulleteye]])
        self.assertIn(bulleteye, game.players[2].discard.cards())
        warchest = make_card("The Warchest", 1)
        run_hook(game, brobnar.the_warchest, warchest)
        self.assertEqual(game.players[1].aember, 0)

    def test_mack_the_knife_can_destroy_itself_and_still_gain(self):
        game = new_game()
        mack = put_creature(game, 1, "Mack the Knife")  # power 3
        mack.type_object.damage = 2
        run_hook(game, shadows.mack_the_knife, mack, answers=[[mack]])
        self.assertIn(mack, game.players[1].discard.cards())
        self.assertEqual(game.players[1].aember, 1)


if __name__ == "__main__":
    unittest.main()

"""Tests for the 37 new Dis cards (Code/PHASE_2_PLAN.md Milestone C)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.helpers import drive, hand_card, make_card, new_game, put_artifact, put_creature, run_hook

from keyforge.cards.card_data import get_card_def
from keyforge.effects import named
from keyforge.enums import House


class TestDisPhase2(unittest.TestCase):
    def test_a_fair_game_discards_reveals_and_gains_both_ways(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        p1.deck._cards.clear()
        p2.deck._cards.clear()
        p1.hand.take_all()
        p2.hand.take_all()
        p1_top = make_card("Dust Imp", 1)  # Dis
        p1.deck.put_on_top(p1_top)
        p2_top = make_card("Doc Bookton", 2)  # Logos
        p2.deck.put_on_top(p2_top)
        p1.hand.add(make_card("Charette", 1))  # Dis, matches p1_top's house
        p2.hand.add(make_card("Mother", 2))  # Logos, matches p2_top's house
        card = make_card("A Fair Game", 1)
        run_hook(game, named.a_fair_game, card)
        # Each deck's top card is discarded into its own owner's discard
        # pile (it never changes hands) -- only the Æmber gain crosses sides.
        self.assertIn(p2_top, p2.discard.cards())
        self.assertIn(p1_top, p1.discard.cards())
        self.assertEqual(p1.aember, 1)  # 1 Logos card in p2's hand matched p2_top's house
        self.assertEqual(p2.aember, 1)  # 1 Dis card in p1's hand matched p1_top's house

    def test_dance_of_doom_destroys_creatures_matching_chosen_power(self):
        game = new_game()
        a = put_creature(game, 1, "Charette")  # power 4
        b = put_creature(game, 2, "Truebaru")  # power 7
        c = put_creature(game, 2, "Drumble")  # power 2
        card = make_card("Dance of Doom", 1)
        run_hook(game, named.dance_of_doom, card, answers=[4])
        self.assertNotIn(a, game.players[1].play_area.creatures)
        self.assertIn(b, game.players[2].play_area.creatures)
        self.assertIn(c, game.players[2].play_area.creatures)

    def test_fear_returns_an_enemy_creature(self):
        game = new_game()
        p2 = game.players[2]
        target = put_creature(game, 2, "Charette")
        card = make_card("Fear", 1)
        run_hook(game, named.fear, card, answers=[[target]])
        self.assertIn(target, p2.hand.cards())
        self.assertNotIn(target, p2.play_area.creatures)

    def test_gongoozle_discards_random_only_if_not_destroyed(self):
        game = new_game()
        p2 = game.players[2]
        survivor = put_creature(game, 2, "Truebaru")  # power 7, survives 3 damage
        p2.hand.add(make_card("Charette", 2))
        card = make_card("Gongoozle", 1)
        before = len(p2.hand)
        run_hook(game, named.gongoozle, card, answers=[[survivor]])
        self.assertEqual(survivor.type_object.damage, 3)
        self.assertEqual(len(p2.hand), before - 1)

    def test_guilty_hearts_destroys_only_creatures_with_aember_on_them(self):
        game = new_game()
        with_aember = put_creature(game, 1, "Charette")
        with_aember.aember_captured = 2
        without = put_creature(game, 2, "Drumble")
        card = make_card("Guilty Hearts", 1)
        run_hook(game, named.guilty_hearts, card)
        self.assertNotIn(with_aember, game.players[1].play_area.creatures)
        self.assertIn(without, game.players[2].play_area.creatures)

    def test_hand_of_dis_only_targets_non_flank_creatures(self):
        game = new_game()
        p2 = game.players[2]
        left = put_creature(game, 2, "Charette", flank="left")
        middle = put_creature(game, 2, "Drumble")
        right = put_creature(game, 2, "Truebaru", flank="right")
        self.assertEqual(list(p2.play_area.creatures), [left, middle, right])
        card = make_card("Hand of Dis", 1)
        run_hook(game, named.hand_of_dis, card, answers=[[middle]])
        self.assertNotIn(middle, p2.play_area.creatures)
        self.assertIn(left, p2.play_area.creatures)
        self.assertIn(right, p2.play_area.creatures)

    def test_hecatomb_destroys_only_dis_creatures_and_pays_controllers(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        dis1 = put_creature(game, 1, "Charette")
        dis2 = put_creature(game, 2, "Drumble")
        non_dis = put_creature(game, 2, "Doc Bookton")
        card = make_card("Hecatomb", 1)
        run_hook(game, named.hecatomb, card)
        self.assertNotIn(dis1, p1.play_area.creatures)
        self.assertNotIn(dis2, p2.play_area.creatures)
        self.assertIn(non_dis, p2.play_area.creatures)
        self.assertEqual(p1.aember, 1)
        self.assertEqual(p2.aember, 1)

    def test_tendrils_of_pain_deals_4_if_opponent_forged_last_turn(self):
        game = new_game()
        a = put_creature(game, 1, "Truebaru")
        b = put_creature(game, 2, "Truebaru")
        game.log.add("forge_key", player=2, keys=1, cost=6, turn=game.turn_number - 1)
        card = make_card("Tendrils of Pain", 1)
        run_hook(game, named.tendrils_of_pain, card)
        self.assertEqual(a.type_object.damage, 4)
        self.assertEqual(b.type_object.damage, 4)

    def test_tendrils_of_pain_deals_1_otherwise(self):
        game = new_game()
        a = put_creature(game, 1, "Truebaru")
        card = make_card("Tendrils of Pain", 1)
        run_hook(game, named.tendrils_of_pain, card)
        self.assertEqual(a.type_object.damage, 1)

    def test_hysteria_returns_every_creature_to_hand(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        a = put_creature(game, 1, "Charette")
        b = put_creature(game, 2, "Drumble")
        card = make_card("Hysteria", 1)
        run_hook(game, named.hysteria, card)
        self.assertIn(a, p1.hand.cards())
        self.assertIn(b, p2.hand.cards())

    def test_key_hammer_unforges_and_pays_opponent_6(self):
        game = new_game()
        p2 = game.players[2]
        p2.keys = 1
        game.log.add("forge_key", player=2, keys=1, cost=6, turn=game.turn_number - 1)
        card = make_card("Key Hammer", 1)
        run_hook(game, named.key_hammer, card)
        self.assertEqual(p2.keys, 0)
        self.assertEqual(p2.aember, 6)

    def test_key_hammer_no_unforge_without_last_turn_forge(self):
        game = new_game()
        p2 = game.players[2]
        p2.keys = 1
        card = make_card("Key Hammer", 1)
        run_hook(game, named.key_hammer, card)
        self.assertEqual(p2.keys, 1)  # unchanged
        self.assertEqual(p2.aember, 6)  # opponent still gains 6

    def test_mind_barb_discards_random_from_opponent(self):
        game = new_game()
        p2 = game.players[2]
        p2.hand.add(make_card("Charette", 2))
        card = make_card("Mind Barb", 1)
        before = len(p2.hand)
        run_hook(game, named.mind_barb, card)
        self.assertEqual(len(p2.hand), before - 1)

    def test_pandemonium_only_undamaged_creatures_capture(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        p2.aember = 5
        undamaged = put_creature(game, 1, "Charette")
        damaged = put_creature(game, 1, "Drumble")
        damaged.type_object.damage = 1
        card = make_card("Pandemonium", 1)
        run_hook(game, named.pandemonium, card)
        self.assertEqual(undamaged.aember_captured, 1)
        self.assertEqual(damaged.aember_captured, 0)

    def test_poltergeist_uses_and_destroys_an_artifact(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        art = put_artifact(game, 2, "Library of Babble", exhausted=False)  # Action: draw a card
        card = make_card("Poltergeist", 1)
        before = len(p1.hand)
        run_hook(game, named.poltergeist, card, answers=[[art]])
        self.assertEqual(len(p1.hand), before + 1)  # used "as if yours": p1 drew
        self.assertNotIn(art, p2.play_area.artifacts)

    def test_poltergeist_can_use_an_omni_only_artifact(self):
        # "Use" covers Omni as well as Action -- an artifact with no Action
        # at all must still be usable "as if it were yours".
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        self.assertIsNone(get_card_def("Lifeward").on_action)
        art = put_artifact(game, 2, "Lifeward", exhausted=False)
        card = make_card("Poltergeist", 1)
        run_hook(game, named.poltergeist, card, answers=[[art]])
        self.assertFalse(p2.get_can_play_creatures(game))  # Poltergeist's controller's opponent is restricted

    def test_use_artifact_ability_respects_the_borrowers_cannot_use_cards(self):
        # Skippy Timehog restricts the BORROWER (p1, using the artifact "as
        # if yours"), not the artifact's real controller (p2).
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        art = put_artifact(game, 2, "Library of Babble", exhausted=False)  # Action: draw a card
        skippy = put_creature(game, 2, "Skippy Timehog")
        run_hook(game, named.skippy_timehog, skippy)
        card = make_card("Poltergeist", 1)
        before = len(p1.hand)
        run_hook(game, named.poltergeist, card, answers=[[art]])
        self.assertEqual(len(p1.hand), before)  # blocked: p1 cannot use cards this turn

    def test_use_artifact_ability_pays_the_borrowers_tentacus_toll(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        p1.aember = 5
        art = put_artifact(game, 2, "Library of Babble", exhausted=False)
        tentacus = put_creature(game, 2, "Tentacus")
        named.tentacus_register(game, tentacus)
        card = make_card("Poltergeist", 1)
        before = len(p1.hand)
        run_hook(game, named.poltergeist, card, answers=[[art]])
        self.assertEqual(len(p1.hand), before + 1)  # the use went through
        self.assertEqual(p1.aember, 4)  # p1 (the borrower) paid the 1Æ toll
        self.assertEqual(p2.aember, 1)  # ...to Tentacus's controller

    def test_use_artifact_ability_blocked_when_borrower_cannot_afford_the_toll(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        p1.aember = 0
        art = put_artifact(game, 2, "Library of Babble", exhausted=False)
        tentacus = put_creature(game, 2, "Tentacus")
        named.tentacus_register(game, tentacus)
        card = make_card("Poltergeist", 1)
        before = len(p1.hand)
        run_hook(game, named.poltergeist, card, answers=[[art]])
        self.assertEqual(len(p1.hand), before)  # blocked: can't afford the toll
        self.assertEqual(p1.aember, 0)
        self.assertEqual(p2.aember, 0)

    def test_annihilation_ritual_purges_destroyed_creatures_instead_of_discard(self):
        game = new_game()
        p1 = game.players[1]
        ritual = put_artifact(game, 1, "Annihilation Ritual")
        target = put_creature(game, 2, "Drumble")
        gen = game.destroy_cards([target])
        drive(gen)
        self.assertIn(target, game.players[2].purged.cards())
        self.assertNotIn(target, game.players[2].discard.cards())

    def test_annihilation_ritual_does_not_purge_destroyed_artifacts(self):
        # Canonical text says "creature", not "card" -- an artifact
        # destroyed while Annihilation Ritual is in play still goes to the
        # discard pile as normal.
        game = new_game()
        put_artifact(game, 1, "Annihilation Ritual")
        target_artifact = put_artifact(game, 2, "Library of Babble")
        gen = game.destroy_cards([target_artifact])
        drive(gen)
        self.assertIn(target_artifact, game.players[2].discard.cards())
        self.assertNotIn(target_artifact, game.players[2].purged.cards())

    def test_key_to_dis_sacrifices_itself_and_destroys_all_creatures(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        key = put_artifact(game, 1, "Key to Dis", exhausted=False)
        a = put_creature(game, 1, "Charette")
        b = put_creature(game, 2, "Drumble")
        run_hook(game, named.key_to_dis, key)
        self.assertNotIn(key, p1.play_area.artifacts)
        self.assertNotIn(a, p1.play_area.creatures)
        self.assertNotIn(b, p2.play_area.creatures)

    def test_sacrificial_altar_purges_human_and_plays_from_discard(self):
        game = new_game()
        p1 = game.players[1]
        human = put_creature(game, 1, "Doc Bookton")  # Human, Scientist
        deck_creature = make_card("Charette", 1)
        p1.discard.push(deck_creature)
        altar = make_card("Sacrificial Altar", 1)
        run_hook(game, named.sacrificial_altar, altar, answers=[[human], [deck_creature]])
        self.assertIn(human, p1.purged.cards())
        self.assertIn(deck_creature, p1.play_area.creatures)

    def test_screaming_cave_shuffles_hand_and_discard_into_deck(self):
        game = new_game()
        p1 = game.players[1]
        h = hand_card(game, 1, "Charette")
        d = make_card("Drumble", 1)
        p1.discard.push(d)
        cave = make_card("Screaming Cave", 1)
        run_hook(game, named.screaming_cave, cave)
        self.assertEqual(len(p1.hand), 0)
        self.assertEqual(len(p1.discard), 0)
        self.assertIn(h, p1.deck.cards())
        self.assertIn(d, p1.deck.cards())

    def test_soul_snatcher_pays_owner_on_any_destroyed_creature(self):
        game = new_game()
        p2 = game.players[2]
        put_artifact(game, 2, "Soul Snatcher")
        victim = put_creature(game, 1, "Charette")  # owned by p1, but Soul Snatcher belongs to p2
        gen = game.destroy_cards([victim])
        drive(gen)
        self.assertEqual(game.players[1].aember, 1)  # victim's OWNER gains, not Soul Snatcher's controller
        self.assertEqual(p2.aember, 0)

    def test_drumble_captures_all_if_opponent_has_7_or_more(self):
        game = new_game()
        p2 = game.players[2]
        p2.aember = 9
        drumble = make_card("Drumble", 1)
        run_hook(game, named.drumble_play, drumble)
        self.assertEqual(drumble.aember_captured, 9)
        self.assertEqual(p2.aember, 0)

    def test_eater_of_the_dead_purges_and_gains_power_counter(self):
        game = new_game()
        p1 = game.players[1]
        eater = put_creature(game, 1, "Eater of the Dead")
        fodder = make_card("Charette", 2)
        game.players[2].discard.push(fodder)
        run_hook(game, named.eater_of_the_dead, eater, answers=[[fodder]])
        self.assertIn(fodder, game.players[2].purged.cards())
        self.assertEqual(eater.power_counters, 1)
        self.assertEqual(game.get_power(eater), 5)

    def test_gabos_longarms_redirects_its_own_fight_damage(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        gabos = put_creature(game, 1, "Gabos Longarms", exhausted=False, can_be_used=True)  # power 5
        target = put_creature(game, 2, "Charette")  # power 4, not elusive: the fight target
        bystander = put_creature(game, 2, "Doc Bookton")  # no keywords: the redirect target
        bystander.type_object.base_power = 10  # survives the redirected hit, so its damage total is checkable
        gen = game._fight(1, gabos)
        drive(gen, answers=[[bystander], [target]])
        self.assertEqual(target.type_object.damage, 0)  # Gabos's damage was redirected away
        self.assertEqual(bystander.type_object.damage, 5)
        self.assertEqual(gabos.type_object.damage, 4)  # still takes the target's counter-damage

    def test_gabos_longarms_redirect_target_is_actually_destroy_checked(self):
        # The redirect target is a third creature, not the original
        # attacker/target -- it must still go through the normal destroy
        # check when the redirected damage is lethal.
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        gabos = put_creature(game, 1, "Gabos Longarms", exhausted=False, can_be_used=True)  # power 5
        target = put_creature(game, 2, "Charette")  # power 4, not elusive: the fight target
        bystander = put_creature(game, 2, "Doc Bookton")  # power 5: exactly lethal to the redirected hit
        gen = game._fight(1, gabos)
        drive(gen, answers=[[bystander], [target]])
        self.assertNotIn(bystander, p2.play_area.creatures)
        self.assertIn(bystander, p2.discard.cards())

    def test_overlord_greking_takes_control_of_creature_it_kills(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        greking = put_creature(game, 1, "Overlord Greking", exhausted=False, can_be_used=True)  # power 7
        weak = put_creature(game, 2, "Charette")  # power 4, not elusive: dies to Greking's counter-hit
        gen = game._fight(1, greking)
        drive(gen, answers=[[weak]])
        self.assertIn(weak, p1.play_area.creatures)
        self.assertEqual(weak.controller, 1)
        self.assertNotIn(weak, p2.play_area.creatures)

    def test_stealer_of_souls_purges_the_victim_and_gains_1(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        stealer = put_creature(game, 1, "Stealer of Souls", exhausted=False, can_be_used=True)  # power 6
        weak = put_creature(game, 2, "Charette")  # power 4: dies to Stealer's counter-hit
        gen = game._fight(1, stealer)
        drive(gen, answers=[[weak]])
        self.assertNotIn(weak, p2.discard.cards())
        self.assertIn(weak, p2.purged.cards())
        self.assertEqual(p1.aember, 1)

    def test_stealer_of_souls_still_gains_1_but_cannot_purge_a_bad_penny(self):
        # Confirmed ruling: Bad Penny's own "Destroyed:" ability returns it
        # to hand before Stealer of Souls' trigger runs, so there is
        # nothing left in the discard pile to purge -- but the Æmber gain
        # still happens, since Bad Penny was still destroyed.
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        stealer = put_creature(game, 1, "Stealer of Souls", exhausted=False, can_be_used=True)  # power 6
        penny = put_creature(game, 2, "Bad Penny")  # power 1: dies to Stealer's counter-hit
        gen = game._fight(1, stealer)
        drive(gen, answers=[[penny]])
        self.assertIn(penny, p2.hand.cards())
        self.assertNotIn(penny, p2.discard.cards())
        self.assertNotIn(penny, p2.purged.cards())
        self.assertEqual(p1.aember, 1)

    def test_stealer_of_souls_does_not_trigger_if_it_also_dies(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        stealer = put_creature(game, 1, "Stealer of Souls", exhausted=False, can_be_used=True)  # power 6
        equal = put_creature(game, 2, "Charette")
        equal.type_object.base_power = 6  # trades evenly: both die
        gen = game._fight(1, stealer)
        drive(gen, answers=[[equal]])
        self.assertIn(equal, p2.discard.cards())  # ordinary discard, not purged
        self.assertEqual(p1.aember, 0)

    def test_master_of_1_may_destroy_1_power_creature(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        master = put_creature(game, 1, "Master of 1", exhausted=False, can_be_used=True)
        weak = put_creature(game, 2, "Charette")
        weak.type_object.base_power = 1
        run_hook(game, named.master_of_n(1), master, answers=[True, [weak]])
        self.assertNotIn(weak, p2.play_area.creatures)

    def test_master_of_1_may_decline(self):
        game = new_game()
        p2 = game.players[2]
        weak = put_creature(game, 2, "Charette")
        weak.type_object.base_power = 1
        master = put_creature(game, 1, "Master of 1")
        run_hook(game, named.master_of_n(1), master, answers=[False])
        self.assertIn(weak, p2.play_area.creatures)

    def test_pitlord_forces_dis_as_active_house(self):
        game = new_game()
        put_creature(game, 1, "Pitlord")  # put_creature already calls register_passive
        self.assertEqual(game.players[1].get_house_selection(game), House.DIS)

    def test_restringuntus_bans_a_house_until_it_leaves_play(self):
        game = new_game()
        p2 = game.players[2]
        restr = put_creature(game, 1, "Restringuntus")
        run_hook(game, named.restringuntus, restr, answers=[House.SHADOWS])
        self.assertIn(House.SHADOWS, p2.get_cannot_choose_houses(game))
        game.players[1].play_area.remove(restr)
        game.leave_play(restr)
        self.assertNotIn(House.SHADOWS, p2.get_cannot_choose_houses(game))

    def test_shaffles_drains_opponent_at_end_of_controllers_turn(self):
        game = new_game()
        p2 = game.players[2]
        p2.aember = 3
        put_creature(game, 1, "Shaffles")  # put_creature already calls register_passive
        drive(game._fire_event("end_of_turn", {"player": 1}))
        self.assertEqual(p2.aember, 2)

    def test_shaffles_does_not_drain_on_opponents_own_turn_end(self):
        game = new_game()
        p1 = game.players[1]
        p1.aember = 3
        put_creature(game, 2, "Shaffles")
        drive(game._fire_event("end_of_turn", {"player": 1}))
        self.assertEqual(p1.aember, 3)  # p1 is not Shaffles's controller's opponent-at-end-of-controller's-turn

    def test_tocsin_after_reap_discards_random(self):
        game = new_game()
        p2 = game.players[2]
        p2.hand.add(make_card("Charette", 2))
        tocsin = put_creature(game, 1, "Tocsin", exhausted=False, can_be_used=True)
        before = len(p2.hand)
        run_hook(game, named.tocsin, tocsin)
        self.assertEqual(len(p2.hand), before - 1)

    def test_tolas_opponent_of_destroyed_owner_gains(self):
        game = new_game()
        put_creature(game, 2, "Tolas")  # put_creature already calls register_passive
        victim = put_creature(game, 1, "Charette")  # owned by p1
        gen = game.destroy_cards([victim])
        drive(gen)
        self.assertEqual(game.players[2].aember, 1)  # p1's opponent (p2) gains

    def test_tolas_destroyed_in_the_same_batch_gains_nothing(self):
        # Confirmed ruling: Tolas + Gateway to Dis (or any mass-destroy that
        # kills Tolas along with everything else) gains nothing at all --
        # Tolas isn't around to see any of the batch, including its own death.
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        tolas = put_creature(game, 2, "Tolas")  # put_creature already calls register_passive
        bystander = put_creature(game, 1, "Charette")
        gen = game.destroy_cards([tolas, bystander])
        drive(gen)
        self.assertEqual(p1.aember, 0)
        self.assertEqual(p2.aember, 0)

    def test_truebaru_costs_3_aember_to_play_and_pays_5_on_death(self):
        game = new_game()
        p1 = game.players[1]
        p1.aember = 5
        card = hand_card(game, 1, "Truebaru")
        self.assertIsNone(game.why_not_playable(1, card))
        gen = game._play_card(1, card)
        drive(gen, answers=["left"])
        self.assertEqual(p1.aember, 2)  # paid 3 to play
        self.assertIn(card, p1.play_area.creatures)
        drive(game.destroy_cards([card]))
        self.assertEqual(p1.aember, 7)  # 2 + 5 on death

    def test_truebaru_destroyed_pays_the_controller_not_the_owner(self):
        # User decision: an unqualified "Destroyed:" ability pays whoever
        # controls the creature at the moment of destruction, not its
        # original owner, when a control-change effect moved it first.
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        truebaru = put_creature(game, 1, "Truebaru")  # owned by p1
        collar = hand_card(game, 2, "Collar of Subordination")
        drive(game._play_card(2, collar), answers=[[truebaru]])
        self.assertEqual(truebaru.controller, 2)
        drive(game.destroy_cards([truebaru]))
        self.assertEqual(p2.aember, 5)  # controller at destruction
        self.assertEqual(p1.aember, 0)  # not the owner

    def test_truebaru_cannot_be_played_without_3_aember(self):
        game = new_game()
        p1 = game.players[1]
        p1.aember = 2
        card = hand_card(game, 1, "Truebaru")
        self.assertIsNotNone(game.why_not_playable(1, card))

    def test_collar_of_subordination_takes_control_of_the_host(self):
        # Exercises the real `_play_card` path (not a direct register_passive
        # call) -- that path sets `card.controller` before register_passive
        # runs, which is what made this a no-op before the fix.
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        host = put_creature(game, 2, "Charette")
        collar = hand_card(game, 1, "Collar of Subordination")
        drive(game._play_card(1, collar), answers=[[host]])
        self.assertIn(host, p1.play_area.creatures)
        self.assertNotIn(host, p2.play_area.creatures)
        self.assertEqual(host.controller, 1)

    def test_flame_wreathed_grants_power_and_hazardous(self):
        game = new_game()
        host = put_creature(game, 1, "Charette")  # power 4
        upgrade = make_card("Flame-Wreathed", 1)
        upgrade.type_object.host = host
        host.type_object.upgrades.append(upgrade)
        self.assertEqual(game.get_power(host), 6)
        self.assertEqual(game.get_hazardous(host), 2)

    def test_tentacus_charges_opponent_to_use_an_artifact(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        put_creature(game, 1, "Tentacus")  # put_creature already calls register_passive
        p2.aember = 5
        art = put_artifact(game, 2, "Library of Babble", exhausted=False)
        run_hook(game, art.card_def.on_action, art)
        # This direct hook call bypasses the toll (that's charged in _use_action);
        # verify the toll machinery itself instead.
        toll = p2.get_artifact_use_toll(game)
        self.assertEqual(toll, (1, 1))


if __name__ == "__main__":
    unittest.main()

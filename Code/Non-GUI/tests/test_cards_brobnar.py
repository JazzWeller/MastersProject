"""Tests for the 52 Brobnar cards (Code/PHASE_3_PLAN.md Milestone D.1)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.helpers import drive, hand_card, make_card, new_game, put_artifact, put_creature, run_hook

from keyforge.effects import steps
from keyforge.effects.named import brobnar as named
from keyforge.enums import House


class TestBrobnarActions(unittest.TestCase):
    def test_anger_readies_and_fights_with_a_chosen_creature(self):
        game = new_game()
        attacker = put_creature(game, 1, "Charette", exhausted=True)
        put_creature(game, 2, "Drumble")
        card = make_card("Anger", 1)
        run_hook(game, named.anger, card, answers=[[attacker]])
        self.assertTrue(attacker.Exhausted)  # it fought, so it's exhausted again

    def test_anger_on_a_creature_with_no_targets_leaves_it_ready(self):
        game = new_game()
        attacker = put_creature(game, 1, "Charette", exhausted=True)
        card = make_card("Anger", 1)
        run_hook(game, named.anger, card, answers=[[attacker]])
        self.assertFalse(attacker.Exhausted)  # no enemy creatures: readied, never fought

    def test_anger_removes_stun_instead_of_fighting(self):
        game = new_game()
        attacker = put_creature(game, 1, "Charette", exhausted=True)
        attacker.stunned = True
        put_creature(game, 2, "Drumble")
        card = make_card("Anger", 1)
        run_hook(game, named.anger, card, answers=[[attacker]])
        self.assertFalse(attacker.stunned)
        self.assertTrue(attacker.Exhausted)  # consuming a stun still exhausts, no fight happens

    def test_barehanded_returns_every_artifact_to_its_owners_deck_top(self):
        game = new_game()
        a1 = put_artifact(game, 1, "Cannon")
        a2 = put_artifact(game, 2, "Screechbomb")
        card = make_card("Barehanded", 1)
        run_hook(game, named.barehanded, card)
        self.assertIs(game.players[1].deck.cards()[0], a1)
        self.assertIs(game.players[2].deck.cards()[0], a2)
        self.assertNotIn(a1, game.players[1].play_area.artifacts)

    def test_blood_money_places_aember_that_releases_to_caster_on_death(self):
        game = new_game()
        target = put_creature(game, 2, "Drumble")
        card = make_card("Blood Money", 1)
        run_hook(game, named.blood_money, card, answers=[[target]])
        self.assertEqual(target.aember_captured, 2)
        drive(game.destroy_cards([target]))
        self.assertEqual(game.players[1].aember, 2)  # caster (target's controller's opponent) gets it

    def test_brothers_in_battle_lets_off_house_creatures_of_that_house_fight(self):
        game = new_game()
        mine = put_creature(game, 1, "Charette", can_be_used=False)  # not the active house
        put_creature(game, 2, "Drumble")
        card = make_card("Brothers in Battle", 1)
        run_hook(game, named.brothers_in_battle, card, answers=[House.DIS])
        self.assertTrue(game._can_fight_off_house(mine))
        actions = game._legal_actions(1)
        from keyforge.actions import Fight
        self.assertTrue(any(isinstance(a, Fight) and a.card is mine for a in actions))

    def test_burn_the_stockpile_only_fires_at_7_or_more(self):
        game = new_game()
        game.players[2].aember = 6
        card = make_card("Burn the Stockpile", 1)
        run_hook(game, named.burn_the_stockpile, card)
        self.assertEqual(game.players[2].aember, 6)
        game.players[2].aember = 7
        run_hook(game, named.burn_the_stockpile, card)
        self.assertEqual(game.players[2].aember, 3)

    def test_champions_challenge_leaves_one_creature_per_side_and_fights(self):
        game = new_game()
        weak_mine = put_creature(game, 1, "Charette")  # power 4
        strong_mine = put_creature(game, 1, "Truebaru")  # power 7
        weak_enemy = put_creature(game, 2, "Charette")  # power 4
        strong_enemy = put_creature(game, 2, "Bumpsy")  # power 5, weaker than strong_mine
        card = make_card("Champion’s Challenge", 1)
        run_hook(game, named.champions_challenge, card)
        self.assertNotIn(weak_mine, game.players[1].play_area.creatures)
        self.assertNotIn(weak_enemy, game.players[2].play_area.creatures)
        self.assertIn(strong_mine, game.players[1].play_area.creatures)
        self.assertTrue(strong_mine.Exhausted)  # readied and fought
        self.assertNotIn(strong_enemy, game.players[2].play_area.creatures)  # outmatched, destroyed

    def test_cowards_end_destroys_only_undamaged_creatures_and_gains_chains(self):
        game = new_game()
        undamaged = put_creature(game, 2, "Drumble")
        damaged = put_creature(game, 1, "Truebaru")
        damaged.type_object.damage = 1
        card = make_card("Coward’s End", 1)
        run_hook(game, named.cowards_end, card)
        self.assertNotIn(undamaged, game.players[2].play_area.creatures)
        self.assertIn(damaged, game.players[1].play_area.creatures)
        self.assertEqual(game.players[1].chains, 3)

    def test_follow_the_leader_lets_any_friendly_creature_fight(self):
        game = new_game()
        mine = put_creature(game, 1, "Drumble", can_be_used=False)
        put_creature(game, 2, "Charette")
        card = make_card("Follow the Leader", 1)
        run_hook(game, named.follow_the_leader, card)
        self.assertTrue(game._can_fight_off_house(mine))

    def test_loot_the_bodies_gains_on_enemy_destroyed_this_turn_only(self):
        game = new_game()
        victim_enemy = put_creature(game, 2, "Drumble")
        victim_mine = put_creature(game, 1, "Drumble")
        card = make_card("Loot the Bodies", 1)
        run_hook(game, named.loot_the_bodies, card)
        drive(game.destroy_cards([victim_enemy]))
        self.assertEqual(game.players[1].aember, 1)
        drive(game.destroy_cards([victim_mine]))
        self.assertEqual(game.players[1].aember, 1)  # friendly deaths don't pay out
        game.active_effects.end_of_turn_tick()
        drive(game.destroy_cards([put_creature(game, 2, "Drumble")]))
        self.assertEqual(game.players[1].aember, 1)  # expired after the turn ends

    def test_take_that_smartypants_needs_three_logos_cards_in_play(self):
        game = new_game()
        put_creature(game, 2, "Doc Bookton")
        put_creature(game, 2, "Mother")
        put_artifact(game, 2, "Library of the Damned")  # Dis, doesn't count
        game.players[2].aember = 5
        card = make_card("Take that, Smartypants", 1)
        run_hook(game, named.take_that_smartypants, card)
        self.assertEqual(game.players[1].aember, 0)
        put_creature(game, 2, "Timetraveller")  # Logos, makes 3
        run_hook(game, named.take_that_smartypants, card)
        self.assertEqual(game.players[1].aember, 2)
        self.assertEqual(game.players[2].aember, 3)

    def test_relentless_assault_fights_up_to_three_creatures_one_at_a_time(self):
        game = new_game()
        c1 = put_creature(game, 1, "Drumble", exhausted=True)
        c2 = put_creature(game, 1, "Charette", exhausted=True)
        put_creature(game, 2, "Truebaru")
        card = make_card("Relentless Assault", 1)
        # Choose c1, continue=True, choose c2, continue=True (but none left -> stop).
        run_hook(game, named.relentless_assault, card, answers=[[c1], True, [c2]])
        self.assertTrue(c1.Exhausted)
        self.assertTrue(c2.Exhausted)

    def test_relentless_assault_stops_early_if_declined(self):
        game = new_game()
        c1 = put_creature(game, 1, "Drumble", exhausted=True)
        c2 = put_creature(game, 1, "Charette", exhausted=True)
        put_creature(game, 2, "Truebaru")
        card = make_card("Relentless Assault", 1)
        run_hook(game, named.relentless_assault, card, answers=[[c1], False])
        self.assertTrue(c1.Exhausted)
        self.assertTrue(c2.Exhausted)  # declined -- untouched, still in its starting state

    def test_smith_gains_only_with_more_creatures(self):
        game = new_game()
        put_creature(game, 1, "Drumble")
        put_creature(game, 2, "Charette")
        card = make_card("Smith", 1)
        run_hook(game, named.smith, card)
        self.assertEqual(game.players[1].aember, 0)
        put_creature(game, 1, "Truebaru")
        run_hook(game, named.smith, card)
        self.assertEqual(game.players[1].aember, 2)

    def test_sound_the_horns_finds_and_returns_a_brobnar_creature(self):
        game = new_game()
        p1 = game.players[1]
        p1.deck._cards.clear()
        logos_card = make_card("Doc Bookton", 1)
        brobnar_card = make_card("Anger", 1)  # Brobnar action -- not a creature, keep digging
        brobnar_creature = make_card("Bumpsy", 1)
        # put_on_top always inserts at the front, so insert in reverse of the
        # intended draw order: logos_card first (drawn), then brobnar_card,
        # then the Brobnar creature the search should find and stop on.
        for c in (brobnar_creature, brobnar_card, logos_card):
            p1.deck.put_on_top(c)
        card = make_card("Sound the Horns", 1)
        run_hook(game, named.sound_the_horns, card)
        self.assertIn(brobnar_creature, p1.hand.cards())
        self.assertIn(logos_card, p1.discard.cards())
        self.assertIn(brobnar_card, p1.discard.cards())

    def test_tremor_stuns_the_target_and_its_neighbors(self):
        game = new_game()
        left = put_creature(game, 2, "Drumble")
        mid = put_creature(game, 2, "Charette")
        right = put_creature(game, 2, "Truebaru")
        card = make_card("Tremor", 1)
        run_hook(game, named.tremor, card, answers=[[mid]])
        self.assertTrue(left.stunned)
        self.assertTrue(mid.stunned)
        self.assertTrue(right.stunned)

    def test_unguarded_camp_captures_once_per_excess_creature(self):
        game = new_game()
        game.players[2].aember = 5
        c1 = put_creature(game, 1, "Drumble")
        c2 = put_creature(game, 1, "Charette")
        put_creature(game, 1, "Truebaru")
        put_creature(game, 2, "Bumpsy")  # excess = 3 - 1 = 2
        card = make_card("Unguarded Camp", 1)
        run_hook(game, named.unguarded_camp, card, answers=[[c1], [c2]])
        self.assertEqual(c1.aember_captured, 1)
        self.assertEqual(c2.aember_captured, 1)
        self.assertEqual(game.players[2].aember, 3)

    def test_warsong_gains_on_friendly_fights_only_this_turn(self):
        game = new_game()
        mine = put_creature(game, 1, "Truebaru")
        put_creature(game, 2, "Drumble")
        card = make_card("Warsong", 1)
        run_hook(game, named.warsong, card)
        drive(game._fight(1, mine))
        self.assertEqual(game.players[1].aember, 1)


class TestBrobnarArtifacts(unittest.TestCase):
    def test_autocannon_damages_any_creature_that_enters_play(self):
        game = new_game()
        put_artifact(game, 1, "Autocannon")
        entering = make_card("Drumble", 2)
        game.players[2].play_area.add_creature(entering)
        drive(game._fire_event("creature_entered_play", {"card": entering}))
        self.assertEqual(entering.type_object.damage, 1)

    def test_banner_of_battle_boosts_only_friendly_power(self):
        game = new_game()
        put_artifact(game, 1, "Banner of Battle")
        mine = put_creature(game, 1, "Drumble")
        theirs = put_creature(game, 2, "Drumble")
        self.assertEqual(game.get_power(mine), 3)
        self.assertEqual(game.get_power(theirs), 2)

    def test_gauntlet_of_command_readies_and_fights(self):
        game = new_game()
        put_artifact(game, 1, "Gauntlet of Command")
        attacker = put_creature(game, 1, "Charette", exhausted=True)
        put_creature(game, 2, "Drumble")
        card = make_card("Gauntlet of Command", 1)
        run_hook(game, named.gauntlet_of_command, card, answers=[[attacker]])
        self.assertTrue(attacker.Exhausted)

    def test_iron_obelisk_raises_opponents_key_cost_per_damaged_brobnar_creature(self):
        game = new_game()
        put_artifact(game, 1, "Iron Obelisk")  # put_artifact already registers its passive
        damaged = put_creature(game, 1, "Bumpsy")  # Brobnar
        damaged.type_object.damage = 1
        self.assertEqual(game.players[2].get_key_forge_cost(game), 7)
        healthy = put_creature(game, 1, "Wardrummer")  # Brobnar, undamaged
        self.assertEqual(game.players[2].get_key_forge_cost(game), 7)
        healthy.type_object.damage = 1
        self.assertEqual(game.players[2].get_key_forge_cost(game), 8)
        self.assertEqual(game.players[1].get_key_forge_cost(game), 6)  # unaffected

    def test_mighty_javelin_sacrifices_itself_and_deals_damage(self):
        game = new_game()
        javelin = put_artifact(game, 1, "Mighty Javelin")
        target = put_creature(game, 2, "Truebaru")
        run_hook(game, named.mighty_javelin, javelin, answers=[[target]])
        self.assertNotIn(javelin, game.players[1].play_area.artifacts)
        self.assertEqual(target.type_object.damage, 4)

    def test_pile_of_skulls_captures_only_for_enemy_destroyed_on_your_turn(self):
        game = new_game()
        put_artifact(game, 1, "Pile of Skulls")  # put_artifact already registers its passive
        friendly = put_creature(game, 1, "Charette")
        game.players[2].aember = 3
        victim = put_creature(game, 2, "Drumble")
        drive(game.destroy_cards([victim]), answers=[[friendly]])
        self.assertEqual(friendly.aember_captured, 1)

    def test_screechbomb_sacrifices_and_opponent_loses_aember(self):
        game = new_game()
        bomb = put_artifact(game, 1, "Screechbomb")
        game.players[2].aember = 3
        run_hook(game, named.screechbomb, bomb)
        self.assertNotIn(bomb, game.players[1].play_area.artifacts)
        self.assertEqual(game.players[2].aember, 1)

    def test_the_warchest_counts_only_fight_destructions_this_turn(self):
        game = new_game()
        warchest = put_artifact(game, 1, "The Warchest")
        attacker = put_creature(game, 1, "Truebaru")
        weak_enemy = put_creature(game, 2, "Charette")  # not elusive -- actually dies in the fight
        drive(game._fight(1, attacker))  # destroys weak_enemy in a real fight
        self.assertNotIn(weak_enemy, game.players[2].play_area.creatures)
        run_hook(game, named.the_warchest, warchest)
        self.assertEqual(game.players[1].aember, 1)


class TestBrobnarCreatures(unittest.TestCase):
    def test_bilgum_avalanche_damages_enemies_after_its_controller_forges_a_key(self):
        game = new_game()
        put_creature(game, 1, "Bilgum Avalanche")  # put_creature already registers its passive
        enemy = put_creature(game, 2, "Truebaru")  # power 7, survives 2 damage
        drive(game._fire_event("key_forged", {"player": 1}))
        self.assertEqual(enemy.type_object.damage, 2)
        drive(game._fire_event("key_forged", {"player": 2}))
        self.assertEqual(enemy.type_object.damage, 2)  # only fires for its own controller's key

    def test_valdr_deals_bonus_damage_only_against_a_flank_target(self):
        game = new_game()
        valdr = put_creature(game, 1, "Valdr")  # put_creature already registers its passive
        flank_target = put_creature(game, 2, "Kelifi Dragon")  # power 12, survives everything here
        drive(game._fight(1, valdr))
        self.assertEqual(flank_target.type_object.damage, 8)  # 6 power + 2 flank bonus
        flank_target.type_object.damage = 0
        put_creature(game, 2, "Charette", flank="left")
        put_creature(game, 2, "Charette", flank="right")
        valdr.Exhausted = False
        drive(game._fight(1, valdr), answers=[[flank_target]])
        self.assertEqual(flank_target.type_object.damage, 6)  # no flank bonus once it's in the center

    def test_bumpsy_makes_opponent_lose_aember(self):
        game = new_game()
        game.players[2].aember = 3
        card = make_card("Bumpsy", 1)
        run_hook(game, __import__("keyforge.effects.generic", fromlist=["lose_n"]).lose_n(1), card)
        self.assertEqual(game.players[2].aember, 2)

    def test_earthshaker_destroys_low_power_creatures_only(self):
        game = new_game()
        weak = put_creature(game, 2, "Drumble")  # power 2
        strong = put_creature(game, 2, "Truebaru")  # power 7
        card = make_card("Earthshaker", 1)
        run_hook(game, named.earthshaker, card)
        self.assertNotIn(weak, game.players[2].play_area.creatures)
        self.assertIn(strong, game.players[2].play_area.creatures)

    def test_firespitter_before_fight_damages_each_enemy_creature(self):
        game = new_game()
        firespitter = put_creature(game, 1, "Firespitter")
        target = put_creature(game, 2, "Truebaru")
        other_enemy = put_creature(game, 2, "Drumble")
        drive(game._fight(1, firespitter))
        self.assertEqual(other_enemy.type_object.damage, 1)

    def test_ganger_chieftain_may_ready_and_fight_a_neighbor(self):
        game = new_game()
        neighbor = put_creature(game, 1, "Drumble", exhausted=True)
        chieftain = put_creature(game, 1, "Ganger Chieftain", flank="right")
        put_creature(game, 2, "Truebaru")
        run_hook(game, named.ganger_chieftain_play, chieftain, answers=[True, [neighbor]])
        self.assertTrue(neighbor.Exhausted)

    def test_grenade_snib_costs_opponent_aember_when_destroyed(self):
        game = new_game()
        game.players[2].aember = 3
        snib = put_creature(game, 1, "Grenade Snib")
        drive(game.destroy_cards([snib]))
        self.assertEqual(game.players[2].aember, 1)

    def test_headhunter_gains_aember_after_fight(self):
        game = new_game()
        headhunter = put_creature(game, 1, "Headhunter")
        put_creature(game, 2, "Drumble")
        drive(game._fight(1, headhunter))
        self.assertEqual(game.players[1].aember, 1)

    def test_hebe_the_huge_damages_other_undamaged_creatures_only(self):
        game = new_game()
        hebe = put_creature(game, 1, "Hebe the Huge")
        already_hurt = put_creature(game, 2, "Truebaru")  # power 7
        already_hurt.type_object.damage = 1
        fresh = put_creature(game, 2, "Kelifi Dragon")  # power 12, survives the 2 damage
        run_hook(game, named.hebe_the_huge, hebe)
        self.assertEqual(already_hurt.type_object.damage, 1)  # untouched: it already had damage
        self.assertEqual(fresh.type_object.damage, 2)

    def test_kelifi_dragon_needs_seven_aember_to_play_but_doesnt_spend_it(self):
        game = new_game()
        game.players[1].selected_house = House.BROBNAR
        cdef = __import__("keyforge.cards.card_data", fromlist=["get_card_def"]).get_card_def("Kelifi Dragon")
        self.assertEqual(cdef.min_aember_to_play, 7)
        game.players[1].aember = 8
        card = hand_card(game, 1, "Kelifi Dragon")
        self.assertIsNone(game.why_not_playable(1, card))
        self.assertEqual(game.players[1].aember, 8)  # the threshold is checked, not spent
        game.players[1].aember = 3
        self.assertIsNotNone(game.why_not_playable(1, card))

    def test_kelifi_dragon_after_fight_or_reap_gains_and_deals_damage(self):
        game = new_game()
        dragon = put_creature(game, 1, "Kelifi Dragon")
        target = put_creature(game, 2, "Truebaru")
        run_hook(game, named.kelifi_dragon_after, dragon, answers=[[target]])
        self.assertEqual(game.players[1].aember, 1)
        self.assertEqual(target.type_object.damage, 5)

    def test_king_of_the_crag_weakens_only_enemy_brobnar_creatures(self):
        game = new_game()
        put_creature(game, 1, "King of the Crag")  # put_creature already registers its passive
        enemy_brobnar = put_creature(game, 2, "Bumpsy")  # power 5
        enemy_other = put_creature(game, 2, "Drumble")  # Shadows, power 2
        self.assertEqual(game.get_power(enemy_brobnar), 3)
        self.assertEqual(game.get_power(enemy_other), 2)

    def test_krump_makes_the_loser_lose_aember_when_destroyed_fighting_it(self):
        game = new_game()
        game.players[2].aember = 3
        krump = put_creature(game, 1, "Krump")  # power 6
        weak = put_creature(game, 2, "Charette")  # power 4, not elusive -- dies to Krump
        drive(game._fight(1, krump))
        self.assertNotIn(weak, game.players[2].play_area.creatures)
        self.assertEqual(game.players[2].aember, 2)

    def test_lomir_flamefist_only_hits_at_seven_or_more(self):
        game = new_game()
        game.players[2].aember = 6
        card = make_card("Lomir Flamefist", 1)
        run_hook(game, named.lomir_flamefist, card)
        self.assertEqual(game.players[2].aember, 6)
        game.players[2].aember = 7
        run_hook(game, named.lomir_flamefist, card)
        self.assertEqual(game.players[2].aember, 5)

    def test_looter_goblin_grants_gain_on_enemy_destroyed_this_turn(self):
        game = new_game()
        goblin = put_creature(game, 1, "Looter Goblin")
        run_hook(game, named.loot_the_bodies, goblin)  # same helper Looter Goblin's reap uses
        victim = put_creature(game, 2, "Drumble")
        drive(game.destroy_cards([victim]))
        self.assertEqual(game.players[1].aember, 1)

    def test_mugwump_heals_and_grows_when_it_wins_a_fight(self):
        game = new_game()
        mugwump = put_creature(game, 1, "Mugwump")  # power 6
        mugwump.type_object.damage = 1  # prior damage, unrelated to this fight
        weak = put_creature(game, 2, "Charette")  # power 4, not elusive -- dies, deals 4 back (1+4=5 < 6: Mugwump survives)
        drive(game._fight(1, mugwump))
        self.assertNotIn(weak, game.players[2].play_area.creatures)
        self.assertEqual(mugwump.type_object.damage, 0)
        self.assertEqual(mugwump.power_counters, 1)

    def test_pingle_who_annoys_damages_enemy_entries_only(self):
        game = new_game()
        put_creature(game, 1, "Pingle Who Annoys")  # put_creature already registers its passive
        enemy_entrant = make_card("Drumble", 2)
        game.players[2].play_area.add_creature(enemy_entrant)
        drive(game._fire_event("creature_entered_play", {"card": enemy_entrant}))
        self.assertEqual(enemy_entrant.type_object.damage, 1)
        friendly_entrant = make_card("Charette", 1)
        game.players[1].play_area.add_creature(friendly_entrant)
        drive(game._fire_event("creature_entered_play", {"card": friendly_entrant}))
        self.assertEqual(friendly_entrant.type_object.damage, 0)

    def test_rock_hurling_giant_offers_damage_on_brobnar_discard_during_your_turn(self):
        game = new_game()
        put_creature(game, 1, "Rock-Hurling Giant")  # put_creature already registers its passive
        target = put_creature(game, 2, "Truebaru")
        p1 = game.players[1]
        brobnar_card = hand_card(game, 1, "Anger")
        drive(steps.discard_from_hand(game, p1, brobnar_card), answers=[True, [target]])
        self.assertEqual(target.type_object.damage, 4)

    def test_rogue_ogre_heals_and_captures_if_exactly_one_card_played(self):
        game = new_game()
        ogre = put_creature(game, 1, "Rogue Ogre")  # put_creature already registers its passive
        ogre.type_object.damage = 3
        game.players[2].aember = 2
        game.players[1].CardsPlayed = {"Something": 1}
        drive(game._fire_event("end_of_turn", {"player": 1}))
        self.assertEqual(ogre.type_object.damage, 1)
        self.assertEqual(ogre.aember_captured, 1)

    def test_smaaash_stuns_a_chosen_creature(self):
        game = new_game()
        smaaash = put_creature(game, 1, "Smaaash")
        target = put_creature(game, 2, "Drumble")
        run_hook(game, named.smaaash, smaaash, answers=[[target]])
        self.assertTrue(target.stunned)

    def test_tireless_crocag_cannot_reap_but_can_fight_off_house(self):
        game = new_game()
        crocag = put_creature(game, 1, "Tireless Crocag", can_be_used=False)
        from keyforge.actions import Reap
        put_creature(game, 2, "Drumble")
        actions = game._legal_actions(1)
        self.assertFalse(any(isinstance(a, Reap) and a.card is crocag for a in actions))

    def test_tireless_crocag_destroyed_if_opponent_has_no_creatures(self):
        game = new_game()
        crocag = put_creature(game, 1, "Tireless Crocag")
        drive(named.tireless_crocag_play(game, crocag))
        self.assertNotIn(crocag, game.players[1].play_area.creatures)

    def test_tireless_crocag_survives_if_opponent_has_a_creature(self):
        game = new_game()
        crocag = put_creature(game, 1, "Tireless Crocag")
        put_creature(game, 2, "Drumble")
        drive(named.tireless_crocag_play(game, crocag))
        self.assertIn(crocag, game.players[1].play_area.creatures)

    def test_troll_heals_after_reap(self):
        game = new_game()
        from keyforge.effects import generic
        troll = put_creature(game, 1, "Troll")
        troll.type_object.damage = 5
        run_hook(game, generic.heal_self_n(3), troll)
        self.assertEqual(troll.type_object.damage, 2)

    def test_wardrummer_returns_other_friendly_brobnar_creatures_only(self):
        game = new_game()
        drummer = put_creature(game, 1, "Wardrummer")
        other_brobnar = put_creature(game, 1, "Bumpsy")
        other_house = put_creature(game, 1, "Charette")
        run_hook(game, named.wardrummer, drummer)
        self.assertIn(other_brobnar, game.players[1].hand.cards())
        self.assertIn(other_house, game.players[1].play_area.creatures)
        self.assertIn(drummer, game.players[1].play_area.creatures)

    def test_wardrummer_returns_a_borrowed_creature_to_its_owners_hand(self):
        # MRB 18.3 "Movement between zones" + the Faygin FAQ: a card leaving
        # play goes to its owner's hand even when the text says "your hand".
        game = new_game()
        drummer = put_creature(game, 1, "Wardrummer")
        stolen = make_card("Bumpsy", owner=2, controller=1)  # Brobnar, owned by 2 but under 1's control
        game.players[1].play_area.add_creature(stolen)
        game._cards_by_id[stolen.instance_id] = stolen
        run_hook(game, named.wardrummer, drummer)
        self.assertIn(stolen, game.players[2].hand.cards())
        self.assertNotIn(stolen, game.players[1].hand.cards())


class TestBrobnarUpgrades(unittest.TestCase):
    def test_blood_of_titans_grants_plus_five_power(self):
        game = new_game()
        host = put_creature(game, 1, "Drumble")
        upgrade = make_card("Blood of Titans", 1)
        upgrade.type_object.host = host
        host.type_object.upgrades.append(upgrade)
        self.assertEqual(game.get_power(host), 7)

    def test_phoenix_heart_returns_host_and_deals_damage_when_destroyed(self):
        game = new_game()
        host = put_creature(game, 1, "Charette")
        upgrade = make_card("Phoenix Heart", 1)
        upgrade.type_object.host = host
        host.type_object.upgrades.append(upgrade)
        named.phoenix_heart_register(game, upgrade)
        bystander = put_creature(game, 2, "Truebaru")
        drive(game.destroy_cards([host]))
        self.assertIn(host, game.players[1].hand.cards())
        self.assertEqual(bystander.type_object.damage, 3)
        self.assertEqual(host.type_object.damage, 0)  # never took its own blast

    def test_yo_mama_mastery_grants_taunt_and_fully_heals_on_play(self):
        game = new_game()
        host = put_creature(game, 1, "Truebaru")
        host.type_object.damage = 4
        upgrade = make_card("Yo Mama Mastery", 1)
        upgrade.type_object.host = host
        host.type_object.upgrades.append(upgrade)
        self.assertIn("taunt", game.get_keywords(host))
        run_hook(game, named.yo_mama_mastery_play, upgrade)
        self.assertEqual(host.type_object.damage, 0)


if __name__ == "__main__":
    unittest.main()

"""Tests for the 52 Mars cards (Code/PHASE_3_PLAN.md Milestone D.3)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.helpers import drive, hand_card, make_card, new_game, put_artifact, put_creature, run_hook

from keyforge.effects import steps
from keyforge.effects.named import mars as named
from keyforge.enums import House


class TestMarsActions(unittest.TestCase):
    def test_ammonia_clouds_hits_every_creature(self):
        game = new_game()
        mine = put_creature(game, 1, "Drumble")
        theirs = put_creature(game, 2, "Kelifi Dragon")
        card = make_card("Ammonia Clouds", 1)
        run_hook(game, named.ammonia_clouds, card)
        self.assertNotIn(mine, game.players[1].play_area.creatures)  # power 2, destroyed
        self.assertEqual(theirs.type_object.damage, 3)

    def test_battle_fleet_draws_per_revealed_mars_card(self):
        game = new_game()
        p1 = game.players[1]
        hand_card(game, 1, "Ammonia Clouds")  # Mars
        hand_card(game, 1, "Squawker")  # Mars
        hand_card(game, 1, "Charette")  # Dis, won't be offered
        for _ in range(5):
            p1.deck.put_on_top(make_card("Drumble", 1))
        card = make_card("Battle Fleet", 1)
        before = len(p1.hand)
        run_hook(game, named.battle_fleet, card, answers=[[c for c in p1.hand.cards() if c.house == House.MARS]])
        self.assertEqual(len(p1.hand), before + 2)  # revealing doesn't remove cards; drew 2 for the 2 revealed

    def test_deep_probe_discards_matching_house_creatures_from_opponent_hand(self):
        game = new_game()
        p2 = game.players[2]
        target = hand_card(game, 2, "Charette")  # Dis creature
        other_house = hand_card(game, 2, "Doc Bookton")  # Logos creature
        card = make_card("Deep Probe", 1)
        run_hook(game, named.deep_probe, card, answers=[House.DIS])
        self.assertIn(target, p2.discard.cards())
        self.assertIn(other_house, p2.hand.cards())

    def test_emp_blast_stuns_mars_and_robots_and_destroys_artifacts(self):
        game = new_game()
        mars_creature = put_creature(game, 1, "Grabber Jammer")  # Mars, robot
        other_creature = put_creature(game, 2, "Charette")
        artifact = put_artifact(game, 2, "Hallowed Blaster")
        card = make_card("EMP Blast", 1)
        run_hook(game, named.emp_blast, card)
        self.assertTrue(mars_creature.stunned)
        self.assertFalse(other_creature.stunned)
        self.assertNotIn(artifact, game.players[2].play_area.artifacts)

    def test_emp_blast_destroys_artifacts_through_the_shared_pipeline(self):
        # Must go through game.destroy_cards (not an ad hoc removal) so any
        # captured Æmber on the artifact properly releases via the standard
        # leave-play bookkeeping (and any future Destroyed: hook would fire).
        game = new_game()
        artifact = put_artifact(game, 2, "Hallowed Blaster")
        artifact.aember_captured = 3
        card = make_card("EMP Blast", 1)
        run_hook(game, named.emp_blast, card)
        self.assertIn(artifact, game.players[2].discard.cards())
        self.assertEqual(game.players[1].aember, 3)  # captured Æmber released to the opponent

    def test_hypnotic_command_captures_from_targets_own_side(self):
        game = new_game()
        put_creature(game, 1, "Yxili Marauder")  # friendly Mars
        game.players[2].aember = 5
        target = put_creature(game, 2, "Charette")
        card = make_card("Hypnotic Command", 1)
        run_hook(game, named.hypnotic_command, card, answers=[[target]])
        self.assertEqual(target.aember_captured, 1)
        self.assertEqual(game.players[2].aember, 4)

    def test_irradiated_aember_only_at_six_or_more(self):
        game = new_game()
        game.players[2].aember = 5
        enemy = put_creature(game, 2, "Kelifi Dragon")
        card = make_card("Irradiated Æmber", 1)
        run_hook(game, named.irradiated_aember, card)
        self.assertEqual(enemy.type_object.damage, 0)
        game.players[2].aember = 6
        run_hook(game, named.irradiated_aember, card)
        self.assertEqual(enemy.type_object.damage, 3)

    def test_key_abduction_returns_mars_creatures_and_may_forge(self):
        game = new_game()
        mine_mars = put_creature(game, 1, "Yxili Marauder")
        theirs_mars = put_creature(game, 2, "Grabber Jammer")
        game.players[1].aember = 20
        card = make_card("Key Abduction", 1)
        run_hook(game, named.key_abduction, card, answers=[True])
        self.assertIn(mine_mars, game.players[1].hand.cards())
        self.assertIn(theirs_mars, game.players[2].hand.cards())
        self.assertEqual(game.players[1].keys, 1)

    def test_key_abduction_reduction_can_go_below_the_base_cost(self):
        # Base key cost is 6; +9 reduced by 1 per card in hand means a hand
        # of 11+ cards should push the final cost BELOW 6, not just to 0
        # bonus -- the modifier itself may go negative.
        game = new_game()
        p1 = game.players[1]
        p1.hand.take_all()
        for _ in range(11):
            hand_card(game, 1, "Ammonia Clouds")
        p1.aember = 20
        card = make_card("Key Abduction", 1)
        run_hook(game, named.key_abduction, card, answers=[True])
        self.assertEqual(p1.keys, 1)
        self.assertEqual(p1.aember, 20 - 4)  # max(0, 6 + (9 - 11)) == 4

    def test_martian_hounds_grants_counters_per_damaged_creature(self):
        game = new_game()
        target = put_creature(game, 1, "Drumble")
        a = put_creature(game, 2, "Charette")
        a.type_object.damage = 1
        b = put_creature(game, 2, "Kelifi Dragon")
        b.type_object.damage = 1
        card = make_card("Martian Hounds", 1)
        run_hook(game, named.martian_hounds, card, answers=[[target]])
        self.assertEqual(target.power_counters, 4)

    def test_martians_make_bad_allies_purges_non_mars_creatures_in_hand(self):
        game = new_game()
        p1 = game.players[1]
        p1.hand.take_all()  # clear the starting hand so only the test's own cards are in play
        non_mars = hand_card(game, 1, "Charette")
        mars_creature = hand_card(game, 1, "Yxili Marauder")
        action_card = hand_card(game, 1, "Squawker")  # not a creature
        card = make_card("Martians Make Bad Allies", 1)
        run_hook(game, named.martians_make_bad_allies, card)
        self.assertIn(non_mars, p1.purged.cards())
        self.assertIn(mars_creature, p1.hand.cards())
        self.assertIn(action_card, p1.hand.cards())
        self.assertEqual(p1.aember, 1)

    def test_martians_make_bad_allies_reveals_own_hand_not_opponents(self):
        # "Reveal your hand" -- must reveal the caster's own hand to the
        # opponent, not the opponent's hand to the caster.
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        card = make_card("Martians Make Bad Allies", 1)
        run_hook(game, named.martians_make_bad_allies, card)
        self.assertIn(2, p1.hand_revealed_to)
        self.assertNotIn(1, p2.hand_revealed_to)

    def test_mass_abduction_archives_damaged_enemies_returning_to_owner(self):
        game = new_game()
        p1 = game.players[1]
        damaged = put_creature(game, 2, "Kelifi Dragon")
        damaged.type_object.damage = 1
        undamaged = put_creature(game, 2, "Charette")
        card = make_card("Mass Abduction", 1)
        run_hook(game, named.mass_abduction, card, answers=[[damaged]])
        self.assertIn(damaged, p1.archive.cards())
        self.assertIn(undamaged, game.players[2].play_area.creatures)
        self.assertTrue(damaged.archive_return_to_owner)
        p1.archive.remove(damaged)
        game.players[2].hand.add(damaged) if damaged.archive_return_to_owner else p1.hand.add(damaged)
        self.assertIn(damaged, game.players[2].hand.cards())

    def test_mating_season_shuffles_mars_creatures_and_pays_owners(self):
        game = new_game()
        mine = put_creature(game, 1, "Yxili Marauder")
        theirs = put_creature(game, 2, "Grabber Jammer")
        other_house = put_creature(game, 1, "Charette")
        card = make_card("Mating Season", 1)
        run_hook(game, named.mating_season, card)
        self.assertNotIn(mine, game.players[1].play_area.creatures)
        self.assertNotIn(theirs, game.players[2].play_area.creatures)
        self.assertIn(other_house, game.players[1].play_area.creatures)
        self.assertEqual(game.players[1].aember, 1)
        self.assertEqual(game.players[2].aember, 1)

    def test_mothership_support_deals_damage_per_ready_mars_creature(self):
        game = new_game()
        put_creature(game, 1, "Yxili Marauder")  # ready Mars
        put_creature(game, 1, "Grabber Jammer", exhausted=True)  # exhausted, doesn't count
        target = put_creature(game, 2, "Kelifi Dragon")
        card = make_card("Mothership Support", 1)
        run_hook(game, named.mothership_support, card, answers=[[target]])
        self.assertEqual(target.type_object.damage, 2)

    def test_orbital_bombardment_deals_damage_per_revealed(self):
        game = new_game()
        hand_card(game, 1, "Ammonia Clouds")
        hand_card(game, 1, "Squawker")
        target = put_creature(game, 2, "Kelifi Dragon")
        card = make_card("Orbital Bombardment", 1)
        marscards = [c for c in game.players[1].hand.cards() if c.house == House.MARS]
        run_hook(game, named.orbital_bombardment, card, answers=[marscards, [target], [target]])
        self.assertEqual(target.type_object.damage, 4)

    def test_phosphorus_stars_stuns_non_mars_and_gains_chains(self):
        game = new_game()
        mars_creature = put_creature(game, 2, "Grabber Jammer")
        other = put_creature(game, 2, "Charette")
        card = make_card("Phosphorus Stars", 1)
        run_hook(game, named.phosphorus_stars, card)
        self.assertFalse(mars_creature.stunned)
        self.assertTrue(other.stunned)
        self.assertEqual(game.players[1].chains, 2)

    def test_psychic_network_steals_per_ready_mars_creature(self):
        game = new_game()
        game.players[2].aember = 5
        put_creature(game, 1, "Yxili Marauder")
        put_creature(game, 1, "Grabber Jammer")
        card = make_card("Psychic Network", 1)
        run_hook(game, named.psychic_network, card)
        self.assertEqual(game.players[1].aember, 2)

    def test_sample_collection_archives_per_key_forged(self):
        game = new_game()
        game.players[2].keys = 2
        e1 = put_creature(game, 2, "Kelifi Dragon")
        e2 = put_creature(game, 2, "Charette")
        card = make_card("Sample Collection", 1)
        run_hook(game, named.sample_collection, card, answers=[[e1], [e2]])
        self.assertIn(e1, game.players[1].archive.cards())
        self.assertIn(e2, game.players[1].archive.cards())

    def test_shatter_storm_loses_all_and_opponent_loses_triple(self):
        game = new_game()
        game.players[1].aember = 3
        game.players[2].aember = 20
        card = make_card("Shatter Storm", 1)
        run_hook(game, named.shatter_storm, card)
        self.assertEqual(game.players[1].aember, 0)
        self.assertEqual(game.players[2].aember, 11)

    def test_soft_landing_next_entry_enters_ready(self):
        game = new_game()
        card = make_card("Soft Landing", 1)
        run_hook(game, named.soft_landing, card)
        self.assertTrue(game.players[1].next_entry_ready)
        new_creature = put_creature(game, 1, "Drumble", exhausted=True)
        # Simulate what _play_card does with the flag (unit-level check).
        self.assertTrue(game.players[1].next_entry_ready)

    def test_squawker_readies_mars_or_stuns_non_mars(self):
        game = new_game()
        mars_creature = put_creature(game, 1, "Yxili Marauder", exhausted=True)
        enemy = put_creature(game, 2, "Charette")
        card = make_card("Squawker", 1)
        run_hook(game, named.squawker, card, answers=["Ready a Mars creature", [mars_creature]])
        self.assertFalse(mars_creature.Exhausted)
        self.assertFalse(enemy.stunned)

    def test_total_recall_gains_per_ready_and_returns_all(self):
        game = new_game()
        ready = put_creature(game, 1, "Drumble")
        exhausted = put_creature(game, 1, "Kelifi Dragon", exhausted=True)
        card = make_card("Total Recall", 1)
        run_hook(game, named.total_recall, card)
        self.assertEqual(game.players[1].aember, 1)
        self.assertIn(ready, game.players[1].hand.cards())
        self.assertIn(exhausted, game.players[1].hand.cards())

    def test_total_recall_returns_a_borrowed_creature_to_its_owners_hand(self):
        # MRB 18.3 "Movement between zones" + the Faygin FAQ: a card leaving
        # play goes to its owner's hand even when the text says "your hand".
        game = new_game()
        stolen = make_card("Drumble", owner=2, controller=1)
        game.players[1].play_area.add_creature(stolen)
        game._cards_by_id[stolen.instance_id] = stolen
        card = make_card("Total Recall", 1)
        run_hook(game, named.total_recall, card)
        self.assertIn(stolen, game.players[2].hand.cards())
        self.assertNotIn(stolen, game.players[1].hand.cards())


class TestMarsArtifacts(unittest.TestCase):
    def test_combat_pheromones_lets_mars_creatures_be_used_off_house(self):
        game = new_game()
        pheromones = put_artifact(game, 1, "Combat Pheromones")
        mine = put_creature(game, 1, "Yxili Marauder", can_be_used=False)
        run_hook(game, named.combat_pheromones, pheromones)
        self.assertNotIn(pheromones, game.players[1].play_area.artifacts)
        self.assertTrue(mine.CanBeUsed)

    def test_commpod_readies_mars_creatures_per_revealed(self):
        game = new_game()
        commpod = put_artifact(game, 1, "Commpod")
        hand_card(game, 1, "Ammonia Clouds")
        exhausted = put_creature(game, 1, "Yxili Marauder", exhausted=True)
        marscards = [c for c in game.players[1].hand.cards() if c.house == House.MARS]
        run_hook(game, named.commpod, commpod, answers=[marscards, True, [exhausted]])
        self.assertFalse(exhausted.Exhausted)

    def test_crystal_hive_gains_on_reaps_this_turn(self):
        game = new_game()
        hive = put_artifact(game, 1, "Crystal Hive")
        reaper = put_creature(game, 1, "Drumble")
        run_hook(game, named.crystal_hive, hive)
        drive(game._reap(1, reaper))
        self.assertEqual(game.players[1].aember, 2)  # 1 from reap + 1 from Crystal Hive

    def test_custom_virus_destroys_shared_trait_creatures(self):
        game = new_game()
        virus = put_artifact(game, 1, "Custom Virus")
        purge_target = hand_card(game, 1, "Yxili Marauder")  # martian, soldier
        matching = put_creature(game, 2, "Yxilo Bolter")  # martian, soldier
        non_matching = put_creature(game, 2, "Charette")  # demon
        run_hook(game, named.custom_virus, virus, answers=[True, [purge_target]])
        self.assertNotIn(virus, game.players[1].play_area.artifacts)
        self.assertNotIn(matching, game.players[2].play_area.creatures)
        self.assertIn(non_matching, game.players[2].play_area.creatures)

    def test_feeding_pit_discards_and_gains(self):
        game = new_game()
        pit = put_artifact(game, 1, "Feeding Pit")
        creature = hand_card(game, 1, "Drumble")
        run_hook(game, named.feeding_pit, pit, answers=[[creature]])
        self.assertIn(creature, game.players[1].discard.cards())
        self.assertEqual(game.players[1].aember, 1)

    def test_invasion_portal_finds_a_mars_creature(self):
        game = new_game()
        p1 = game.players[1]
        p1.deck._cards.clear()
        mars_creature = make_card("Yxili Marauder", 1)
        other = make_card("Charette", 1)
        for c in (mars_creature, other):
            p1.deck.put_on_top(c)
        portal = put_artifact(game, 1, "Invasion Portal")
        run_hook(game, named.invasion_portal, portal)
        self.assertIn(mars_creature, p1.hand.cards())
        self.assertIn(other, p1.discard.cards())

    def test_incubation_chamber_archives_a_revealed_mars_creature(self):
        game = new_game()
        chamber = put_artifact(game, 1, "Incubation Chamber")
        creature = hand_card(game, 1, "Yxili Marauder")
        run_hook(game, named.incubation_chamber, chamber, answers=[True, [creature]])
        self.assertIn(creature, game.players[1].archive.cards())

    def test_mothergun_deals_damage_equal_to_revealed(self):
        game = new_game()
        gun = put_artifact(game, 1, "Mothergun")
        hand_card(game, 1, "Ammonia Clouds")
        hand_card(game, 1, "Squawker")
        target = put_creature(game, 2, "Kelifi Dragon")
        marscards = [c for c in game.players[1].hand.cards() if c.house == House.MARS]
        run_hook(game, named.mothergun, gun, answers=[marscards, [target]])
        self.assertEqual(target.type_object.damage, 2)

    def test_sniffer_removes_elusive_for_the_turn(self):
        game = new_game()
        sniffer = put_artifact(game, 1, "Sniffer")
        elusive_creature = put_creature(game, 2, "Drumble")
        self.assertIn("elusive", game.get_keywords(elusive_creature))
        run_hook(game, named.sniffer_action, sniffer)
        self.assertNotIn("elusive", game.get_keywords(elusive_creature))

    def test_swap_widget_swaps_a_ready_mars_creature(self):
        game = new_game()
        widget = put_artifact(game, 1, "Swap Widget")
        in_play = put_creature(game, 1, "Yxili Marauder")
        in_hand = hand_card(game, 1, "Grabber Jammer")
        run_hook(game, named.swap_widget, widget, answers=[[in_play], [in_hand]])
        self.assertIn(in_play, game.players[1].hand.cards())
        self.assertIn(in_hand, game.players[1].play_area.creatures)
        self.assertFalse(in_hand.Exhausted)

    def test_swap_widget_still_returns_with_no_replacement_in_hand(self):
        # "Return ... to your hand" is mandatory; "If you do, put ..." only
        # conditions the second half on the return happening, not on a
        # replacement being available.
        game = new_game()
        widget = put_artifact(game, 1, "Swap Widget")
        in_play = put_creature(game, 1, "Yxili Marauder")
        run_hook(game, named.swap_widget, widget, answers=[[in_play]])
        self.assertIn(in_play, game.players[1].hand.cards())
        self.assertNotIn(in_play, game.players[1].play_area.creatures)


class TestMarsCreatures(unittest.TestCase):
    def test_chuff_ape_enters_stunned_and_may_sacrifice_to_heal(self):
        game = new_game()
        ape = put_creature(game, 1, "Chuff Ape")
        named.chuff_ape_register(game, ape)
        self.assertTrue(ape.stunned)
        ape.type_object.damage = 5
        other = put_creature(game, 1, "Drumble")
        run_hook(game, named.chuff_ape_after, ape, answers=[True, [other]])
        self.assertNotIn(other, game.players[1].play_area.creatures)
        self.assertEqual(ape.type_object.damage, 0)

    def test_ether_spider_captures_aember_that_would_go_to_opponent(self):
        game = new_game()
        spider = put_creature(game, 1, "Ether Spider")
        named.ether_spider_register(game, spider)
        steps.gain(game, game.players[2], 3)
        self.assertEqual(game.players[2].aember, 0)
        self.assertEqual(spider.aember_captured, 3)
        steps.gain(game, game.players[1], 2)
        self.assertEqual(game.players[1].aember, 2)  # own player's gains are unaffected

    def test_ether_spider_deals_no_fight_damage(self):
        game = new_game()
        spider = put_creature(game, 1, "Ether Spider")
        target = put_creature(game, 2, "Charette")
        drive(game._fight(1, spider))
        self.assertEqual(target.type_object.damage, 0)

    def test_grabber_jammer_raises_key_cost_and_captures_after_fight(self):
        game = new_game()
        game.players[2].aember = 3
        jammer = put_creature(game, 1, "Grabber Jammer")
        put_creature(game, 2, "Charette")
        self.assertEqual(game.players[2].get_key_forge_cost(game), 7)
        drive(game._fight(1, jammer))
        self.assertEqual(jammer.aember_captured, 1)

    def test_grommid_stops_creature_plays_while_in_play(self):
        game = new_game()
        put_creature(game, 1, "Grommid")
        self.assertFalse(game.players[1].get_can_play_creatures(game))

    def test_john_smyth_readies_a_non_agent_mars_creature(self):
        game = new_game()
        smyth = put_creature(game, 1, "“John Smyth”")
        target = put_creature(game, 1, "Yxili Marauder", exhausted=True)
        run_hook(game, named.john_smyth_after, smyth, answers=[[target]])
        self.assertFalse(target.Exhausted)

    def test_mindwarper_captures_from_the_targets_own_controller(self):
        game = new_game()
        mindwarper = put_creature(game, 1, "Mindwarper")
        game.players[2].aember = 3
        target = put_creature(game, 2, "Charette")
        run_hook(game, named.mindwarper_action, mindwarper, answers=[[target]])
        self.assertEqual(target.aember_captured, 1)
        self.assertEqual(game.players[2].aember, 2)

    def test_phylyx_the_disintegrator_action(self):
        game = new_game()
        phylyx = put_creature(game, 1, "Phylyx the Disintegrator")
        put_creature(game, 1, "Yxili Marauder")
        put_creature(game, 1, "Grabber Jammer")
        game.players[2].aember = 5
        run_hook(game, named.phylyx_the_disintegrator_action, phylyx)
        self.assertEqual(game.players[2].aember, 3)

    def test_qyxxlyx_plague_master_ignores_armor(self):
        game = new_game()
        plague_master = put_creature(game, 1, "Qyxxlyx Plague Master")
        human = put_creature(game, 2, "Champion Tabris")  # human, knight, armor 2
        run_hook(game, named.qyxxlyx_plague_master_after, plague_master)
        self.assertEqual(human.type_object.damage, 3)

    def test_tunk_heals_when_another_mars_creature_is_played(self):
        game = new_game()
        tunk = put_creature(game, 1, "Tunk")
        tunk.type_object.damage = 4
        new_creature = make_card("Yxili Marauder", 1)
        game.players[1].play_area.add_creature(new_creature)
        drive(game._fire_event("card_played", {"player": 1, "card": new_creature}))
        self.assertEqual(tunk.type_object.damage, 0)

    def test_ulyq_megamouth_uses_a_friendly_non_mars_creature(self):
        game = new_game()
        ulyq = put_creature(game, 1, "Ulyq Megamouth")
        other = put_creature(game, 1, "Charette")
        run_hook(game, named.ulyq_megamouth_after, ulyq, answers=[[other]])
        self.assertEqual(game.players[1].aember, 1)  # reaped by default

    def test_ulyq_megamouth_does_not_offer_an_exhausted_non_mars_creature(self):
        # Picking an exhausted creature would silently waste the trigger --
        # it must be filtered out of the options, same as Dominator Bauble.
        game = new_game()
        ulyq = put_creature(game, 1, "Ulyq Megamouth")
        exhausted = put_creature(game, 1, "Charette", exhausted=True)
        before = game.players[1].aember
        run_hook(game, named.ulyq_megamouth_after, ulyq)
        self.assertEqual(game.players[1].aember, before)  # nothing usable, no-op
        self.assertTrue(exhausted.Exhausted)

    def test_uxlyx_the_zookeeper_archives_an_enemy_returning_to_owner(self):
        game = new_game()
        zookeeper = put_creature(game, 1, "Uxlyx the Zookeeper")
        target = put_creature(game, 2, "Charette")
        run_hook(game, named.uxlyx_the_zookeeper_after, zookeeper, answers=[[target]])
        self.assertIn(target, game.players[1].archive.cards())
        self.assertTrue(target.archive_return_to_owner)

    def test_vezyma_thinkdrone_may_archive_a_friendly_card(self):
        game = new_game()
        thinkdrone = put_creature(game, 1, "Vezyma Thinkdrone")
        other = put_creature(game, 1, "Charette")
        run_hook(game, named.vezyma_thinkdrone_after, thinkdrone, answers=[True, [other]])
        self.assertIn(other, game.players[1].archive.cards())

    def test_vezyma_thinkdrone_may_archive_itself(self):
        # Printed text has no "another" qualifier, unlike Chuff Ape's
        # sacrifice -- Vezyma may target itself.
        game = new_game()
        thinkdrone = put_creature(game, 1, "Vezyma Thinkdrone")
        run_hook(game, named.vezyma_thinkdrone_after, thinkdrone, answers=[True, [thinkdrone]])
        self.assertIn(thinkdrone, game.players[1].archive.cards())

    def test_yxili_marauder_gets_power_from_its_own_aember(self):
        game = new_game()
        marauder = put_creature(game, 1, "Yxili Marauder")  # put_creature already registers its passive
        self.assertEqual(game.get_power(marauder), 2)
        marauder.aember_captured = 3
        self.assertEqual(game.get_power(marauder), 5)

    def test_yxili_marauder_play_captures_per_ready_mars_creature(self):
        game = new_game()
        marauder = put_creature(game, 1, "Yxili Marauder")
        put_creature(game, 1, "Grabber Jammer")
        game.players[2].aember = 5
        run_hook(game, named.yxili_marauder_play, marauder)
        self.assertEqual(marauder.aember_captured, 2)

    def test_yxilo_bolter_purges_if_the_damage_is_lethal(self):
        game = new_game()
        bolter = put_creature(game, 1, "Yxilo Bolter")
        weak = put_creature(game, 2, "Drumble")  # power 2, dies to 2 damage
        run_hook(game, named.yxilo_bolter_after, bolter, answers=[[weak]])
        self.assertIn(weak, game.players[2].purged.cards())

    def test_yxilo_bolter_does_not_purge_if_the_damage_is_not_lethal(self):
        game = new_game()
        bolter = put_creature(game, 1, "Yxilo Bolter")
        tough = put_creature(game, 2, "Kelifi Dragon")
        run_hook(game, named.yxilo_bolter_after, bolter, answers=[[tough]])
        self.assertIn(tough, game.players[2].play_area.creatures)
        self.assertEqual(tough.type_object.damage, 2)

    def test_yxilx_dominator_enters_play_stunned(self):
        game = new_game()
        dominator = put_creature(game, 1, "Yxilx Dominator")
        named.yxilx_dominator_register(game, dominator)
        self.assertTrue(dominator.stunned)

    def test_zorg_enters_stunned_and_stuns_target_and_neighbors_before_fight(self):
        game = new_game()
        zorg = put_creature(game, 1, "Zorg")
        named.zorg_register(game, zorg)
        self.assertTrue(zorg.stunned)
        zorg.stunned = False  # clear for this test so it can actually fight
        left = put_creature(game, 2, "Drumble", flank="left")
        mid = put_creature(game, 2, "Kelifi Dragon")
        right = put_creature(game, 2, "Charette", flank="right")
        drive(game._fight(1, zorg), answers=[[mid]])
        self.assertTrue(left.stunned)
        self.assertTrue(mid.stunned)
        self.assertTrue(right.stunned)

    def test_zyzzix_the_many_may_archive_and_grows(self):
        game = new_game()
        zyzzix = put_creature(game, 1, "Zyzzix the Many")
        card_to_reveal = hand_card(game, 1, "Charette")
        run_hook(game, named.zyzzix_the_many_after, zyzzix, answers=[True, [card_to_reveal]])
        self.assertIn(card_to_reveal, game.players[1].archive.cards())
        self.assertEqual(zyzzix.power_counters, 3)

    def test_zyzzix_the_many_only_offers_creatures_to_reveal(self):
        # "Reveal a CREATURE from your hand" -- an action/artifact/upgrade
        # card in hand must not be a legal choice.
        game = new_game()
        game.players[1].hand.take_all()  # clear the starting hand
        zyzzix = put_creature(game, 1, "Zyzzix the Many")
        non_creature = hand_card(game, 1, "Squawker")
        before = len(game.players[1].hand)
        run_hook(game, named.zyzzix_the_many_after, zyzzix)
        self.assertEqual(zyzzix.power_counters, 0)  # no legal target, no-op
        self.assertIn(non_creature, game.players[1].hand.cards())
        self.assertEqual(len(game.players[1].hand), before)


class TestMarsUpgrades(unittest.TestCase):
    def test_biomatrix_backup_sends_host_to_archive_on_destroy(self):
        game = new_game()
        host = put_creature(game, 1, "Drumble")
        upgrade = make_card("Biomatrix Backup", 1)
        upgrade.type_object.host = host
        host.type_object.upgrades.append(upgrade)
        named.biomatrix_backup_register(game, upgrade)
        drive(game.destroy_cards([host]))
        self.assertIn(host, game.players[1].archive.cards())

    def test_brain_stem_antenna_readies_and_reclassifies_host(self):
        game = new_game()
        game.players[1].selected_house = House.MARS
        host = put_creature(game, 1, "Charette", exhausted=True, can_be_used=False)
        upgrade = make_card("Brain Stem Antenna", 1)
        upgrade.type_object.host = host
        host.type_object.upgrades.append(upgrade)
        named.brain_stem_antenna_register(game, upgrade)
        new_mars_creature = make_card("Yxili Marauder", 1)
        game.players[1].play_area.add_creature(new_mars_creature)
        drive(game._fire_event("card_played", {"player": 1, "card": new_mars_creature}))
        self.assertFalse(host.Exhausted)
        self.assertTrue(host.CanBeUsed)
        self.assertEqual(game.get_effective_house(host), House.MARS)

    def test_mars_tallies_count_a_creature_rehoused_by_brain_stem_antenna(self):
        # A "for each friendly ready Mars creature" tally (Psychic Network,
        # here) must count a creature re-housed to Mars for the turn (Brain
        # Stem Antenna), not just creatures printed Mars -- Milestone:
        # core-engine sweep, bug #5.
        game = new_game()
        game.players[1].selected_house = House.MARS
        host = put_creature(game, 1, "Charette", exhausted=True, can_be_used=False)  # Dis, not Mars
        upgrade = make_card("Brain Stem Antenna", 1)
        upgrade.type_object.host = host
        host.type_object.upgrades.append(upgrade)
        named.brain_stem_antenna_register(game, upgrade)
        new_mars_creature = make_card("Yxili Marauder", 1)
        game.players[1].play_area.add_creature(new_mars_creature)
        drive(game._fire_event("card_played", {"player": 1, "card": new_mars_creature}))
        self.assertFalse(host.Exhausted)  # readied by Brain Stem Antenna, and now effectively Mars
        game.players[2].aember = 5
        card = make_card("Psychic Network", 1)
        run_hook(game, named.psychic_network, card)
        self.assertEqual(game.players[1].aember, 1)  # only `host` is ready -- new_mars_creature entered exhausted

    def test_jammer_pack_raises_opponent_key_cost(self):
        game = new_game()
        host = put_creature(game, 1, "Drumble")
        upgrade = make_card("Jammer Pack", 1)
        upgrade.type_object.host = host
        host.type_object.upgrades.append(upgrade)
        named.jammer_pack_register(game, upgrade)
        self.assertEqual(game.players[2].get_key_forge_cost(game), 8)

    def test_red_planet_ray_gun_deals_damage_per_mars_creature_in_play(self):
        game = new_game()
        host = put_creature(game, 1, "Drumble")
        put_creature(game, 1, "Yxili Marauder")
        put_creature(game, 2, "Grabber Jammer")
        upgrade = make_card("Red Planet Ray Gun", 1)
        upgrade.type_object.host = host
        host.type_object.upgrades.append(upgrade)
        named.red_planet_ray_gun_register(game, upgrade)
        target = put_creature(game, 2, "Kelifi Dragon")
        drive(named._red_planet_ray_gun_effect(game, host), answers=[[target]])
        self.assertEqual(target.type_object.damage, 2)


if __name__ == "__main__":
    unittest.main()

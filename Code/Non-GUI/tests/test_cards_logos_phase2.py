"""Tests for the 39 new Logos cards (Code/PHASE_2_PLAN.md Milestone C)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.helpers import drive, hand_card, make_card, new_game, put_artifact, put_creature, run_hook

from keyforge.effects import named
from keyforge.enums import House


class TestLogosPhase2(unittest.TestCase):
    def test_bouncing_deathquark_repeats_while_possible(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        e1 = put_creature(game, 2, "Batdrone")
        e2 = put_creature(game, 2, "Dr. Escotera")
        f1 = put_creature(game, 1, "Batdrone")
        f2 = put_creature(game, 1, "Dr. Escotera")
        card = make_card("Bouncing Deathquark", 1)
        run_hook(game, named.bouncing_deathquark, card, answers=[[e1], [f1], True, [e2], [f2]])
        self.assertNotIn(e1, p2.play_area.creatures)
        self.assertNotIn(e2, p2.play_area.creatures)
        self.assertNotIn(f1, p1.play_area.creatures)
        self.assertNotIn(f2, p1.play_area.creatures)

    def test_dimension_door_makes_reap_gain_a_steal(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        p2.aember = 5
        door = make_card("Dimension Door", 1)
        run_hook(game, named.dimension_door, door)
        reaper = put_creature(game, 1, "Batdrone", exhausted=False, can_be_used=True)
        drive(game._reap(1, reaper))
        self.assertEqual(p1.aember, 1)
        self.assertEqual(p2.aember, 4)

    def test_effervescent_principle_halves_aember_and_gains_chain(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        p1.aember = 7
        p2.aember = 4
        card = make_card("Effervescent Principle", 1)
        run_hook(game, named.effervescent_principle, card)
        self.assertEqual(p1.aember, 4)  # loses 3 (floor(7/2))
        self.assertEqual(p2.aember, 2)  # loses 2 (floor(4/2))
        self.assertEqual(p1.chains, 1)

    def test_foggify_disables_enemy_fighting_next_turn(self):
        game = new_game()
        card = make_card("Foggify", 1)
        run_hook(game, named.foggify, card)
        attacker = put_creature(game, 2, "Batdrone", exhausted=False, can_be_used=True)
        put_creature(game, 1, "Dr. Escotera")
        self.assertEqual(game.legal_fight_targets(attacker), [])

    def test_interdimensional_graft_steals_forgers_remaining_aember(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        card = make_card("Interdimensional Graft", 1)
        run_hook(game, named.interdimensional_graft, card)
        p2.aember = 9
        drive(game._pay_and_forge_key(2, 6))
        self.assertEqual(p1.aember, 3)  # p2's remaining 3 after paying 6 of 9
        self.assertEqual(p2.aember, 0)

    def test_interdimensional_graft_does_not_fire_for_own_forge(self):
        game = new_game()
        p1 = game.players[1]
        card = make_card("Interdimensional Graft", 1)
        run_hook(game, named.interdimensional_graft, card)
        p1.aember = 9
        drive(game._pay_and_forge_key(1, 6))
        self.assertEqual(p1.aember, 3)  # unaffected: p1's own forge doesn't trigger it

    def test_knowledge_is_power_archive_mode(self):
        game = new_game()
        p1 = game.players[1]
        c = hand_card(game, 1, "Batdrone")
        card = make_card("Knowledge is Power", 1)
        run_hook(game, named.knowledge_is_power, card, answers=["Archive a card", [c]])
        self.assertIn(c, p1.archive.cards())

    def test_knowledge_is_power_gain_mode(self):
        game = new_game()
        p1 = game.players[1]
        p1.archive.add(make_card("Batdrone", 1))
        p1.archive.add(make_card("Dr. Escotera", 1))
        card = make_card("Knowledge is Power", 1)
        run_hook(game, named.knowledge_is_power, card, answers=["Gain 1Æ per archived card"])
        self.assertEqual(p1.aember, 2)

    def test_neuro_syphon_steals_and_draws_if_opponent_ahead(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        p2.aember = 3
        card = make_card("Neuro Syphon", 1)
        before = len(p1.hand)
        run_hook(game, named.neuro_syphon, card)
        self.assertEqual(p1.aember, 1)
        self.assertEqual(p2.aember, 2)
        self.assertEqual(len(p1.hand), before + 1)

    def test_positron_bolt_hits_flank_then_neighbor_chain(self):
        game = new_game()
        p2 = game.players[2]
        # Powers all comfortably survive their share of damage, so the
        # damage stays visible (a destroyed creature's damage resets to 0
        # on leaving play).
        a = put_creature(game, 2, "Brain Eater", flank="left")  # power 6, flank: takes 3
        b = put_creature(game, 2, "Titan Mechanic")  # power 6, middle neighbor: takes 2
        c = put_creature(game, 2, "Doc Bookton")  # power 5, other flank: takes 1
        self.assertEqual(list(p2.play_area.creatures), [a, b, c])
        card = make_card("Positron Bolt", 1)
        run_hook(game, named.positron_bolt, card, answers=[[a]])
        self.assertEqual(a.type_object.damage, 3)
        self.assertEqual(b.type_object.damage, 2)
        self.assertEqual(c.type_object.damage, 1)

    def test_random_access_archives_archives_deck_top(self):
        game = new_game()
        p1 = game.players[1]
        top = make_card("Batdrone", 1)
        p1.deck.put_on_top(top)
        card = make_card("Random Access Archives", 1)
        run_hook(game, named.random_access_archives, card)
        self.assertIn(top, p1.archive.cards())

    def test_remote_access_uses_enemy_artifact_as_if_yours(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        art = put_artifact(game, 2, "Library of Babble", exhausted=False)
        card = make_card("Remote Access", 1)
        before = len(p1.hand)
        run_hook(game, named.remote_access, card, answers=[[art]])
        self.assertEqual(len(p1.hand), before + 1)
        self.assertEqual(art.controller, 2)  # control reverts after one-time use

    def test_reverse_time_swaps_deck_and_discard(self):
        game = new_game()
        p1 = game.players[1]
        p1.deck._cards.clear()
        p1.discard._cards.clear()
        d = make_card("Batdrone", 1)
        p1.deck.put_on_top(d)
        disc = make_card("Dr. Escotera", 1)
        p1.discard.push(disc)
        card = make_card("Reverse Time", 1)
        run_hook(game, named.reverse_time, card)
        self.assertEqual(len(p1.discard), 1)
        self.assertIn(d, p1.discard.cards())
        self.assertIn(disc, p1.deck.cards())

    def test_twin_bolt_emission_hits_two_different_creatures(self):
        game = new_game()
        a = put_creature(game, 2, "Brain Eater")  # power 6, survives 2 damage
        b = put_creature(game, 2, "Titan Mechanic")  # power 6, survives 2 damage
        card = make_card("Twin Bolt Emission", 1)
        run_hook(game, named.twin_bolt_emission, card, answers=[[a], [b]])
        self.assertEqual(a.type_object.damage, 2)
        self.assertEqual(b.type_object.damage, 2)

    def test_anomaly_exploiter_destroys_only_damaged_creature(self):
        game = new_game()
        p2 = game.players[2]
        damaged = put_creature(game, 2, "Batdrone")
        damaged.type_object.damage = 1
        healthy = put_creature(game, 2, "Dr. Escotera")
        card = make_card("Anomaly Exploiter", 1)
        run_hook(game, named.anomaly_exploiter, card, answers=[[damaged]])
        self.assertNotIn(damaged, p2.play_area.creatures)
        self.assertIn(healthy, p2.play_area.creatures)

    def test_chaos_portal_plays_matching_house_top_card(self):
        game = new_game()
        p1 = game.players[1]
        top = make_card("Batdrone", 1)  # Logos
        p1.deck.put_on_top(top)
        card = make_card("Chaos Portal", 1)
        run_hook(game, named.chaos_portal, card, answers=[House.LOGOS, "left"])
        self.assertIn(top, p1.play_area.creatures)

    def test_chaos_portal_does_not_play_mismatched_house(self):
        game = new_game()
        p1 = game.players[1]
        top = make_card("Charette", 1)  # Dis
        p1.deck.put_on_top(top)
        card = make_card("Chaos Portal", 1)
        run_hook(game, named.chaos_portal, card, answers=[House.LOGOS])
        self.assertNotIn(top, p1.play_area.creatures)
        self.assertIs(p1.deck.peek_top(), top)

    def test_mobius_scroll_archives_itself_and_up_to_2_cards(self):
        game = new_game()
        p1 = game.players[1]
        scroll = put_artifact(game, 1, "Mobius Scroll", exhausted=False)
        c1 = hand_card(game, 1, "Batdrone")
        c2 = hand_card(game, 1, "Dr. Escotera")
        run_hook(game, named.mobius_scroll, scroll, answers=[[c1, c2]])
        self.assertIn(scroll, p1.archive.cards())
        self.assertIn(c1, p1.archive.cards())
        self.assertIn(c2, p1.archive.cards())
        self.assertNotIn(scroll, p1.play_area.artifacts)

    def test_pocket_universe_moves_aember_and_is_spendable(self):
        game = new_game()
        p1 = game.players[1]
        p1.aember = 3
        card = put_artifact(game, 1, "Pocket Universe", exhausted=False)
        run_hook(game, card.card_def.on_action, card)
        self.assertEqual(p1.aember, 2)
        self.assertEqual(card.aember_stored, 1)
        self.assertTrue(card.card_def.spendable_for_keys)

    def test_spangler_box_purges_and_flips_control(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        box = put_artifact(game, 1, "Spangler Box", exhausted=False)
        target = put_creature(game, 2, "Batdrone")
        run_hook(game, named.spangler_box, box, answers=[[target]])
        self.assertIn(target, p2.purged.cards())
        self.assertEqual(target.purged_by, box)
        self.assertIn(box, p2.play_area.artifacts)
        self.assertEqual(box.controller, 2)

    def test_spangler_box_returns_purged_cards_when_it_leaves_play(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        box = put_artifact(game, 1, "Spangler Box", exhausted=False)
        target = put_creature(game, 2, "Batdrone")
        run_hook(game, named.spangler_box, box, answers=[[target]])
        # box now controlled by p2; destroy it (destroy_cards removes it from
        # play and calls leave_play itself) and check the purged card returns
        drive(game.destroy_cards([box]))
        self.assertIn(target, p2.play_area.creatures)
        self.assertIsNone(target.purged_by)

    def test_spectral_tunneler_grants_flank_status_and_after_reap_draw(self):
        game = new_game()
        p1 = game.players[1]
        middle = put_creature(game, 1, "Dr. Escotera")
        put_creature(game, 1, "Batdrone", flank="left")
        put_creature(game, 1, "Doc Bookton")
        self.assertFalse(p1.play_area.is_flank(middle))
        tunneler = put_artifact(game, 1, "Spectral Tunneler", exhausted=False)
        run_hook(game, named.spectral_tunneler, tunneler, answers=[[middle]])
        self.assertTrue(p1.play_area.is_flank(middle))
        before = len(p1.hand)
        middle.Exhausted = False
        middle.CanBeUsed = True
        drive(game._reap(1, middle))
        self.assertEqual(len(p1.hand), before + 1)
        drive(game._cleanup_turn(1))
        self.assertFalse(p1.play_area.is_flank(middle))

    def test_strange_gizmo_destroys_everything_after_its_controller_forges(self):
        game = new_game()
        p1 = game.players[1]
        gizmo = put_artifact(game, 1, "Strange Gizmo")
        gizmo.card_def.register_passive(game, gizmo)
        a = put_creature(game, 1, "Batdrone")
        b = put_creature(game, 2, "Dr. Escotera")
        p1.aember = 6
        drive(game._pay_and_forge_key(1, 6))
        self.assertNotIn(a, game.players[1].play_area.creatures)
        self.assertNotIn(b, game.players[2].play_area.creatures)
        self.assertNotIn(gizmo, game.players[1].play_area.artifacts)

    def test_batdrone_fight_steals(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        p2.aember = 3
        card = put_creature(game, 1, "Batdrone")
        run_hook(game, named.batdrone_fight, card)
        self.assertEqual(p1.aember, 1)
        self.assertEqual(p2.aember, 2)

    def test_brain_eater_draws_on_destroyed_fighting(self):
        game = new_game()
        p1 = game.players[1]
        brain_eater = put_creature(game, 1, "Brain Eater", exhausted=False, can_be_used=True)  # power 6
        weak = put_creature(game, 2, "Batdrone")  # power 2, not elusive
        before = len(p1.hand)
        drive(game._fight(1, brain_eater), answers=[[weak]])
        self.assertEqual(len(p1.hand), before + 1)

    def test_dextre_returns_to_top_of_deck_when_destroyed(self):
        game = new_game()
        p1 = game.players[1]
        dextre = put_creature(game, 1, "Dextre")
        drive(game.destroy_cards([dextre]))
        self.assertIs(p1.deck.peek_top(), dextre)

    def test_dr_escotera_gains_for_opponent_keys(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        p2.keys = 2
        card = make_card("Dr. Escotera", 1)
        run_hook(game, named.dr_escotera, card)
        self.assertEqual(p1.aember, 2)

    def test_dysania_discards_archive_and_gains(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        p2.archive.add(make_card("Batdrone", 2))
        p2.archive.add(make_card("Dr. Escotera", 2))
        card = make_card("Dysania", 1)
        run_hook(game, named.dysania, card)
        self.assertEqual(len(p2.archive), 0)
        self.assertEqual(len(p2.discard), 2)
        self.assertEqual(p1.aember, 2)

    def test_harland_mindlock_takes_control_until_it_leaves_play(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        harland = put_creature(game, 1, "Harland Mindlock")
        flank_creature = put_creature(game, 2, "Batdrone", flank="left")
        run_hook(game, named.harland_mindlock, harland, answers=[[flank_creature], "left"])
        self.assertIn(flank_creature, p1.play_area.creatures)
        p1.play_area.remove(harland)
        game.leave_play(harland)
        self.assertIn(flank_creature, p2.play_area.creatures)

    def test_neutron_shark_chains_while_discarding_non_logos(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        shark = put_creature(game, 1, "Neutron Shark", exhausted=False, can_be_used=True)
        e1 = put_creature(game, 2, "Batdrone")
        e2 = put_creature(game, 2, "Dr. Escotera")
        f1 = put_creature(game, 1, "Doc Bookton")
        p1.deck._cards.clear()
        p1.deck.put_on_top(make_card("Charette", 1))  # Dis -> triggers again
        run_hook(game, named.neutron_shark, shark, answers=[[e1], [f1], [e2], [shark]])
        self.assertNotIn(e1, p2.play_area.creatures)
        self.assertNotIn(f1, p1.play_area.creatures)

    def test_novu_archaeologist_archives_from_discard(self):
        game = new_game()
        p1 = game.players[1]
        c = make_card("Batdrone", 1)
        p1.discard.push(c)
        novu = put_creature(game, 1, "Novu Archaeologist", exhausted=False, can_be_used=True)
        run_hook(game, named.novu_archaeologist, novu, answers=[[c]])
        self.assertIn(c, p1.archive.cards())

    def test_ozmo_is_elusive_and_has_no_valid_target_in_this_pool(self):
        game = new_game()
        ozmo = put_creature(game, 1, "Ozmo, Martianologist")
        self.assertIn("elusive", game.get_keywords(ozmo))
        run_hook(game, named.ozmo, ozmo)  # no Mars creatures -> shortfall, no crash

    def test_replicator_triggers_another_creatures_reap(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        replicator = put_creature(game, 1, "Replicator")
        doc = put_creature(game, 2, "Doc Bookton")  # After Reap: draw a card, controlled by p2
        before_p1 = len(p1.hand)
        run_hook(game, named.replicator, replicator, answers=[[doc]])
        self.assertEqual(len(p1.hand), before_p1 + 1)  # p1 got the draw, "as if" controlling doc
        self.assertEqual(doc.controller, 2)  # control reverts after

    def test_research_smoko_archives_on_destroyed(self):
        game = new_game()
        p1 = game.players[1]
        top = make_card("Batdrone", 1)
        p1.deck.put_on_top(top)
        smoko = put_creature(game, 1, "Research Smoko")
        drive(game.destroy_cards([smoko]))
        self.assertIn(top, p1.archive.cards())

    def test_research_smoko_destroyed_pays_the_controller_not_the_owner(self):
        # Same "user decision" as Truebaru (Dis): an unqualified "Destroyed:"
        # ability archives from whoever controls the creature at the moment
        # of destruction, not its original owner, when a control-change
        # effect moved it first.
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        owner_top = make_card("Batdrone", 1)
        p1.deck.put_on_top(owner_top)
        controller_top = make_card("Dr. Escotera", 2)
        p2.deck.put_on_top(controller_top)
        smoko = put_creature(game, 1, "Research Smoko")  # owned by p1
        smoko.controller = 2  # taken control of by p2 before being destroyed
        drive(game.destroy_cards([smoko]))
        self.assertIn(controller_top, p2.archive.cards())  # controller's deck
        self.assertNotIn(owner_top, p1.archive.cards())  # not the owner's

    def test_skippy_timehog_prevents_opponent_from_using_cards_next_turn(self):
        game = new_game()
        p2 = game.players[2]
        card = make_card("Skippy Timehog", 1)
        run_hook(game, named.skippy_timehog, card)
        self.assertTrue(p2.get_cannot_use_cards(game))

    def test_vespilon_theorist_archives_matching_house_and_gains(self):
        game = new_game()
        p1 = game.players[1]
        top = make_card("Batdrone", 1)  # Logos
        p1.deck.put_on_top(top)
        vt = put_creature(game, 1, "Vespilon Theorist", exhausted=False, can_be_used=True)
        run_hook(game, named.vespilon_theorist, vt, answers=[House.LOGOS])
        self.assertIn(top, p1.archive.cards())
        self.assertEqual(p1.aember, 1)

    def test_vespilon_theorist_discards_mismatched_house(self):
        game = new_game()
        p1 = game.players[1]
        top = make_card("Charette", 1)  # Dis
        p1.deck.put_on_top(top)
        vt = put_creature(game, 1, "Vespilon Theorist")
        run_hook(game, named.vespilon_theorist, vt, answers=[House.LOGOS])
        self.assertIn(top, p1.discard.cards())
        self.assertEqual(p1.aember, 0)

    def test_veylan_analyst_gains_when_controller_uses_an_artifact(self):
        game = new_game()
        p1 = game.players[1]
        put_creature(game, 1, "Veylan Analyst")  # put_creature already calls register_passive
        art = put_artifact(game, 1, "Library of Babble", exhausted=False)
        drive(game._use_action(1, art))
        self.assertEqual(p1.aember, 1)

    def test_experimental_therapy_grants_versatile_and_stuns_host(self):
        game = new_game()
        host = put_creature(game, 1, "Batdrone", exhausted=False)
        upgrade = make_card("Experimental Therapy", 1)
        upgrade.type_object.host = host
        host.type_object.upgrades.append(upgrade)
        run_hook(game, named.experimental_therapy, upgrade)
        self.assertTrue(host.stunned)
        self.assertTrue(host.Exhausted)
        self.assertIn("versatile", game.get_keywords(host))

    def test_experimental_therapy_play_effect_fires_through_the_real_play_path(self):
        # `_play_card`'s UPGRADE branch must actually call on_play -- it's
        # the only upgrade in the pool that has one, and it's easy for a
        # register_passive-only code path to silently skip it.
        game = new_game()
        host = put_creature(game, 1, "Batdrone", exhausted=False)
        upgrade = hand_card(game, 1, "Experimental Therapy")
        drive(game._play_card(1, upgrade), answers=[[host]])
        self.assertTrue(host.stunned)
        self.assertTrue(host.Exhausted)
        self.assertIn("versatile", game.get_keywords(host))

    def test_rocket_boots_readies_host_after_first_use_this_turn(self):
        game = new_game()
        host = put_creature(game, 1, "Batdrone", exhausted=False, can_be_used=True)
        upgrade = make_card("Rocket Boots", 1)
        upgrade.type_object.host = host
        host.type_object.upgrades.append(upgrade)
        named.rocket_boots_register(game, upgrade)
        drive(game._reap(1, host))
        self.assertFalse(host.Exhausted)

    def test_transposition_sandals_swaps_and_offers_use(self):
        game = new_game()
        p1 = game.players[1]
        host = put_creature(game, 1, "Batdrone", flank="left")
        other = put_creature(game, 1, "Dr. Escotera", exhausted=False, can_be_used=True)
        self.assertEqual(list(p1.play_area.creatures), [host, other])
        upgrade = make_card("Transposition Sandals", 1)
        upgrade.type_object.host = host
        host.type_object.upgrades.append(upgrade)
        named.transposition_sandals_register(game, upgrade)
        before = p1.aember
        drive(host.granted_action(game, host), answers=[[other], True])
        self.assertEqual(list(p1.play_area.creatures), [other, host])
        self.assertGreater(p1.aember, before)  # other reaped (default_chooser picks "reap" first)


if __name__ == "__main__":
    unittest.main()

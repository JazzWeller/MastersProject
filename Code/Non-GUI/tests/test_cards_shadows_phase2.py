"""Tests for the 34 new Shadows cards (Code/PHASE_2_PLAN.md Milestone C)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.helpers import drive, hand_card, make_card, new_game, put_artifact, put_creature, run_hook

from keyforge.effects import generic, named
from keyforge.enums import House


class TestShadowsPhase2(unittest.TestCase):
    def test_finishing_blow_destroys_damaged_creature_and_steals(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        p2.aember = 3
        target = put_creature(game, 2, "Charette")  # power 4
        target.type_object.damage = 2
        card = make_card("Finishing Blow", 1)
        run_hook(game, named.finishing_blow, card, answers=[[target]])
        self.assertNotIn(target, p2.play_area.creatures)
        self.assertEqual(p1.aember, 1)
        self.assertEqual(p2.aember, 2)

    def test_finishing_blow_shortfall_when_no_damaged_creature(self):
        game = new_game()
        p1 = game.players[1]
        target = put_creature(game, 2, "Charette")
        card = make_card("Finishing Blow", 1)
        run_hook(game, named.finishing_blow, card)
        self.assertIn(target, game.players[2].play_area.creatures)
        self.assertEqual(p1.aember, 0)

    def test_imperial_traitor_reveals_hand_and_finds_no_target_without_a_sanctum_card(self):
        game = new_game()
        p2 = game.players[2]
        hand_card(game, 2, "Charette")  # Mars, not Sanctum
        before = len(p2.hand)
        card = make_card("Imperial Traitor", 1)
        run_hook(game, named.imperial_traitor, card)
        events = [e for e in game.log.events if e.kind == "reveal_hand"]
        self.assertEqual(events[-1].data["player"], 2)
        self.assertEqual(len(p2.hand), before)

    def test_imperial_traitor_may_purge_a_sanctum_card_from_opponent_hand(self):
        game = new_game()
        p2 = game.players[2]
        sanctum_card = hand_card(game, 2, "Begone!")
        other_card = hand_card(game, 2, "Charette")
        card = make_card("Imperial Traitor", 1)
        run_hook(game, named.imperial_traitor, card, answers=[[sanctum_card]])
        self.assertNotIn(sanctum_card, p2.hand.cards())
        self.assertIn(other_card, p2.hand.cards())

    def test_key_of_darkness_uses_plus_6_when_opponent_has_aember(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        p2.aember = 1
        p1.aember = 12  # base cost 6 + 6
        card = make_card("Key of Darkness", 1)
        run_hook(game, named.key_of_darkness, card)
        self.assertEqual(p1.keys, 1)
        self.assertEqual(p1.aember, 0)

    def test_key_of_darkness_uses_plus_2_when_opponent_has_no_aember(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        p2.aember = 0
        p1.aember = 8  # base cost 6 + 2
        card = make_card("Key of Darkness", 1)
        run_hook(game, named.key_of_darkness, card)
        self.assertEqual(p1.keys, 1)
        self.assertEqual(p1.aember, 0)

    def test_key_of_darkness_still_works_while_miasma_skips_the_normal_forge_step(self):
        # Miasma/The Sting say "skip your forge a key step" -- that must
        # only disable the turn's own Step 1, not a card effect (Key of
        # Darkness) that forges a key directly.
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        p2.aember = 1
        p1.aember = 12  # base cost 6 + 6
        miasma = make_card("Miasma", 2)
        run_hook(game, generic.duration_effect("CanKeyForge", "=", False, 2, "enemy"), miasma)
        self.assertFalse(p1.get_can_key_forge(game))  # the turn-step block is active on p1
        card = make_card("Key of Darkness", 1)
        run_hook(game, named.key_of_darkness, card)
        self.assertEqual(p1.keys, 1)
        self.assertEqual(p1.aember, 0)

    def test_poison_wave_damages_every_creature_in_play(self):
        game = new_game()
        a = put_creature(game, 1, "Charette")  # power 4
        b = put_creature(game, 2, "Drumble")  # power 2, destroyed by 2 damage
        card = make_card("Poison Wave", 1)
        run_hook(game, named.poison_wave, card)
        self.assertEqual(a.type_object.damage, 2)
        self.assertNotIn(b, game.players[2].play_area.creatures)

    def test_routine_job_steals_more_for_each_copy_in_discard(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        p2.aember = 5
        p1.discard.push(make_card("Routine Job", 1))
        p1.discard.push(make_card("Routine Job", 1))
        card = make_card("Routine Job", 1)
        run_hook(game, named.routine_job, card)
        self.assertEqual(p1.aember, 3)  # 1 base + 2 copies already in discard
        self.assertEqual(p2.aember, 2)

    def test_treasure_map_gains_3_when_its_the_only_card_played(self):
        game = new_game()
        p1 = game.players[1]
        p1.hand.take_all()
        card = hand_card(game, 1, "Treasure Map")
        p1.selected_house = House.SHADOWS
        drive(game._play_card(1, card))
        self.assertEqual(p1.aember, 4)  # printed 1Æ bonus + the conditional 3Æ gain
        self.assertFalse(p1.get_can_play_cards(game))

    def test_treasure_map_gains_nothing_if_another_card_already_played(self):
        game = new_game()
        p1 = game.players[1]
        p1.hand.take_all()
        p1.selected_house = House.SHADOWS
        first = hand_card(game, 1, "Hidden Stash")
        drive(game._play_card(1, first))
        card = hand_card(game, 1, "Treasure Map")
        drive(game._play_card(1, card))
        # Both cards' printed 1Æ bonuses still apply; only the conditional 3Æ is skipped.
        self.assertEqual(p1.aember, 2)

    def test_customs_office_makes_opponent_pay_to_play_an_artifact(self):
        game = new_game()
        p2 = game.players[2]
        put_artifact(game, 1, "Customs Office")
        toll = p2.get_artifact_play_toll(game)
        self.assertEqual(toll, (1, 1))

    def test_evasion_sigil_exhausts_attacker_with_no_effect_on_active_house_match(self):
        game = new_game()
        p1 = game.players[1]
        put_artifact(game, 2, "Evasion Sigil")
        attacker = put_creature(game, 1, "Old Bruno")
        p1.selected_house = House.SHADOWS
        p1.deck._cards.clear()
        p1.deck.put_on_top(make_card("Old Bruno", 1))  # Shadows: matches active house
        event = {"attacker": attacker, "cancelled": False}
        drive(game._fire_event("before_fight", event))
        self.assertTrue(event["cancelled"])
        self.assertEqual(len(p1.discard), 1)

    def test_evasion_sigil_no_effect_when_discard_is_not_the_active_house(self):
        game = new_game()
        p1 = game.players[1]
        put_artifact(game, 2, "Evasion Sigil")
        attacker = put_creature(game, 1, "Old Bruno")
        p1.selected_house = House.SHADOWS
        p1.deck._cards.clear()
        p1.deck.put_on_top(make_card("Charette", 1))  # Dis: doesn't match active house
        event = {"attacker": attacker, "cancelled": False}
        drive(game._fire_event("before_fight", event))
        self.assertFalse(event["cancelled"])
        self.assertEqual(len(p1.discard), 1)

    def test_longfused_mines_sacrifices_and_damages_non_flank_enemies_only(self):
        game = new_game()
        mines = put_artifact(game, 1, "Longfused Mines")
        left = put_creature(game, 2, "Truebaru", flank="left")  # power 7
        middle = put_creature(game, 2, "Charette")  # power 4
        right = put_creature(game, 2, "Doc Bookton", flank="right")  # power 5
        run_hook(game, named.longfused_mines, mines)
        self.assertNotIn(mines, game.players[1].play_area.artifacts)
        self.assertEqual(middle.type_object.damage, 3)
        self.assertEqual(left.type_object.damage, 0)
        self.assertEqual(right.type_object.damage, 0)

    def test_masterplan_stores_a_card_then_omni_plays_it_and_sacrifices(self):
        game = new_game()
        p1 = game.players[1]
        masterplan = put_artifact(game, 1, "Masterplan")
        stashed = hand_card(game, 1, "Charette")
        p1.selected_house = House.DIS
        run_hook(game, named.masterplan_play, masterplan, answers=[[stashed]])
        self.assertEqual(masterplan.under_cards, [stashed])
        self.assertNotIn(stashed, p1.hand.cards())
        run_hook(game, named.masterplan_omni, masterplan)
        self.assertIn(stashed, p1.play_area.creatures)
        self.assertNotIn(masterplan, p1.play_area.artifacts)

    def test_masterplan_omni_sacrifices_even_with_nothing_beneath(self):
        game = new_game()
        p1 = game.players[1]
        masterplan = put_artifact(game, 1, "Masterplan")
        run_hook(game, named.masterplan_omni, masterplan)
        self.assertNotIn(masterplan, p1.play_area.artifacts)

    def test_safe_place_moves_aember_and_is_spendable_for_keys(self):
        game = new_game()
        p1 = game.players[1]
        p1.aember = 3
        card = put_artifact(game, 1, "Safe Place", exhausted=False)
        run_hook(game, card.card_def.on_action, card)
        self.assertEqual(p1.aember, 2)
        self.assertEqual(card.aember_stored, 1)
        self.assertTrue(card.card_def.spendable_for_keys)

    def test_seeker_needle_and_mack_the_knife_gain_aember_on_kill(self):
        game = new_game()
        p1 = game.players[1]
        for card_name, effect in (("Seeker Needle", named.seeker_needle), ("Mack the Knife", named.mack_the_knife)):
            with self.subTest(card_name):
                target = put_creature(game, 2, "Drumble")  # power 2
                target.type_object.damage = 1
                source = make_card(card_name, 1)
                before = p1.aember
                run_hook(game, effect, source, answers=[[target]])
                self.assertNotIn(target, game.players[2].play_area.creatures)
                self.assertEqual(p1.aember, before + 1)

    def test_skeleton_key_captures_with_a_friendly_creature(self):
        game = new_game()
        p2 = game.players[2]
        p2.aember = 2
        creature = put_creature(game, 1, "Charette")
        key = put_artifact(game, 1, "Skeleton Key")
        run_hook(game, named.skeleton_key, key, answers=[[creature]])
        self.assertEqual(creature.aember_captured, 1)
        self.assertEqual(p2.aember, 1)

    def test_special_delivery_purges_a_destroyed_flank_creature(self):
        game = new_game()
        p2 = game.players[2]
        delivery = put_artifact(game, 1, "Special Delivery")
        target = put_creature(game, 2, "Drumble", flank="left")  # power 2, dies to 3
        run_hook(game, named.special_delivery, delivery, answers=[[target]])
        self.assertNotIn(delivery, game.players[1].play_area.artifacts)
        self.assertIn(target, p2.purged.cards())
        self.assertNotIn(target, p2.discard.cards())

    def test_speed_sigil_makes_first_creature_played_enter_ready(self):
        game = new_game()
        p1 = game.players[1]
        put_artifact(game, 1, "Speed Sigil")
        p1.selected_house = House.SHADOWS
        card = hand_card(game, 1, "Charette")
        drive(game._play_card(1, card))
        self.assertFalse(card.Exhausted)

    def test_the_sting_skips_own_forge_and_redirects_opponents_payment(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        put_artifact(game, 1, "The Sting")
        self.assertFalse(p1.get_can_key_forge(game))
        p2.aember = 6
        drive(game._pay_and_forge_key(2, 6))
        self.assertEqual(p2.keys, 1)
        self.assertEqual(p2.aember, 0)
        self.assertEqual(p1.aember, 6)

    def test_the_sting_action_sacrifices_itself(self):
        game = new_game()
        p1 = game.players[1]
        sting = put_artifact(game, 1, "The Sting")
        run_hook(game, named.the_sting_action, sting)
        self.assertNotIn(sting, p1.play_area.artifacts)

    def test_bulleteye_destroys_a_flank_creature_on_reap(self):
        game = new_game()
        bulleteye = put_creature(game, 1, "Bulleteye")
        flank_target = put_creature(game, 2, "Charette", flank="left")
        middle = put_creature(game, 2, "Drumble")
        run_hook(game, named.bulleteye, bulleteye, answers=[[flank_target]])
        self.assertNotIn(flank_target, game.players[2].play_area.creatures)
        self.assertIn(middle, game.players[2].play_area.creatures)

    def test_carlo_phantom_steals_only_when_an_artifact_is_played(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        p2.aember = 2
        put_creature(game, 1, "Carlo Phantom")
        # Drumble's own Play effect only captures if the opponent has 7+ Æ,
        # so it won't disturb p2's pool and confound the assertions below.
        creature_card = hand_card(game, 1, "Drumble")
        p1.selected_house = House.DIS
        drive(game._play_card(1, creature_card))
        self.assertEqual(p1.aember, 0)  # a creature was played, not an artifact
        artifact_card = hand_card(game, 1, "Skeleton Key")
        drive(game._play_card(1, artifact_card))
        self.assertEqual(p1.aember, 1)
        self.assertEqual(p2.aember, 1)

    def test_deipno_spymaster_may_use_the_chosen_creature(self):
        game = new_game()
        p2 = game.players[2]
        p2.aember = 1
        spymaster = put_creature(game, 1, "Deipno Spymaster")
        other = put_creature(game, 1, "Noddy the Thief", exhausted=False)
        run_hook(game, named.deipno_spymaster, spymaster, answers=[[other], True, ["action"]])
        self.assertTrue(other.Exhausted)
        self.assertEqual(game.players[1].aember, 1)
        self.assertEqual(p2.aember, 0)

    def test_faygin_returns_urchin_from_play_or_discard(self):
        game = new_game()
        p1 = game.players[1]
        faygin = put_creature(game, 1, "Faygin")
        urchin_in_discard = make_card("Urchin", 1)
        p1.discard.push(urchin_in_discard)
        run_hook(game, named.faygin, faygin, answers=[[urchin_in_discard]])
        self.assertIn(urchin_in_discard, p1.hand.cards())
        self.assertNotIn(urchin_in_discard, p1.discard.cards())

    def test_faygin_can_return_an_enemy_urchin_to_its_owners_hand(self):
        # "Return an Urchin from play" is unqualified by owner -- an enemy
        # Urchin in play is a legal target, and it goes to ITS owner's
        # hand, not Faygin's controller's.
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        faygin = put_creature(game, 1, "Faygin")
        enemy_urchin = put_creature(game, 2, "Urchin")
        run_hook(game, named.faygin, faygin, answers=[[enemy_urchin]])
        self.assertIn(enemy_urchin, p2.hand.cards())
        self.assertNotIn(enemy_urchin, p1.hand.cards())
        self.assertNotIn(enemy_urchin, p2.play_area.creatures)

    def test_macis_asp_has_skirmish_and_poison_keywords(self):
        game = new_game()
        card = put_creature(game, 1, "Macis Asp")
        self.assertEqual(game.get_keywords(card), frozenset({"skirmish", "poison"}))

    def test_magda_the_rat_steals_on_play_and_when_it_leaves_play(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        p2.aember = 5
        card = hand_card(game, 1, "Magda the Rat")
        p1.selected_house = House.SHADOWS
        drive(game._play_card(1, card))
        self.assertEqual(p1.aember, 2)
        self.assertEqual(p2.aember, 3)
        p1.play_area.remove(card)
        game.leave_play(card)
        self.assertEqual(p1.aember, 0)
        self.assertEqual(p2.aember, 5)

    def test_mooncurser_and_umbra_steal_on_fight(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        for name in ("Mooncurser", "Umbra"):
            with self.subTest(name):
                p2.aember = 1
                attacker = put_creature(game, 1, name, exhausted=False)
                run_hook(game, attacker.card_def.on_fight, attacker)
                self.assertEqual(p2.aember, 0)
        self.assertEqual(p1.aember, 2)

    def test_dodger_steals_after_fight(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        p2.aember = 1
        dodger = put_creature(game, 1, "Dodger", exhausted=False)
        run_hook(game, dodger.card_def.on_fight, dodger)
        self.assertEqual(p1.aember, 1)
        self.assertEqual(p2.aember, 0)

    def test_nexus_uses_an_enemy_artifact_as_if_it_were_its_own(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        p2.aember = 3
        nexus = put_creature(game, 1, "Nexus")
        enemy_artifact = put_artifact(game, 2, "Skeleton Key", exhausted=False)
        # Skeleton Key resolves as if controlled by Nexus's controller, so the
        # only "friendly creature" it can capture Æmber onto is Nexus itself.
        run_hook(game, named.nexus, nexus, answers=[[enemy_artifact], [nexus]])
        self.assertEqual(nexus.aember_captured, 1)
        self.assertEqual(p2.aember, 2)
        self.assertEqual(enemy_artifact.controller, 2)  # control reverts after the one-time use
        self.assertTrue(enemy_artifact.Exhausted)

    def test_selwyn_the_fence_moves_aember_from_a_card_to_the_pool(self):
        game = new_game()
        p1 = game.players[1]
        selwyn = put_creature(game, 1, "Selwyn the Fence")
        source = put_creature(game, 1, "Charette")
        source.aember_captured = 2
        run_hook(game, named.selwyn_the_fence, selwyn, answers=[[source]])
        self.assertEqual(source.aember_captured, 1)
        self.assertEqual(p1.aember, 1)

    def test_shadow_self_has_the_no_fight_damage_keyword(self):
        game = new_game()
        shadow_self = put_creature(game, 1, "Shadow Self")
        self.assertIn("no_fight_damage", game.get_keywords(shadow_self))

    def test_shadow_self_redirects_damage_dealt_to_a_non_specter_neighbor(self):
        from keyforge.effects import steps

        game = new_game()
        neighbor = put_creature(game, 1, "Charette", flank="left")
        shadow_self = put_creature(game, 1, "Shadow Self")
        steps.deal_damage(game, neighbor, 3)
        self.assertEqual(neighbor.type_object.damage, 0)
        self.assertEqual(shadow_self.type_object.damage, 3)

    def test_shadow_self_does_not_redirect_its_own_damage(self):
        from keyforge.effects import steps

        game = new_game()
        put_creature(game, 1, "Charette", flank="left")
        shadow_self = put_creature(game, 1, "Shadow Self")
        steps.deal_damage(game, shadow_self, 4)
        self.assertEqual(shadow_self.type_object.damage, 4)

    def test_shadow_self_redirect_via_a_real_fight_moves_poisons_kill_with_it(self):
        # A poison attacker's damage on Shadow Self's neighbor redirects to
        # Shadow Self -- and poison's "any damage destroys" must follow the
        # damage to whichever creature actually received it. The neighbor
        # (power 1) takes 0 real damage and must survive; Shadow Self
        # (power 9) takes the poison hit and must die despite its power.
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        neighbor = put_creature(game, 1, "Charette", flank="left")  # power 4, not elusive
        shadow_self = put_creature(game, 1, "Shadow Self")  # power 9
        attacker = put_creature(game, 2, "Macis Asp", exhausted=False, can_be_used=True)  # power 3, skirmish+poison
        gen = game._fight(2, attacker)
        drive(gen, answers=[[neighbor]])
        self.assertIn(neighbor, p1.play_area.creatures)  # took 0 real damage: survives
        self.assertEqual(neighbor.type_object.damage, 0)
        self.assertNotIn(shadow_self, p1.play_area.creatures)  # poison killed the creature that actually took the hit

    def test_smiling_ruth_only_takes_control_after_forging_a_key_this_turn(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        ruth = put_creature(game, 1, "Smiling Ruth")
        target = put_creature(game, 2, "Charette", flank="left")
        run_hook(game, named.smiling_ruth, ruth)
        self.assertIn(target, p2.play_area.creatures)
        game.log.add("forge_key", player=1, keys=1, cost=6, turn=game.turn_number, source=None)
        run_hook(game, named.smiling_ruth, ruth, answers=[[target]])
        self.assertIn(target, p1.play_area.creatures)

    def test_sneklifter_takes_control_of_an_enemy_artifact(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        card = hand_card(game, 1, "Sneklifter")
        p1.selected_house = House.SHADOWS
        artifact = put_artifact(game, 2, "Skeleton Key")
        drive(game._play_card(1, card), answers=["left", [artifact]])
        self.assertIn(artifact, p1.play_area.artifacts)
        self.assertEqual(artifact.controller, 1)
        self.assertIsNone(artifact.house_override)  # Shadows already one of this deck's 3 houses

    def test_ring_of_invisibility_grants_elusive_and_skirmish(self):
        game = new_game()
        p1 = game.players[1]
        host = put_creature(game, 1, "Charette")
        upgrade = hand_card(game, 1, "Ring of Invisibility")
        p1.selected_house = House.DIS
        drive(game._play_card(1, upgrade), answers=[[host]])
        self.assertEqual(game.get_keywords(host), frozenset({"elusive", "skirmish"}))

    def test_silent_dagger_grants_after_reap_damage_to_a_flank_creature(self):
        game = new_game()
        p1 = game.players[1]
        host = put_creature(game, 1, "Charette")
        upgrade = hand_card(game, 1, "Silent Dagger")
        p1.selected_house = House.DIS
        drive(game._play_card(1, upgrade), answers=[[host]])
        target = put_creature(game, 2, "Drumble", flank="left")  # power 2, dies to 4
        for extra in list(host.extra_triggers["after_reap"]):
            drive(extra(game, host), answers=[[target]])
        self.assertNotIn(target, game.players[2].play_area.creatures)

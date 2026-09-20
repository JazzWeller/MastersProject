"""Tests for the 55 Sanctum cards (Code/PHASE_3_PLAN.md Milestone D.2)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.helpers import drive, hand_card, make_card, new_game, put_artifact, put_creature, run_hook

from keyforge.effects import steps
from keyforge.effects.named import sanctum as named
from keyforge.enums import House


class TestSanctumActions(unittest.TestCase):
    def test_begone_can_destroy_dis_or_gain_aember(self):
        game = new_game()
        dis_creature = put_creature(game, 2, "Charette")  # Dis
        card = make_card("Begone!", 1)
        run_hook(game, named.begone, card, answers=["Destroy each Dis creature"])
        self.assertNotIn(dis_creature, game.players[2].play_area.creatures)
        card2 = make_card("Begone!", 1)
        run_hook(game, named.begone, card2, answers=["Gain 1Æ"])
        self.assertEqual(game.players[1].aember, 1)

    def test_blinding_light_stuns_only_the_chosen_house(self):
        game = new_game()
        dis_creature = put_creature(game, 2, "Charette")
        logos_creature = put_creature(game, 1, "Doc Bookton")
        card = make_card("Blinding Light", 1)
        run_hook(game, named.blinding_light, card, answers=[House.DIS])
        self.assertTrue(dis_creature.stunned)
        self.assertFalse(logos_creature.stunned)

    def test_charge_grants_creatures_played_this_turn_a_damage_trigger(self):
        game = new_game()
        card = make_card("Charge!", 1)
        run_hook(game, named.charge, card)
        target = put_creature(game, 2, "Truebaru")
        new_creature = make_card("Drumble", 1)
        game.players[1].play_area.add_creature(new_creature)
        drive(game._fire_event("card_played", {"player": 1, "card": new_creature}), answers=[[target]])
        self.assertEqual(target.type_object.damage, 2)

    def test_cleansing_wave_heals_and_gains_per_creature_healed(self):
        game = new_game()
        a = put_creature(game, 1, "Truebaru")
        a.type_object.damage = 2
        b = put_creature(game, 2, "Charette")
        b.type_object.damage = 1
        c = put_creature(game, 2, "Drumble")  # undamaged, not healed, not counted
        card = make_card("Cleansing Wave", 1)
        run_hook(game, named.cleansing_wave, card)
        self.assertEqual(a.type_object.damage, 1)
        self.assertEqual(b.type_object.damage, 0)
        self.assertEqual(c.type_object.damage, 0)
        self.assertEqual(game.players[1].aember, 2)

    def test_clear_mind_unstuns_only_friendly_creatures(self):
        game = new_game()
        mine = put_creature(game, 1, "Drumble")
        mine.stunned = True
        theirs = put_creature(game, 2, "Charette")
        theirs.stunned = True
        card = make_card("Clear Mind", 1)
        run_hook(game, named.clear_mind, card)
        self.assertFalse(mine.stunned)
        self.assertTrue(theirs.stunned)

    def test_doorstep_to_heaven_caps_high_pools_at_five(self):
        game = new_game()
        game.players[1].aember = 8
        game.players[2].aember = 3
        card = make_card("Doorstep to Heaven", 1)
        run_hook(game, named.doorstep_to_heaven, card)
        self.assertEqual(game.players[1].aember, 5)
        self.assertEqual(game.players[2].aember, 3)

    def test_glorious_few_gains_per_excess_enemy_creature(self):
        game = new_game()
        put_creature(game, 1, "Drumble")
        put_creature(game, 2, "Charette")
        put_creature(game, 2, "Bumpsy")
        card = make_card("Glorious Few", 1)
        run_hook(game, named.glorious_few, card)
        self.assertEqual(game.players[1].aember, 1)

    def test_honorable_claim_captures_for_each_friendly_knight(self):
        game = new_game()
        game.players[2].aember = 5
        knight1 = put_creature(game, 1, "Raiding Knight")
        knight2 = put_creature(game, 1, "Sequis")
        non_knight = put_creature(game, 1, "Duma the Martyr")
        card = make_card("Honorable Claim", 1)
        run_hook(game, named.honorable_claim, card)
        self.assertEqual(knight1.aember_captured, 1)
        self.assertEqual(knight2.aember_captured, 1)
        self.assertEqual(non_knight.aember_captured, 0)

    def test_inspiration_readies_and_uses_a_friendly_creature(self):
        game = new_game()
        creature = put_creature(game, 1, "Duma the Martyr", exhausted=True)
        card = make_card("Inspiration", 1)
        run_hook(game, named.inspiration, card, answers=[[creature]])
        self.assertEqual(game.players[1].aember, 1)  # reaped by default

    def test_mighty_lance_hits_target_and_one_neighbor(self):
        game = new_game()
        left = put_creature(game, 2, "Truebaru")
        mid = put_creature(game, 2, "Kelifi Dragon")
        right = put_creature(game, 2, "Truebaru")
        card = make_card("Mighty Lance", 1)
        run_hook(game, named.mighty_lance, card, answers=[[mid], [left]])
        self.assertEqual(mid.type_object.damage, 3)
        self.assertEqual(left.type_object.damage, 3)
        self.assertEqual(right.type_object.damage, 0)

    def test_oath_of_poverty_destroys_own_artifacts_and_gains(self):
        game = new_game()
        a1 = put_artifact(game, 1, "Hallowed Blaster")
        a2 = put_artifact(game, 1, "Whispering Reliquary")
        put_artifact(game, 2, "Gorm of Omm")  # untouched
        card = make_card("Oath of Poverty", 1)
        run_hook(game, named.oath_of_poverty, card)
        self.assertNotIn(a1, game.players[1].play_area.artifacts)
        self.assertNotIn(a2, game.players[1].play_area.artifacts)
        self.assertEqual(game.players[1].aember, 4)

    def test_one_stood_against_many_fights_three_different_enemies(self):
        game = new_game()
        fighter = put_creature(game, 1, "Drumble", exhausted=True)  # power 2, won't one-shot anything
        e1 = put_creature(game, 2, "Kelifi Dragon")
        e2 = put_creature(game, 2, "Kelifi Dragon")
        e3 = put_creature(game, 2, "Kelifi Dragon")
        card = make_card("One Stood Against Many", 1)
        # `fighter` is the only friendly creature in play, so choose_cards
        # auto-selects it without consuming an answer -- only the 3 enemy
        # target choices need scripted answers.
        run_hook(game, named.one_stood_against_many, card, answers=[[e1], [e2], [e3]])
        fights = [e for e in game.log.events if e.kind == "fight" and e.data["attacker_iid"] == fighter.instance_id]
        self.assertEqual(len(fights), 3)
        targets_fought = {e.data["target_iid"] for e in fights}
        self.assertEqual(targets_fought, {e1.instance_id, e2.instance_id, e3.instance_id})

    def test_radiant_truth_stuns_non_flank_enemies_only(self):
        game = new_game()
        left = put_creature(game, 2, "Drumble")
        mid = put_creature(game, 2, "Charette")
        right = put_creature(game, 2, "Truebaru")
        card = make_card("Radiant Truth", 1)
        run_hook(game, named.radiant_truth, card)
        self.assertFalse(left.stunned)
        self.assertTrue(mid.stunned)
        self.assertFalse(right.stunned)

    def test_shield_of_justice_prevents_damage_to_friendly_creatures_this_turn(self):
        game = new_game()
        mine = put_creature(game, 1, "Drumble")
        run_hook(game, __import__("keyforge.effects.generic", fromlist=["duration_effect"]).duration_effect(
            "CannotBeDealtDamage", "=", True, 1, "self"), make_card("Shield of Justice", 1))
        steps.deal_damage(game, mine, 5)
        self.assertEqual(mine.type_object.damage, 0)

    def test_take_hostages_captures_on_friendly_fights_this_turn(self):
        game = new_game()
        game.players[2].aember = 3
        attacker = put_creature(game, 1, "Truebaru")
        put_creature(game, 2, "Charette")
        card = make_card("Take Hostages", 1)
        run_hook(game, named.take_hostages, card)
        drive(game._fight(1, attacker))
        self.assertEqual(attacker.aember_captured, 1)

    def test_terms_of_redress_captures_two_on_a_chosen_creature(self):
        game = new_game()
        game.players[2].aember = 3
        mine = put_creature(game, 1, "Drumble")
        card = make_card("Terms of Redress", 1)
        run_hook(game, named.terms_of_redress, card, answers=[[mine]])
        self.assertEqual(mine.aember_captured, 2)

    def test_the_harder_they_come_purges_high_power_only(self):
        game = new_game()
        weak = put_creature(game, 2, "Drumble")
        strong = put_creature(game, 2, "Truebaru")  # power 7
        card = make_card("The Harder They Come", 1)
        run_hook(game, named.the_harder_they_come, card, answers=[[strong]])
        self.assertIn(strong, game.players[2].purged.cards())
        self.assertIn(weak, game.players[2].play_area.creatures)

    def test_the_spirits_way_destroys_power_three_or_higher(self):
        game = new_game()
        weak = put_creature(game, 2, "Drumble")  # power 2
        strong = put_creature(game, 2, "Charette")  # power 4
        card = make_card("The Spirit’s Way", 1)
        run_hook(game, named.the_spirits_way, card)
        self.assertIn(weak, game.players[2].play_area.creatures)
        self.assertNotIn(strong, game.players[2].play_area.creatures)


class TestSanctumArtifacts(unittest.TestCase):
    def test_epic_quest_play_archives_friendly_knights(self):
        game = new_game()
        knight = put_creature(game, 1, "Raiding Knight")
        other = put_creature(game, 1, "Duma the Martyr")
        card = put_artifact(game, 1, "Epic Quest")
        run_hook(game, named.epic_quest_play, card)
        self.assertIn(knight, game.players[1].archive.cards())
        self.assertIn(other, game.players[1].play_area.creatures)

    def test_epic_quest_omni_needs_seven_sanctum_cards_played(self):
        game = new_game()
        quest = put_artifact(game, 1, "Epic Quest")
        # CardsPlayed is only ever populated with cards the player actually
        # owns (in all_cards); extend it here since the test deck (Igor) has
        # no Sanctum cards of its own.
        game.players[1].all_cards.append(make_card("Begone!", 1))
        game.players[1].all_cards.append(make_card("Honorable Claim", 1))
        game.players[1].CardsPlayed = {"Begone!": 3, "Honorable Claim": 3}
        run_hook(game, named.epic_quest_omni, quest)
        self.assertIn(quest, game.players[1].play_area.artifacts)  # not enough yet
        game.players[1].CardsPlayed = {"Begone!": 4, "Honorable Claim": 3}
        run_hook(game, named.epic_quest_omni, quest)
        self.assertNotIn(quest, game.players[1].play_area.artifacts)
        self.assertEqual(game.players[1].keys, 1)

    def test_gorm_of_omm_sacrifices_and_destroys_an_artifact(self):
        game = new_game()
        gorm = put_artifact(game, 1, "Gorm of Omm")
        target = put_artifact(game, 2, "Hallowed Blaster")
        run_hook(game, named.gorm_of_omm, gorm, answers=[[target]])
        self.assertNotIn(gorm, game.players[1].play_area.artifacts)
        self.assertNotIn(target, game.players[2].play_area.artifacts)

    def test_hallowed_blaster_heals_a_chosen_creature(self):
        game = new_game()
        blaster = put_artifact(game, 1, "Hallowed Blaster")
        target = put_creature(game, 1, "Truebaru")
        target.type_object.damage = 5
        run_hook(game, named.hallowed_blaster, blaster, answers=[[target]])
        self.assertEqual(target.type_object.damage, 2)

    def test_potion_of_invulnerability_sacrifices_and_prevents_damage(self):
        game = new_game()
        potion = put_artifact(game, 1, "Potion of Invulnerability")
        mine = put_creature(game, 1, "Drumble")
        run_hook(game, named.potion_of_invulnerability, potion)
        self.assertNotIn(potion, game.players[1].play_area.artifacts)
        steps.deal_damage(game, mine, 5)
        self.assertEqual(mine.type_object.damage, 0)

    def test_round_table_boosts_only_friendly_knights(self):
        game = new_game()
        put_artifact(game, 1, "Round Table")  # put_artifact registers its passive
        knight = put_creature(game, 1, "Raiding Knight")  # power 4
        non_knight = put_creature(game, 1, "Duma the Martyr")
        enemy_knight = put_creature(game, 2, "Sequis")
        self.assertEqual(game.get_power(knight), 5)
        self.assertIn("taunt", game.get_keywords(knight))
        self.assertNotIn("taunt", game.get_keywords(non_knight))
        self.assertNotIn("taunt", game.get_keywords(enemy_knight))

    def test_sigil_of_brotherhood_lets_sanctum_creatures_be_used_off_house(self):
        game = new_game()
        sigil = put_artifact(game, 1, "Sigil of Brotherhood")
        mine = put_creature(game, 1, "Duma the Martyr", can_be_used=False)  # Sanctum, off-house
        run_hook(game, named.sigil_of_brotherhood, sigil)
        self.assertNotIn(sigil, game.players[1].play_area.artifacts)
        self.assertTrue(mine.CanBeUsed)
        self.assertTrue(game._can_use_off_house(mine))

    def test_whispering_reliquary_returns_an_artifact_to_hand(self):
        game = new_game()
        reliquary = put_artifact(game, 1, "Whispering Reliquary")
        target = put_artifact(game, 2, "Gorm of Omm")
        run_hook(game, named.whispering_reliquary, reliquary, answers=[[target]])
        self.assertIn(target, game.players[2].hand.cards())


class TestSanctumCreatures(unittest.TestCase):
    def test_bulwark_grants_armor_to_neighbors_only(self):
        game = new_game()
        left = put_creature(game, 1, "Drumble", flank="left")
        bulwark = put_creature(game, 1, "Bulwark")
        right = put_creature(game, 1, "Truebaru", flank="right")
        self.assertEqual(game.get_armor(left), 2)
        self.assertEqual(game.get_armor(right), 2)
        self.assertEqual(game.get_armor(bulwark), 2)  # its own printed armor, no self-buff

    def test_champion_tabris_captures_after_fight(self):
        game = new_game()
        game.players[2].aember = 3
        tabris = put_creature(game, 1, "Champion Tabris")
        put_creature(game, 2, "Charette")
        drive(game._fight(1, tabris))
        self.assertEqual(tabris.aember_captured, 1)

    def test_commander_remiel_uses_a_friendly_non_sanctum_creature(self):
        game = new_game()
        remiel = put_creature(game, 1, "Commander Remiel")
        other = put_creature(game, 1, "Charette", exhausted=False)  # Dis
        run_hook(game, named.commander_remiel, remiel, answers=[[other]])
        self.assertEqual(game.players[1].aember, 1)  # reaped by default
        self.assertTrue(other.Exhausted)

    def test_duma_the_martyr_heals_others_and_draws_on_death(self):
        game = new_game()
        p1 = game.players[1]
        duma = put_creature(game, 1, "Duma the Martyr")
        other = put_creature(game, 1, "Truebaru")
        other.type_object.damage = 3
        p1.deck.put_on_top(make_card("Charette", 1))
        p1.deck.put_on_top(make_card("Bumpsy", 1))
        before_hand = len(p1.hand)
        drive(game.destroy_cards([duma]))
        self.assertEqual(other.type_object.damage, 0)
        self.assertEqual(len(p1.hand), before_hand + 2)

    def test_francus_captures_when_enemy_dies_fighting_it(self):
        game = new_game()
        game.players[2].aember = 3
        francus = put_creature(game, 1, "Francus")
        put_creature(game, 2, "Charette")
        drive(game._fight(1, francus))
        self.assertEqual(francus.aember_captured, 1)

    def test_grey_monk_grants_armor_and_heals_on_reap(self):
        game = new_game()
        monk = put_creature(game, 1, "Grey Monk")
        friendly = put_creature(game, 1, "Drumble")
        enemy = put_creature(game, 2, "Charette")
        self.assertEqual(game.get_armor(friendly), 1)
        self.assertEqual(game.get_armor(enemy), 0)
        friendly.type_object.damage = 2
        run_hook(game, named.grey_monk_after_reap, monk, answers=[[friendly]])
        self.assertEqual(friendly.type_object.damage, 0)

    def test_hayyel_the_merchant_gains_on_artifact_plays_only(self):
        game = new_game()
        put_creature(game, 1, "Hayyel the Merchant")
        artifact_card = make_card("Hallowed Blaster", 1)
        creature_card = make_card("Drumble", 1)
        drive(game._fire_event("card_played", {"player": 1, "card": artifact_card}))
        self.assertEqual(game.players[1].aember, 1)
        drive(game._fire_event("card_played", {"player": 1, "card": creature_card}))
        self.assertEqual(game.players[1].aember, 1)

    def test_horseman_of_death_returns_horsemen_from_discard(self):
        game = new_game()
        p1 = game.players[1]
        horseman = make_card("Horseman of Famine", 1)
        p1.discard.push(horseman)
        non_horseman = make_card("Drumble", 1)
        p1.discard.push(non_horseman)
        death = put_creature(game, 1, "Horseman of Death")
        run_hook(game, named.horseman_of_death, death)
        self.assertIn(horseman, p1.hand.cards())
        self.assertIn(non_horseman, p1.discard.cards())

    def test_horseman_of_famine_destroys_least_powerful(self):
        game = new_game()
        famine = put_creature(game, 1, "Horseman of Famine")
        weakest = put_creature(game, 2, "Drumble")  # power 2
        put_creature(game, 2, "Truebaru")  # power 7
        run_hook(game, named.horseman_of_famine, famine)
        self.assertNotIn(weakest, game.players[2].play_area.creatures)

    def test_horseman_of_pestilence_hits_non_horseman_creatures(self):
        game = new_game()
        pestilence = put_creature(game, 1, "Horseman of Pestilence")
        horseman = put_creature(game, 1, "Horseman of Famine")
        other = put_creature(game, 2, "Truebaru")
        run_hook(game, named.horseman_of_pestilence, pestilence)
        self.assertEqual(horseman.type_object.damage, 0)
        self.assertEqual(other.type_object.damage, 1)

    def test_horseman_of_war_restricts_friendly_creatures_to_fighting_only(self):
        from keyforge.actions import Reap, Fight
        game = new_game()
        war = put_creature(game, 1, "Horseman of War")
        offhouse = put_creature(game, 1, "Charette", can_be_used=False)  # Dis, off-house
        put_creature(game, 2, "Drumble")
        run_hook(game, named.horseman_of_war, war)
        self.assertTrue(offhouse.CanBeUsed)
        actions = game._legal_actions(1)
        self.assertFalse(any(isinstance(a, Reap) and a.card is offhouse for a in actions))
        self.assertTrue(any(isinstance(a, Fight) and a.card is offhouse for a in actions))

    def test_jehu_the_bureaucrat_gains_when_sanctum_chosen(self):
        game = new_game()
        put_creature(game, 1, "Jehu the Bureaucrat")
        drive(game._fire_event("house_chosen", {"player": 1, "house": House.SANCTUM}))
        self.assertEqual(game.players[1].aember, 2)
        drive(game._fire_event("house_chosen", {"player": 1, "house": House.DIS}))
        self.assertEqual(game.players[1].aember, 2)

    def test_lady_maxena_stuns_on_play_and_bounces_via_action(self):
        game = new_game()
        maxena = put_creature(game, 1, "Lady Maxena")
        target = put_creature(game, 2, "Drumble")
        run_hook(game, named.lady_maxena_play, maxena, answers=[[target]])
        self.assertTrue(target.stunned)
        run_hook(game, named.lady_maxena_action, maxena)
        self.assertIn(maxena, game.players[1].hand.cards())

    def test_lord_golgotha_before_fight_damages_targets_neighbors(self):
        game = new_game()
        golgotha = put_creature(game, 1, "Lord Golgotha")
        left = put_creature(game, 2, "Truebaru", flank="left")
        mid = put_creature(game, 2, "Kelifi Dragon")
        right = put_creature(game, 2, "Truebaru", flank="right")
        drive(game._fight(1, golgotha), answers=[[mid]])
        self.assertEqual(left.type_object.damage, 3)
        self.assertEqual(right.type_object.damage, 3)

    def test_numquid_the_fair_repeats_while_opponent_has_more_creatures(self):
        game = new_game()
        numquid = put_creature(game, 1, "Numquid the Fair")
        e1 = put_creature(game, 2, "Drumble")
        e2 = put_creature(game, 2, "Charette")
        e3 = put_creature(game, 2, "Bumpsy")
        run_hook(game, named.numquid_the_fair, numquid, answers=[[e1], [e2]])
        self.assertEqual(len(game.players[2].play_area.creatures), 1)

    def test_protectrix_may_fully_heal_and_prevent_damage(self):
        game = new_game()
        protectrix = put_creature(game, 1, "Protectrix")
        target = put_creature(game, 1, "Truebaru")
        target.type_object.damage = 4
        run_hook(game, named.protectrix_after_reap, protectrix, answers=[True, [target]])
        self.assertEqual(target.type_object.damage, 0)
        steps.deal_damage(game, target, 5)
        self.assertEqual(target.type_object.damage, 0)

    def test_sanctum_guardian_swaps_battleline_positions(self):
        game = new_game()
        left = put_creature(game, 1, "Drumble", flank="left")
        guardian = put_creature(game, 1, "Sanctum Guardian", flank="right")
        self.assertEqual(list(game.players[1].play_area.creatures), [left, guardian])
        run_hook(game, named.sanctum_guardian_after, guardian, answers=[[left]])
        self.assertEqual(list(game.players[1].play_area.creatures), [guardian, left])

    def test_sergeant_zakiel_may_ready_and_fight_a_neighbor(self):
        game = new_game()
        neighbor = put_creature(game, 1, "Drumble", exhausted=True)
        zakiel = put_creature(game, 1, "Sergeant Zakiel", flank="right")
        put_creature(game, 2, "Truebaru")
        run_hook(game, named.sergeant_zakiel_play, zakiel, answers=[True, [neighbor]])
        self.assertTrue(neighbor.Exhausted)

    def test_staunch_knight_gets_bonus_power_only_on_a_flank(self):
        game = new_game()
        knight = put_creature(game, 1, "Staunch Knight")  # alone: a flank by definition
        self.assertEqual(game.get_power(knight), 6)
        put_creature(game, 1, "Drumble", flank="left")
        put_creature(game, 1, "Charette")  # [Drumble, Knight, Charette]: knight now in the center
        self.assertEqual(game.get_power(knight), 4)

    def test_gatekeeper_captures_all_but_five(self):
        game = new_game()
        gatekeeper = put_creature(game, 1, "Gatekeeper")
        game.players[2].aember = 9
        run_hook(game, named.gatekeeper, gatekeeper)
        self.assertEqual(gatekeeper.aember_captured, 4)
        self.assertEqual(game.players[2].aember, 5)

    def test_the_vaultkeeper_prevents_being_stolen_from(self):
        game = new_game()
        put_creature(game, 1, "The Vaultkeeper")
        game.players[1].aember = 5
        steps.steal(game, game.players[1], game.players[2], 3)
        self.assertEqual(game.players[1].aember, 5)

    def test_veemos_lightbringer_destroys_elusive_creatures(self):
        game = new_game()
        put_creature(game, 1, "Veemos Lightbringer")
        elusive = put_creature(game, 2, "Drumble")  # elusive
        non_elusive = put_creature(game, 2, "Charette")
        card = make_card("Veemos Lightbringer", 1)
        run_hook(game, named.veemos_lightbringer, card)
        self.assertNotIn(elusive, game.players[2].play_area.creatures)
        self.assertIn(non_elusive, game.players[2].play_area.creatures)


class TestSanctumUpgrades(unittest.TestCase):
    def test_armageddon_cloak_replaces_destruction_with_full_heal(self):
        game = new_game()
        host = put_creature(game, 1, "Drumble")
        cloak = make_card("Armageddon Cloak", 1)
        cloak.type_object.host = host
        host.type_object.upgrades.append(cloak)
        named.armageddon_cloak_register(game, cloak)
        host.type_object.damage = 2  # lethal for power-2 Drumble
        drive(game.check_destroyed([host]))
        self.assertIn(host, game.players[1].play_area.creatures)
        self.assertEqual(host.type_object.damage, 0)
        self.assertNotIn(cloak, host.type_object.upgrades)
        self.assertIn(cloak, game.players[1].discard.cards())

    def test_armageddon_cloak_grants_hazardous_two(self):
        game = new_game()
        host = put_creature(game, 1, "Drumble")
        cloak = make_card("Armageddon Cloak", 1)
        cloak.type_object.host = host
        host.type_object.upgrades.append(cloak)
        self.assertEqual(game.get_hazardous(host), 2)

    def test_mantle_of_the_zealot_grants_versatile(self):
        game = new_game()
        host = put_creature(game, 1, "Drumble")
        upgrade = make_card("Mantle of the Zealot", 1)
        upgrade.type_object.host = host
        host.type_object.upgrades.append(upgrade)
        self.assertIn("versatile", game.get_keywords(host))

    def test_protect_the_weak_grants_armor_and_taunt(self):
        game = new_game()
        host = put_creature(game, 1, "Drumble")
        upgrade = make_card("Protect the Weak", 1)
        upgrade.type_object.host = host
        host.type_object.upgrades.append(upgrade)
        self.assertEqual(game.get_armor(host), 1)
        self.assertIn("taunt", game.get_keywords(host))

    def test_shoulder_armor_boosts_power_and_armor_only_on_a_flank(self):
        game = new_game()
        host = put_creature(game, 1, "Drumble")  # alone: a flank by definition
        upgrade = make_card("Shoulder Armor", 1)
        upgrade.type_object.host = host
        host.type_object.upgrades.append(upgrade)
        named.shoulder_armor_register(game, upgrade)
        self.assertEqual(game.get_power(host), 4)
        self.assertEqual(game.get_armor(host), 2)
        put_creature(game, 1, "Charette", flank="left")
        put_creature(game, 1, "Bumpsy")  # [Charette, host, Bumpsy]: host now in the center
        self.assertEqual(game.get_power(host), 2)
        self.assertEqual(game.get_armor(host), 0)


if __name__ == "__main__":
    unittest.main()

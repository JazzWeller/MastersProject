"""Tests for the 52 Untamed cards (Code/PHASE_3_PLAN.md Milestone D.4)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.helpers import drive, hand_card, make_card, new_game, put_artifact, put_creature, run_hook

from keyforge.effects import steps
from keyforge.effects.named import untamed as named
from keyforge.enums import House


class TestUntamedActions(unittest.TestCase):
    def test_cooperative_hunting_deals_one_per_friendly_creature(self):
        game = new_game()
        put_creature(game, 1, "Drumble")
        put_creature(game, 1, "Charette")
        target = put_creature(game, 2, "Kelifi Dragon")
        card = make_card("Cooperative Hunting", 1)
        run_hook(game, named.cooperative_hunting, card, answers=[[target], [target]])
        self.assertEqual(target.type_object.damage, 2)

    def test_curiosity_destroys_scientist_creatures(self):
        game = new_game()
        scientist = put_creature(game, 2, "Qyxxlyx Plague Master")  # martian, scientist
        other = put_creature(game, 2, "Charette")
        card = make_card("Curiosity", 1)
        run_hook(game, named.curiosity, card)
        self.assertNotIn(scientist, game.players[2].play_area.creatures)
        self.assertIn(other, game.players[2].play_area.creatures)

    def test_fertility_chant_gives_opponent_aember(self):
        game = new_game()
        card = make_card("Fertility Chant", 1)
        run_hook(game, named.fertility_chant, card)
        self.assertEqual(game.players[2].aember, 2)

    def test_full_moon_gains_on_creature_plays_this_turn(self):
        game = new_game()
        card = make_card("Full Moon", 1)
        run_hook(game, named.full_moon, card)
        new_creature = make_card("Drumble", 1)
        game.players[1].play_area.add_creature(new_creature)
        drive(game._fire_event("card_played", {"player": 1, "card": new_creature}))
        self.assertEqual(game.players[1].aember, 1)

    def test_grasping_vines_returns_up_to_three_artifacts(self):
        game = new_game()
        a1 = put_artifact(game, 1, "Nepenthe Seed")
        a2 = put_artifact(game, 2, "World Tree")
        card = make_card("Grasping Vines", 1)
        run_hook(game, named.grasping_vines, card, answers=[[a1, a2]])
        self.assertIn(a1, game.players[1].hand.cards())
        self.assertIn(a2, game.players[2].hand.cards())

    def test_key_charge_loses_one_then_may_forge(self):
        game = new_game()
        game.players[1].aember = 7  # loses 1 (to 6), then must still afford the unchanged cost-6 forge
        card = make_card("Key Charge", 1)
        run_hook(game, named.key_charge, card, answers=[True])
        self.assertEqual(game.players[1].keys, 1)
        self.assertEqual(game.players[1].aember, 0)

    def test_key_charge_does_nothing_if_no_aember_to_lose(self):
        game = new_game()
        game.players[1].aember = 0
        card = make_card("Key Charge", 1)
        run_hook(game, named.key_charge, card)
        self.assertEqual(game.players[1].keys, 0)

    def test_lifeweb_steals_if_opponent_played_three_creatures_last_turn(self):
        game = new_game()
        for _ in range(3):
            put_creature(game, 2, "Drumble")
            game.log.add("play_card", player=2, card="Drumble", iid=0, type="Creature", turn=game.turn_number - 1)
        game.players[2].aember = 5
        card = make_card("Lifeweb", 1)
        run_hook(game, named.lifeweb, card)
        self.assertEqual(game.players[1].aember, 2)

    def test_lost_in_the_woods_shuffles_chosen_creatures(self):
        game = new_game()
        mine = put_creature(game, 1, "Drumble")
        theirs = put_creature(game, 2, "Charette")
        card = make_card("Lost in the Woods", 1)
        run_hook(game, named.lost_in_the_woods, card, answers=[[mine], [theirs]])
        self.assertNotIn(mine, game.players[1].play_area.creatures)
        self.assertNotIn(theirs, game.players[2].play_area.creatures)

    def test_mimicry_copies_an_opponents_discarded_action(self):
        game = new_game()
        p2 = game.players[2]
        copied = make_card("Punch", 2)  # Brobnar action: deal 3 damage to a creature
        p2.discard.push(copied)
        target = put_creature(game, 2, "Kelifi Dragon")
        card = make_card("Mimicry", 1)
        run_hook(game, named.mimicry_play, card, answers=[[copied], [target]])
        self.assertEqual(target.type_object.damage, 3)
        self.assertEqual(card.house_override, House.BROBNAR)

    def test_mimicry_cannot_copy_another_mimicry(self):
        game = new_game()
        p2 = game.players[2]
        other_mimicry = make_card("Mimicry", 2)
        p2.discard.push(other_mimicry)
        real_action = make_card("Punch", 2)
        p2.discard.push(real_action)
        card = make_card("Mimicry", 1)
        run_hook(game, named.mimicry_play, card)  # must not infinitely recurse
        options_seen = [
            c for c in p2.discard.cards() if c.type.value == "Action" and c.name != "Mimicry"
        ]
        self.assertEqual(options_seen, [real_action])

    def test_natures_call_returns_up_to_three_creatures(self):
        game = new_game()
        mine = put_creature(game, 1, "Drumble")
        theirs = put_creature(game, 2, "Charette")
        card = make_card("Nature’s Call", 1)
        run_hook(game, named.natures_call, card, answers=[[mine, theirs]])
        self.assertIn(mine, game.players[1].hand.cards())
        self.assertIn(theirs, game.players[2].hand.cards())

    def test_nocturnal_maneuver_exhausts_up_to_three(self):
        game = new_game()
        a = put_creature(game, 2, "Drumble")
        b = put_creature(game, 1, "Charette")
        card = make_card("Nocturnal Maneuver", 1)
        run_hook(game, named.nocturnal_maneuver, card, answers=[[a, b]])
        self.assertTrue(a.Exhausted)
        self.assertTrue(b.Exhausted)

    def test_perilous_wild_destroys_elusive_creatures(self):
        game = new_game()
        elusive = put_creature(game, 2, "Drumble")
        other = put_creature(game, 2, "Charette")
        card = make_card("Perilous Wild", 1)
        run_hook(game, named.perilous_wild, card)
        self.assertNotIn(elusive, game.players[2].play_area.creatures)
        self.assertIn(other, game.players[2].play_area.creatures)

    def test_regrowth_returns_a_creature_from_discard(self):
        game = new_game()
        p1 = game.players[1]
        creature = make_card("Drumble", 1)
        p1.discard.push(creature)
        card = make_card("Regrowth", 1)
        run_hook(game, named.regrowth, card, answers=[[creature]])
        self.assertIn(creature, p1.hand.cards())

    def test_save_the_pack_destroys_damaged_and_gains_chain(self):
        game = new_game()
        damaged = put_creature(game, 2, "Charette")
        damaged.type_object.damage = 1
        undamaged = put_creature(game, 1, "Drumble")
        card = make_card("Save the Pack", 1)
        run_hook(game, named.save_the_pack, card)
        self.assertNotIn(damaged, game.players[2].play_area.creatures)
        self.assertIn(undamaged, game.players[1].play_area.creatures)
        self.assertEqual(game.players[1].chains, 1)

    def test_scout_grants_skirmish_and_fights_chosen_creatures(self):
        game = new_game()
        fighter = put_creature(game, 1, "Drumble", exhausted=True)
        put_creature(game, 2, "Truebaru")
        card = make_card("Scout", 1)
        run_hook(game, named.scout, card, answers=[[fighter]])
        self.assertIn("skirmish", game.get_keywords(fighter))
        self.assertTrue(fighter.Exhausted)  # readied then fought

    def test_stampede_steals_after_three_uses(self):
        game = new_game()
        game.players[2].aember = 5
        game.players[1].used_this_turn = {"A": 1, "B": 1, "C": 1}
        card = make_card("Stampede", 1)
        run_hook(game, named.stampede, card)
        self.assertEqual(game.players[1].aember, 2)

    def test_the_common_cold_damages_all_and_may_destroy_mars(self):
        game = new_game()
        mars_creature = put_creature(game, 2, "Grabber Jammer")
        other = put_creature(game, 1, "Drumble")
        card = make_card("The Common Cold", 1)
        run_hook(game, named.the_common_cold, card, answers=[True])
        self.assertNotIn(mars_creature, game.players[2].play_area.creatures)
        self.assertEqual(other.type_object.damage, 1)

    def test_troop_call_returns_niffle_creatures_from_discard_and_play(self):
        game = new_game()
        p1 = game.players[1]
        in_discard = make_card("Niffle Ape", 1)
        p1.discard.push(in_discard)
        in_play = put_creature(game, 1, "Niffle Queen")
        card = make_card("Troop Call", 1)
        run_hook(game, named.troop_call, card)
        self.assertIn(in_discard, p1.hand.cards())
        self.assertIn(in_play, p1.hand.cards())

    def test_vigor_heals_and_gains_only_on_full_heal(self):
        game = new_game()
        partial = put_creature(game, 1, "Truebaru")
        partial.type_object.damage = 2
        card = make_card("Vigor", 1)
        run_hook(game, named.vigor, card, answers=[[partial]])
        self.assertEqual(partial.type_object.damage, 0)
        self.assertEqual(game.players[1].aember, 0)  # only healed 2, not 3
        full = put_creature(game, 1, "Kelifi Dragon")
        full.type_object.damage = 5
        card2 = make_card("Vigor", 1)
        run_hook(game, named.vigor, card2, answers=[[full]])
        self.assertEqual(full.type_object.damage, 2)
        self.assertEqual(game.players[1].aember, 1)

    def test_word_of_returning_damages_and_returns_captured_aember(self):
        game = new_game()
        target = put_creature(game, 2, "Kelifi Dragon")
        target.aember_captured = 3
        card = make_card("Word of Returning", 1)
        run_hook(game, named.word_of_returning, card)
        self.assertEqual(target.type_object.damage, 3)
        self.assertEqual(target.aember_captured, 0)
        self.assertEqual(game.players[1].aember, 3)


class TestUntamedArtifacts(unittest.TestCase):
    def test_bear_flute_heals_an_ancient_bear_in_play(self):
        game = new_game()
        flute = put_artifact(game, 1, "Bear Flute")
        bear = put_creature(game, 1, "Ancient Bear")
        bear.type_object.damage = 3
        run_hook(game, named.bear_flute, flute)
        self.assertEqual(bear.type_object.damage, 0)

    def test_bear_flute_finds_a_bear_in_deck_or_discard(self):
        game = new_game()
        flute = put_artifact(game, 1, "Bear Flute")
        p1 = game.players[1]
        bear = make_card("Ancient Bear", 1)
        p1.deck.put_on_top(bear)
        run_hook(game, named.bear_flute, flute)
        self.assertIn(bear, p1.hand.cards())

    def test_nepenthe_seed_sacrifices_and_returns_a_card(self):
        game = new_game()
        seed = put_artifact(game, 1, "Nepenthe Seed")
        p1 = game.players[1]
        discarded = make_card("Drumble", 1)
        p1.discard.push(discarded)
        run_hook(game, named.nepenthe_seed, seed, answers=[[discarded]])
        self.assertNotIn(seed, p1.play_area.artifacts)
        self.assertIn(discarded, p1.hand.cards())

    def test_ritual_of_balance_steals_only_at_six_or_more(self):
        game = new_game()
        ritual = put_artifact(game, 1, "Ritual of Balance")
        game.players[2].aember = 5
        run_hook(game, named.ritual_of_balance, ritual)
        self.assertEqual(game.players[1].aember, 0)
        game.players[2].aember = 6
        run_hook(game, named.ritual_of_balance, ritual)
        self.assertEqual(game.players[1].aember, 1)

    def test_ritual_of_the_hunt_lets_untamed_creatures_be_used_off_house(self):
        game = new_game()
        ritual = put_artifact(game, 1, "Ritual of the Hunt")
        mine = put_creature(game, 1, "Halacor", can_be_used=False)  # Untamed
        run_hook(game, named.ritual_of_the_hunt, ritual)
        self.assertNotIn(ritual, game.players[1].play_area.artifacts)
        self.assertTrue(mine.CanBeUsed)

    def test_world_tree_returns_a_creature_to_deck_top(self):
        game = new_game()
        tree = put_artifact(game, 1, "World Tree")
        p1 = game.players[1]
        creature = make_card("Drumble", 1)
        p1.discard.push(creature)
        run_hook(game, named.world_tree, tree, answers=[[creature]])
        self.assertIs(p1.deck.cards()[0], creature)


class TestUntamedCreatures(unittest.TestCase):
    def test_ancient_bear_deals_assault_damage_before_the_fight(self):
        game = new_game()
        bear = put_creature(game, 1, "Ancient Bear")
        target = put_creature(game, 2, "Kelifi Dragon")
        drive(game._fight(1, bear))
        self.assertEqual(target.type_object.damage, 2 + 5)  # 2 assault + 5 fight power

    def test_bigtwig_can_only_fight_stunned_creatures(self):
        game = new_game()
        bigtwig = put_creature(game, 1, "Bigtwig")
        unstunned = put_creature(game, 2, "Charette")
        self.assertEqual(game.legal_fight_targets(bigtwig), [])
        unstunned.stunned = True
        self.assertEqual(game.legal_fight_targets(bigtwig), [unstunned])

    def test_bigtwig_after_reap_stuns_and_exhausts(self):
        game = new_game()
        bigtwig = put_creature(game, 1, "Bigtwig")
        target = put_creature(game, 2, "Drumble")
        run_hook(game, named.bigtwig_after_reap, bigtwig, answers=[[target]])
        self.assertTrue(target.stunned)
        self.assertTrue(target.Exhausted)

    def test_witch_of_the_wilds_grants_an_off_house_play_when_not_untamed(self):
        game = new_game()
        put_creature(game, 1, "Witch of the Wilds")
        drive(game._fire_event("house_chosen", {"player": 1, "house": House.DIS}))
        self.assertEqual(game.players[1].NonLogosCardsPlayable, 1)
        drive(game._fire_event("house_chosen", {"player": 1, "house": House.UNTAMED}))
        self.assertEqual(game.players[1].NonLogosCardsPlayable, 1)  # unchanged

    def test_briar_grubbling_has_printed_hazardous(self):
        game = new_game()
        grubbling = put_creature(game, 2, "Briar Grubbling")  # power 2
        attacker = put_creature(game, 1, "Grommid")  # power 10, no on_fight side effects; survives 5+2=7
        drive(game._fight(1, attacker))
        self.assertEqual(attacker.type_object.damage, 7)  # 5 hazardous + 2 from Briar Grubbling's own power

    def test_chota_hazri_loses_one_then_may_forge(self):
        game = new_game()
        game.players[1].aember = 7  # loses 1 (to 6), then must still afford the unchanged cost-6 forge
        hazri = put_creature(game, 1, "Chota Hazri")
        run_hook(game, named.chota_hazri, hazri, answers=[True])
        self.assertEqual(game.players[1].keys, 1)

    def test_flaxia_gains_with_more_creatures(self):
        game = new_game()
        put_creature(game, 1, "Drumble")
        flaxia = put_creature(game, 1, "Flaxia")
        put_creature(game, 2, "Charette")
        run_hook(game, named.flaxia, flaxia)
        self.assertEqual(game.players[1].aember, 2)

    def test_giant_sloth_needs_an_untamed_discard_this_turn(self):
        game = new_game()
        sloth = put_creature(game, 1, "Giant Sloth")
        self.assertFalse(sloth.card_def.use_restriction(game, sloth))
        untamed_card = hand_card(game, 1, "Curiosity")
        drive(steps.discard_from_hand(game, game.players[1], untamed_card))
        self.assertTrue(sloth.card_def.use_restriction(game, sloth))

    def test_halacor_grants_skirmish_to_flank_creatures_only(self):
        game = new_game()
        left = put_creature(game, 1, "Drumble")
        put_creature(game, 1, "Halacor")
        right = put_creature(game, 1, "Charette")
        self.assertIn("skirmish", game.get_keywords(left))
        self.assertIn("skirmish", game.get_keywords(right))

    def test_inka_the_spider_has_poison_and_stuns(self):
        game = new_game()
        spider = put_creature(game, 1, "Inka the Spider")
        self.assertIn("poison", game.get_keywords(spider))
        target = put_creature(game, 2, "Drumble")
        run_hook(game, named.inka_the_spider_effect, spider, answers=[[target]])
        self.assertTrue(target.stunned)

    def test_kindrith_longshot_deals_damage_after_reap(self):
        game = new_game()
        kindrith = put_creature(game, 1, "Kindrith Longshot")
        target = put_creature(game, 2, "Kelifi Dragon")  # power 12, survives 2 damage
        run_hook(game, named.kindrith_longshot_after_reap, kindrith, answers=[[target]])
        self.assertEqual(target.type_object.damage, 2)

    def test_lupo_the_scarred_play_damages_an_enemy(self):
        game = new_game()
        lupo = put_creature(game, 1, "Lupo the Scarred")
        target = put_creature(game, 2, "Kelifi Dragon")  # power 12, survives 2 damage
        run_hook(game, named.lupo_the_scarred_play, lupo, answers=[[target]])
        self.assertEqual(target.type_object.damage, 2)

    def test_murmook_raises_opponent_key_cost(self):
        game = new_game()
        put_creature(game, 1, "Murmook")
        self.assertEqual(game.players[2].get_key_forge_cost(game), 7)

    def test_mushroom_man_scales_with_unforged_keys(self):
        game = new_game()
        mushroom = put_creature(game, 1, "Mushroom Man")
        self.assertEqual(game.get_power(mushroom), 2 + 9)  # 3 unforged keys
        game.players[1].keys = 2
        self.assertEqual(game.get_power(mushroom), 2 + 3)  # 1 unforged key

    def test_niffle_ape_ignores_taunt_and_elusive_while_attacking(self):
        game = new_game()
        ape = put_creature(game, 1, "Niffle Ape")  # power 3
        drumble = put_creature(game, 2, "Drumble")  # elusive, power 2
        put_creature(game, 2, "Sanctum Guardian")  # taunt, neighbors Drumble -- normally protects it
        targets = game.legal_fight_targets(ape)
        self.assertIn(drumble, targets)  # taunt-protection ignored while Niffle Ape attacks
        drive(game._fight(1, ape), answers=[[drumble]])
        self.assertNotIn(drumble, game.players[2].play_area.creatures)  # elusive ignored: actually destroyed

    def test_niffle_queen_boosts_beast_and_niffle_creatures(self):
        game = new_game()
        put_creature(game, 1, "Niffle Queen")
        beast_and_niffle = put_creature(game, 1, "Niffle Ape")  # beast, niffle
        neither = put_creature(game, 1, "Drumble")
        self.assertEqual(game.get_power(beast_and_niffle), 3 + 2)
        self.assertEqual(game.get_power(neither), 2)

    def test_piranha_monkeys_damages_every_other_creature(self):
        game = new_game()
        monkeys = put_creature(game, 1, "Piranha Monkeys")
        mine = put_creature(game, 1, "Truebaru")
        theirs = put_creature(game, 2, "Kelifi Dragon")
        run_hook(game, named.piranha_monkeys_effect, monkeys)
        self.assertEqual(mine.type_object.damage, 2)
        self.assertEqual(theirs.type_object.damage, 2)
        self.assertEqual(monkeys.type_object.damage, 0)

    def test_teliga_gains_only_on_opponent_creature_plays(self):
        game = new_game()
        put_creature(game, 1, "Teliga")
        enemy_creature = make_card("Charette", 2)
        game.players[2].play_area.add_creature(enemy_creature)
        drive(game._fire_event("any_card_played", {"player": 2, "card": enemy_creature}))
        self.assertEqual(game.players[1].aember, 1)
        own_creature = make_card("Drumble", 1)
        game.players[1].play_area.add_creature(own_creature)
        drive(game._fire_event("any_card_played", {"player": 1, "card": own_creature}))
        self.assertEqual(game.players[1].aember, 1)  # unchanged

    def test_hunting_witch_gains_on_own_other_creature_plays(self):
        game = new_game()
        put_creature(game, 1, "Hunting Witch")
        new_creature = make_card("Drumble", 1)
        game.players[1].play_area.add_creature(new_creature)
        drive(game._fire_event("card_played", {"player": 1, "card": new_creature}))
        self.assertEqual(game.players[1].aember, 1)

    def test_witch_of_the_eye_returns_a_card_from_discard(self):
        game = new_game()
        witch = put_creature(game, 1, "Witch of the Eye")
        p1 = game.players[1]
        discarded = make_card("Drumble", 1)
        p1.discard.push(discarded)
        run_hook(game, named.witch_of_the_eye_after_reap, witch, answers=[[discarded]])
        self.assertIn(discarded, p1.hand.cards())


class TestUntamedUpgrades(unittest.TestCase):
    def test_way_of_the_bear_grants_assault(self):
        game = new_game()
        host = put_creature(game, 1, "Drumble")
        upgrade = make_card("Way of the Bear", 1)
        upgrade.type_object.host = host
        host.type_object.upgrades.append(upgrade)
        self.assertEqual(game.get_assault(host), 2)

    def test_way_of_the_wolf_grants_skirmish(self):
        game = new_game()
        host = put_creature(game, 1, "Drumble")
        upgrade = make_card("Way of the Wolf", 1)
        upgrade.type_object.host = host
        host.type_object.upgrades.append(upgrade)
        self.assertIn("skirmish", game.get_keywords(host))


if __name__ == "__main__":
    unittest.main()

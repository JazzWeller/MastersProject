"""Phase 3 rules generalizations exercised across houses (Code/PHASE_3_PLAN.md
Verification item 1): armor sources, prevention, assault, ready-and-fight,
dynamic key cost, capture from own side, steal immunity, Mimicry, and the
marquee interactions Milestone G names (armor + Red-Hot Armor, taunt walls,
Key Charge, Ether Spider, Armageddon Cloak). The per-card files test each
card alone; these combine cards from different houses through the real
fight/damage/forge pipeline."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.helpers import drive, make_card, new_game, put_artifact, put_creature, run_hook

from keyforge.effects.effect_object import resolve_cleanup
from keyforge.effects.named import brobnar, dis, mars, sanctum, shadows, untamed
from keyforge.enums import DecisionKind


def _attach(game, host, name, register=None):
    upgrade = make_card(name, host.controller)
    upgrade.type_object.host = host
    host.type_object.upgrades.append(upgrade)
    if register is not None:
        register(game, upgrade)
    return upgrade


class TestArmorSources(unittest.TestCase):
    def test_printed_neighbor_and_global_armor_stack(self):
        game = new_game()
        put_creature(game, 1, "Grey Monk")  # each friendly creature +1
        bulwark = put_creature(game, 1, "Bulwark")  # printed 2, neighbors +2
        sequis = put_creature(game, 1, "Sequis")  # printed 2, Bulwark's neighbor
        self.assertEqual(game.get_armor(sequis), 2 + 2 + 1)
        self.assertEqual(game.get_armor(bulwark), 2 + 1)  # not its own neighbor

    def test_red_hot_armor_strips_the_computed_armor_and_deals_that_much(self):
        game = new_game()
        monk = put_creature(game, 2, "Grey Monk")
        bulwark = put_creature(game, 2, "Bulwark")
        sequis = put_creature(game, 2, "Sequis")  # 5 armor, 4 power
        troll = put_creature(game, 2, "Troll")  # 0 printed + 1 Grey Monk
        run_hook(game, dis.red_hot_armor, make_card("Red-Hot Armor", 1))
        p2 = game.players[2].play_area.creatures
        self.assertNotIn(sequis, p2)  # 5 damage >= 4 power
        self.assertNotIn(monk, p2)  # 1 own + 2 from Bulwark = 3 damage >= 3 power
        self.assertEqual(bulwark.type_object.damage, 3)  # 2 printed + 1 Grey Monk; survives at power 4
        self.assertEqual(troll.type_object.damage, 1)
        self.assertEqual(game.get_armor(troll), 0)
        for operation, iid in game._end_of_turn_cleanups:
            resolve_cleanup(game, operation, iid)
        # Armor returns, recomputed from the new board: Grey Monk is gone,
        # and with Sequis gone Troll is now Bulwark's neighbor.
        self.assertEqual(game.get_armor(troll), 2)

    def test_armor_is_a_per_turn_budget_shared_by_assault_and_fight_damage(self):
        game = new_game()
        bear = put_creature(game, 1, "Ancient Bear")  # power 5, assault 2
        troll = put_creature(game, 2, "Troll")  # power 8
        _attach(game, troll, "Protect the Weak")  # +1 armor
        drive(game._fight(1, bear))
        # Assault 2: 1 absorbed, 1 dealt. Fight 5: armor already spent, 5 dealt.
        self.assertEqual(troll.type_object.damage, 6)
        self.assertIn(troll, game.players[2].play_area.creatures)


class TestPreventionAndKeywords(unittest.TestCase):
    def test_shield_of_justice_prevents_fight_damage_to_a_friendly_attacker(self):
        game = new_game()
        imp = put_creature(game, 1, "Dust Imp")  # Dis, power 2
        troll = put_creature(game, 2, "Troll")
        shield = make_card("Shield of Justice", 1)
        run_hook(game, shield.card_def.on_play, shield)
        drive(game._fight(1, imp))
        self.assertIn(imp, game.players[1].play_area.creatures)
        self.assertEqual(imp.type_object.damage, 0)
        self.assertEqual(troll.type_object.damage, 2)

    def test_protect_the_weak_turns_an_off_house_creature_into_a_taunt_wall(self):
        game = new_game()
        attacker = put_creature(game, 1, "Troll")
        left = put_creature(game, 2, "Charette")
        middle = put_creature(game, 2, "Dust Imp")
        right = put_creature(game, 2, "Doc Bookton")
        _attach(game, middle, "Protect the Weak")
        self.assertEqual(game.legal_fight_targets(attacker), [middle])
        self.assertNotIn(left, game.legal_fight_targets(attacker))
        self.assertNotIn(right, game.legal_fight_targets(attacker))

    def test_halacor_gives_a_friendly_flank_creature_skirmish(self):
        game = new_game()
        put_creature(game, 1, "Halacor")
        flank = put_creature(game, 1, "Charette")  # Dis, power 4, right flank
        enemy = put_creature(game, 2, "Troll")
        drive(game._fight(1, flank))
        self.assertEqual(flank.type_object.damage, 0)
        self.assertEqual(enemy.type_object.damage, 4)

    def test_mushroom_man_power_tracks_unforged_keys(self):
        game = new_game()
        shroom = put_creature(game, 1, "Mushroom Man")
        self.assertEqual(game.get_power(shroom), 2 + 3 * 3)
        game.players[1].keys = 2
        self.assertEqual(game.get_power(shroom), 2 + 3 * 1)


class TestReadyAndFight(unittest.TestCase):
    def test_anger_on_a_stunned_creature_only_removes_the_stun(self):
        # MRB FAQ (PHASE_3_CARD_RULINGS.md, Anger): using a stunned creature
        # removes the stun instead of doing anything else.
        game = new_game()
        troll = put_creature(game, 1, "Troll", exhausted=True)
        troll.stunned = True
        enemy = put_creature(game, 2, "Charette")
        run_hook(game, brobnar.anger, make_card("Anger", 1), answers=[[troll]])
        self.assertFalse(troll.stunned)
        self.assertEqual(enemy.type_object.damage, 0)

    def test_anger_readies_an_exhausted_off_house_creature_and_fights(self):
        game = new_game()
        charette = put_creature(game, 1, "Charette", exhausted=True)  # Dis
        enemy = put_creature(game, 2, "Dust Imp")  # power 2, not elusive
        run_hook(game, brobnar.anger, make_card("Anger", 1), answers=[[charette], [enemy]])
        self.assertNotIn(enemy, game.players[2].play_area.creatures)
        self.assertTrue(charette.Exhausted)


class TestKeyCost(unittest.TestCase):
    def test_key_charge_forges_at_a_cost_raised_by_the_enemy_iron_obelisk(self):
        game = new_game()
        put_artifact(game, 2, "Iron Obelisk")
        brute = put_creature(game, 2, "Krump")
        brute.type_object.damage = 1  # one damaged friendly Brobnar creature
        self.assertEqual(game.players[1].get_key_forge_cost(game), 7)

        game.players[1].aember = 7  # 6 after Key Charge's loss: can't pay 7
        run_hook(game, untamed.key_charge, make_card("Key Charge", 1), answers=[True])
        self.assertEqual(game.players[1].keys, 0)

        game.players[1].aember = 8  # 7 after the loss: exactly enough
        run_hook(game, untamed.key_charge, make_card("Key Charge", 1), answers=[True])
        self.assertEqual(game.players[1].keys, 1)
        self.assertEqual(game.players[1].aember, 0)

    def test_iron_obelisk_cost_is_recomputed_when_the_damage_is_healed(self):
        game = new_game()
        put_artifact(game, 2, "Iron Obelisk")
        brute = put_creature(game, 2, "Krump")
        brute.type_object.damage = 2
        self.assertEqual(game.players[1].get_key_forge_cost(game), 7)
        brute.type_object.damage = 0
        self.assertEqual(game.players[1].get_key_forge_cost(game), 6)


class TestAemberFlow(unittest.TestCase):
    def test_ether_spider_captures_an_off_house_gain_by_the_opponent(self):
        game = new_game()
        spider = put_creature(game, 1, "Ether Spider")
        mars.ether_spider_register(game, spider)
        run_hook(game, sanctum.begone, make_card("Begone!", 2), answers=["Gain 1Æ"])
        self.assertEqual(game.players[2].aember, 0)
        self.assertEqual(spider.aember_captured, 1)

    def test_the_vaultkeeper_stops_a_shadows_steal(self):
        game = new_game()
        put_creature(game, 2, "The Vaultkeeper")
        game.players[2].aember = 3
        run_hook(game, shadows.urchin_play, make_card("Urchin", 1))
        self.assertEqual(game.players[2].aember, 3)
        self.assertEqual(game.players[1].aember, 0)

    def test_hypnotic_command_captures_from_each_targets_own_side(self):
        game = new_game()
        put_creature(game, 1, "Zorg")
        put_creature(game, 1, "Tunk")  # two friendly Mars creatures
        target = put_creature(game, 2, "Troll")
        game.players[2].aember = 5
        run_hook(game, mars.hypnotic_command, make_card("Hypnotic Command", 1), answers=[[target], [target]])
        self.assertEqual(target.aember_captured, 2)
        self.assertEqual(game.players[2].aember, 3)
        self.assertEqual(game.players[1].aember, 0)


class TestCopiesAndReplacements(unittest.TestCase):
    def test_mimicry_turns_the_opponents_red_hot_armor_against_them(self):
        game = new_game()
        discarded = make_card("Red-Hot Armor", 2)
        game.players[2].discard.push(discarded)
        sequis = put_creature(game, 2, "Sequis")  # armor 2, power 4
        mimicry = make_card("Mimicry", 1)
        run_hook(game, untamed.mimicry_play, mimicry, answers=[[discarded]])
        self.assertEqual(sequis.type_object.damage, 2)
        self.assertEqual(game.get_armor(sequis), 0)

    def test_armageddon_cloak_survives_gateway_to_dis_then_is_gone(self):
        game = new_game()
        host = put_creature(game, 1, "Troll")
        cloak = _attach(game, host, "Armageddon Cloak", sanctum.armageddon_cloak_register)
        other = put_creature(game, 2, "Charette")
        run_hook(game, dis.gateway_to_dis, make_card("Gateway to Dis", 1))
        self.assertIn(host, game.players[1].play_area.creatures)
        self.assertNotIn(other, game.players[2].play_area.creatures)
        self.assertIn(cloak, game.players[1].discard.cards())
        # The replacement went with the cloak: a second wipe destroys the host.
        run_hook(game, dis.gateway_to_dis, make_card("Gateway to Dis", 1))
        self.assertNotIn(host, game.players[1].play_area.creatures)

    def test_armageddon_cloak_hazardous_hits_an_attacker_before_the_fight(self):
        game = new_game()
        host = put_creature(game, 2, "Troll")
        _attach(game, host, "Armageddon Cloak", sanctum.armageddon_cloak_register)
        attacker = put_creature(game, 1, "Dust Imp")  # power 2: dies to hazardous 2
        drive(game._fight(1, attacker))
        self.assertNotIn(attacker, game.players[1].play_area.creatures)
        self.assertEqual(host.type_object.damage, 0)  # died before dealing damage


class TestPlayOrdering(unittest.TestCase):
    """Found in the Phase 3 GUI playtest: playing Tunk asked "choose what
    resolves first" between a Play effect it doesn't have and its own
    freshly registered passive."""

    def _play(self, name, others=()):
        """Plays `name` for player 1 through the real play path, with
        `others` already in play, and returns the kinds of every decision
        the play asked for."""
        game = new_game()
        for other in others:
            put_creature(game, 1, other)
        card = make_card(name, 1)
        game.players[1].hand.add(card)
        game._cards_by_id[card.instance_id] = card
        kinds = []
        gen = game._play_card(1, card)
        try:
            d = next(gen)
            while True:
                kinds.append(d.kind)
                if d.kind == DecisionKind.ORDER_EFFECTS:
                    choice = list(d.options)
                elif d.kind == DecisionKind.CHOOSE_CARDS:
                    choice = list(d.options[: max(d.min_n, 1)])
                else:
                    choice = d.options[0]
                d = gen.send(choice)
        except StopIteration:
            pass
        return kinds

    def test_a_creature_with_no_play_effect_asks_nothing_about_order(self):
        # Tunk registers its own card_played trigger as it enters play.
        self.assertNotIn(DecisionKind.ORDER_EFFECTS, self._play("Tunk"))

    def test_another_cards_trigger_still_competes_with_a_play_effect(self):
        kinds = self._play("Yxili Marauder", others=["Tunk"])  # Play: capture ...
        self.assertIn(DecisionKind.ORDER_EFFECTS, kinds)

if __name__ == "__main__":
    unittest.main()

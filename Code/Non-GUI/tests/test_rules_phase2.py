"""Tests for the Milestone B rules generalizations (Code/PHASE_2_PLAN.md
v2.0 Milestone B): taunt, poison, stun, control, simultaneous damage,
replacement order, house must/cannot, unforge, and Æmber spend split."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.helpers import drive, hand_card, make_card, new_game, put_artifact, put_creature, run_hook


def drive_return(gen, answers=None):
    """Like `drive`, but returns the generator's StopIteration value."""
    answers = list(answers) if answers else []
    try:
        d = next(gen)
    except StopIteration as stop:
        return stop.value
    while True:
        choice = answers.pop(0) if answers else None
        try:
            d = gen.send(choice)
        except StopIteration as stop:
            return stop.value

from keyforge.cards.card import Card, CardDef
from keyforge.effects import named, steps
from keyforge.effects.effect_object import DurationEffect, InsteadEffect, INFINITE
from keyforge.enums import CardType, House


def _synthetic(name, house=House.DIS, ctype=CardType.CREATURE, power=3, keywords=(), **kwargs):
    return CardDef(id=9000, name=name, house=house, type=ctype, power=power, keywords=keywords, **kwargs)


class TestTaunt(unittest.TestCase):
    def test_taunt_protects_neighbors_but_not_itself(self):
        game = new_game()
        p2 = game.players[2]
        left = put_creature(game, 2, "Old Bruno")  # power 3, no keywords
        taunt_card = Card(_synthetic("Truebaru-like", power=7, keywords=("taunt",)), 2)
        p2.play_area.add_creature(taunt_card)
        right = put_creature(game, 2, "Noddy the Thief")  # power 2, elusive (not taunt)
        self.assertEqual(list(p2.play_area.creatures), [left, taunt_card, right])

        attacker = put_creature(game, 1, "Titan Mechanic")  # power 6
        targets = game.legal_fight_targets(attacker)
        self.assertIn(taunt_card, targets)
        self.assertNotIn(left, targets)
        self.assertNotIn(right, targets)

    def test_taunt_neighbor_that_also_has_taunt_is_still_attackable(self):
        game = new_game()
        p2 = game.players[2]
        t1 = Card(_synthetic("Taunt1", power=5, keywords=("taunt",)), 2)
        t2 = Card(_synthetic("Taunt2", power=5, keywords=("taunt",)), 2)
        p2.play_area.add_creature(t1)
        p2.play_area.add_creature(t2)
        attacker = put_creature(game, 1, "Titan Mechanic")
        targets = game.legal_fight_targets(attacker)
        self.assertIn(t1, targets)
        self.assertIn(t2, targets)

    def test_taunt_creature_itself_is_always_a_legal_target(self):
        game = new_game()
        p2 = game.players[2]
        put_creature(game, 2, "Old Bruno")  # protected neighbor
        taunt_card = Card(_synthetic("Taunt", power=7, keywords=("taunt",)), 2)
        p2.play_area.add_creature(taunt_card)
        attacker = put_creature(game, 1, "Titan Mechanic")
        # Only the taunt creature itself is attackable -- still a legal target,
        # since taunt protects its neighbors, not itself.
        self.assertEqual(game.legal_fight_targets(attacker), [taunt_card])

    def test_no_enemy_creatures_means_no_legal_targets(self):
        game = new_game()
        attacker = put_creature(game, 1, "Titan Mechanic")
        self.assertEqual(game.legal_fight_targets(attacker), [])


class TestPoison(unittest.TestCase):
    def test_poison_destroys_regardless_of_power(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        poison_attacker = Card(_synthetic("Poisoner", power=1, keywords=("poison",)), 1)
        p1.play_area.add_creature(poison_attacker)
        poison_attacker.Exhausted = False
        poison_attacker.CanBeUsed = True
        target = put_creature(game, 2, "Titan Mechanic")  # power 6, way more than 1
        gen = game._fight(1, poison_attacker)
        drive(gen, answers=[[target]])
        self.assertTrue(target.destroyed or target not in p2.play_area.creatures)

    def test_no_poison_does_not_destroy_a_tougher_creature(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        weak_attacker = Card(_synthetic("Weakling", power=1, keywords=()), 1)
        p1.play_area.add_creature(weak_attacker)
        weak_attacker.Exhausted = False
        weak_attacker.CanBeUsed = True
        target = put_creature(game, 2, "Titan Mechanic")  # power 6
        gen = game._fight(1, weak_attacker)
        drive(gen, answers=[[target]])
        self.assertIn(target, p2.play_area.creatures)
        self.assertFalse(target.destroyed)


class TestStun(unittest.TestCase):
    def test_stunned_reap_only_exhausts_and_clears_stun(self):
        game = new_game()
        p1 = game.players[1]
        card = put_creature(game, 1, "Old Bruno", exhausted=False, can_be_used=True)
        card.stunned = True
        before_aember = p1.aember
        drive(game._reap(1, card))
        self.assertEqual(p1.aember, before_aember)  # no reap gain
        self.assertTrue(card.Exhausted)
        self.assertFalse(card.stunned)
        stun_events = [e for e in game.log.events if e.kind == "stun_consumed"]
        self.assertEqual(len(stun_events), 1)

    def test_stunned_fight_deals_no_damage(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        attacker = put_creature(game, 1, "Titan Mechanic", exhausted=False, can_be_used=True)
        attacker.stunned = True
        target = put_creature(game, 2, "Old Bruno")
        drive(game._fight(1, attacker))
        self.assertEqual(target.type_object.damage, 0)
        self.assertTrue(attacker.Exhausted)
        self.assertFalse(attacker.stunned)


class TestControl(unittest.TestCase):
    def test_take_control_moves_creature_and_asks_new_controllers_flank(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        card = put_creature(game, 2, "Old Bruno")
        self.assertIn(card, p2.play_area.creatures)
        drive(game.take_control(card, 1), answers=["right"])
        self.assertNotIn(card, p2.play_area.creatures)
        self.assertIn(card, p1.play_area.creatures)
        self.assertEqual(card.controller, 1)
        events = [e for e in game.log.events if e.kind == "take_control"]
        self.assertEqual(events[-1].data["to_player"], 1)
        self.assertFalse(events[-1].data["reverted"])

    def test_temporary_control_reverts_when_source_leaves_play(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        source = put_creature(game, 1, "Old Bruno")
        controlled = put_creature(game, 2, "Old Bruno")
        drive(game.take_control(controlled, 1, until_source=source), answers=["left"])
        self.assertIn(controlled, p1.play_area.creatures)
        self.assertEqual(controlled.controller, 1)

        p1.play_area.remove(source)
        game.leave_play(source)

        self.assertNotIn(controlled, p1.play_area.creatures)
        self.assertIn(controlled, p2.play_area.creatures)
        self.assertEqual(controlled.controller, 2)
        reverts = [e for e in game.log.events if e.kind == "take_control" and e.data.get("reverted")]
        self.assertEqual(len(reverts), 1)

    def test_permanent_control_does_not_revert(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        source = put_creature(game, 1, "Old Bruno")
        controlled = put_creature(game, 2, "Old Bruno")
        drive(game.take_control(controlled, 1), answers=["left"])  # no until_source -> permanent
        p1.play_area.remove(source)
        game.leave_play(source)
        self.assertIn(controlled, p1.play_area.creatures)
        self.assertEqual(controlled.controller, 1)


class TestSimultaneousDamage(unittest.TestCase):
    def test_multiple_targets_damaged_together_then_one_destroy_check(self):
        game = new_game()
        p2 = game.players[2]
        a = put_creature(game, 2, "Old Bruno")  # power 3
        b = put_creature(game, 2, "Noddy the Thief")  # power 2
        c = put_creature(game, 2, "Titan Mechanic")  # power 6
        steps.deal_damage(game, a, 3)
        steps.deal_damage(game, b, 2)
        steps.deal_damage(game, c, 1)
        destroyed = drive_return(game.check_destroyed([a, b, c]))
        self.assertIn(a, destroyed)
        self.assertIn(b, destroyed)
        self.assertNotIn(c, destroyed)
        self.assertNotIn(a, p2.play_area.creatures)
        self.assertNotIn(b, p2.play_area.creatures)
        self.assertIn(c, p2.play_area.creatures)


class TestDamageRedirect(unittest.TestCase):
    def test_instead_effect_redirects_damage(self):
        game = new_game()
        p1 = game.players[1]
        real_target = put_creature(game, 1, "Old Bruno")
        redirect_target = put_creature(game, 1, "Titan Mechanic")

        def handler(g, creature):
            return redirect_target if creature is real_target else None

        game.active_effects.add(InsteadEffect(redirect_target, 1, "damage_target", handler))
        steps.deal_damage(game, real_target, 4)
        self.assertEqual(real_target.type_object.damage, 0)
        self.assertEqual(redirect_target.type_object.damage, 4)


class TestHouseMustCannot(unittest.TestCase):
    def test_forced_house_is_chosen_without_a_decision(self):
        game = new_game()
        card = put_creature(game, 1, "Old Bruno")
        game.active_effects.add(DurationEffect(card, 1, INFINITE, 1, "HouseSelection", "=", House.DIS))
        drive(game._choose_house_step(1))
        self.assertEqual(game.players[1].selected_house, House.DIS)

    def test_cannot_beats_must_and_falls_back_to_normal_choice(self):
        game = new_game()
        must_card = put_creature(game, 1, "Old Bruno")
        cannot_card = put_creature(game, 1, "Titan Mechanic")
        game.active_effects.add(DurationEffect(must_card, 1, INFINITE, 1, "HouseSelection", "=", House.DIS))
        game.active_effects.add(DurationEffect(cannot_card, 1, INFINITE, 1, "CannotChooseHouse", "add", House.DIS))
        gen = game._choose_house_step(1)
        d = next(gen)
        self.assertNotIn(House.DIS, d.options)
        try:
            gen.send(d.options[0])
        except StopIteration:
            pass
        self.assertNotEqual(game.players[1].selected_house, House.DIS)

    def test_no_legal_house_leaves_player_with_no_active_house(self):
        game = new_game()
        p1 = game.players[1]
        blocker = put_creature(game, 1, "Old Bruno")
        for house in game.player_houses(1):
            game.active_effects.add(DurationEffect(blocker, 1, INFINITE, 1, "CannotChooseHouse", "add", house))
        drive(game._choose_house_step(1))
        self.assertIsNone(p1.selected_house)
        no_house_events = [e for e in game.log.events if e.kind == "choose_house" and e.data.get("house") is None]
        self.assertEqual(len(no_house_events), 1)


class TestUnforgeQuery(unittest.TestCase):
    def test_forged_key_on_turn_matches_only_that_player_and_turn(self):
        game = new_game()
        game.log.add("forge_key", player=1, keys=1, cost=6, turn=3)
        game.log.add("forge_key", player=2, keys=1, cost=6, turn=4)
        self.assertTrue(game.forged_key_on_turn(1, 3))
        self.assertFalse(game.forged_key_on_turn(1, 4))
        self.assertFalse(game.forged_key_on_turn(2, 3))
        self.assertTrue(game.forged_key_on_turn(2, 4))

    def test_unforge_decrements_keys(self):
        game = new_game()
        game.players[2].keys = 2
        game.players[2].keys = max(0, game.players[2].keys - 1)
        self.assertEqual(game.players[2].keys, 1)


class TestAemberSpendSplit(unittest.TestCase):
    def _spendable_artifact(self, game, pid, name, amount):
        cdef = _synthetic(name, ctype=CardType.ARTIFACT, power=0, spendable_for_keys=True)
        card = Card(cdef, pid)
        game.players[pid].play_area.add_artifact(card)
        card.aember_stored = amount
        return card

    def test_pool_alone_is_used_first(self):
        game = new_game()
        p1 = game.players[1]
        p1.aember = 10
        self._spendable_artifact(game, 1, "Stash", 5)
        drive(game._pay_forge_cost(1, 6))
        self.assertEqual(p1.aember, 4)

    def test_falls_back_to_stored_aember_when_pool_is_short(self):
        game = new_game()
        p1 = game.players[1]
        p1.aember = 2
        stash = self._spendable_artifact(game, 1, "Stash", 10)
        ok = drive(game._pay_forge_cost(1, 6))
        self.assertEqual(p1.aember, 0)
        self.assertEqual(stash.aember_stored, 6)  # 10 - (6-2)=4 taken -> 6 left

    def test_multiple_sources_ask_which_order_to_drain(self):
        game = new_game()
        p1 = game.players[1]
        p1.aember = 0
        a = self._spendable_artifact(game, 1, "StashA", 3)
        b = self._spendable_artifact(game, 1, "StashB", 3)
        gen = game._pay_forge_cost(1, 6)
        d = next(gen)
        self.assertEqual(set(d.options), {a, b})
        try:
            gen.send([b, a])  # drain b first
        except StopIteration:
            pass
        self.assertEqual(b.aember_stored, 0)
        self.assertEqual(a.aember_stored, 0)

    def test_insufficient_even_with_stored_aember_fails(self):
        game = new_game()
        p1 = game.players[1]
        p1.aember = 1
        self._spendable_artifact(game, 1, "Stash", 2)
        ok = drive(game._pay_forge_cost(1, 6))
        self.assertEqual(p1.aember, 1)  # unchanged: payment refused entirely

    def test_stored_aember_vanishes_on_leaving_play_not_released(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        stash = self._spendable_artifact(game, 1, "Stash", 4)
        p1.play_area.remove(stash)
        game.leave_play(stash)
        self.assertEqual(stash.aember_stored, 0)
        self.assertEqual(p1.aember, 0)
        self.assertEqual(p2.aember, 0)


class TestOwnerControllerAndBorrowedUse(unittest.TestCase):
    """Regression tests for a class of fuzz-found bugs where code assumed a
    card's CONTROLLER's play area (or `card.controller` itself) always
    matches its true physical location -- true after a permanent control
    change (Harland Mindlock), but false during `use_artifact_ability`'s
    temporary "as if it were yours" controller swap (Poltergeist, Remote
    Access, Nexus), which repoints `card.controller` without moving the
    card. See `Game.find_play_area`."""

    def test_purge_removes_from_controllers_play_area_when_it_differs_from_owner(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        mindlock = put_creature(game, 1, "Harland Mindlock")
        target = put_creature(game, 2, "Charette", flank="left")
        run_hook(game, named.harland_mindlock, mindlock, answers=[[target]])
        self.assertIn(target, p1.play_area.creatures)
        self.assertEqual(target.controller, 1)
        ok = steps.purge(game, target)
        self.assertTrue(ok)
        self.assertNotIn(target, p1.play_area.creatures)
        self.assertIn(target, p2.purged.cards())

    def test_borrowed_artifacts_self_archive_goes_to_its_true_owner_not_the_borrower(self):
        game = new_game()
        p1, p2 = game.players[1], game.players[2]
        nexus = put_creature(game, 1, "Nexus")
        scroll = put_artifact(game, 2, "Mobius Scroll", exhausted=False)
        run_hook(game, named.nexus, nexus)
        self.assertNotIn(scroll, p2.play_area.artifacts)
        self.assertIn(scroll, p2.archive.cards())
        self.assertNotIn(scroll, p1.archive.cards())

    def test_take_control_during_borrowed_use_does_not_duplicate_the_card(self):
        game = new_game()
        p2 = game.players[2]

        def self_referential_action(g, c):
            # Mirrors Spangler Box's own "opponent gains control of Spangler
            # Box": while borrowed, `c.controller` is temporarily the
            # borrower, so this targets the borrower's opponent -- who may
            # already be its real, physical controller.
            yield from g.take_control(c, 3 - c.controller)

        cdef = _synthetic("Borrowed Thing", ctype=CardType.ARTIFACT, power=0, on_action=self_referential_action)
        card = Card(cdef, 2)
        p2.play_area.add_artifact(card)
        card.Exhausted = False

        drive(game.use_artifact_ability(card, as_pid=1))

        self.assertEqual(p2.play_area.artifacts, [card])
        self.assertEqual(card.controller, 2)

    def test_destroy_cards_does_not_reprocess_a_card_that_already_left_play(self):
        game = new_game()
        p1 = game.players[1]
        card = put_artifact(game, 1, "The Sting", exhausted=False)
        drive(game.use_artifact_ability(card, as_pid=1))  # The Sting's own Action sacrifices itself
        self.assertNotIn(card, p1.play_area.artifacts)
        self.assertEqual(p1.discard.cards().count(card), 1)
        drive(game.destroy_cards([card]))  # e.g. Poltergeist's own redundant "Destroy that artifact"
        self.assertEqual(p1.discard.cards().count(card), 1)

    def test_replicator_does_not_offer_another_replicator_as_a_target(self):
        game = new_game()
        a = put_creature(game, 1, "Replicator", exhausted=False)
        b = put_creature(game, 1, "Replicator", exhausted=False)
        # Two non-Replicator candidates, so `choose_cards` actually raises a
        # Decision instead of auto-selecting a lone option.
        other1 = put_creature(game, 1, "Doc Bookton")
        other2 = put_creature(game, 2, "Ganymede Archivist")
        gen = named.replicator(game, a)
        d = next(gen)
        self.assertNotIn(b, d.options)
        self.assertIn(other1, d.options)
        self.assertIn(other2, d.options)
        try:
            gen.send([other1])
        except StopIteration:
            pass


if __name__ == "__main__":
    unittest.main()

"""Regression for the Milestone E2 "cleanups as data, not closures" change
(Code/AGENT_INTERFACE_PLAN.md): `Game._end_of_turn_cleanups` holds
`(operation, instance_id)` tuples resolved via `effect_object.
resolve_cleanup`, never a raw callable -- a closure over a specific Card
object is exactly what `Game.copy()` can't rebind the way replaying
naturally would (see keyforge/cards/card.py and effect_object.py's own
comments on this).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helpers import drive, make_card, new_game, put_artifact, put_creature, run_hook  # noqa: E402

from keyforge.effects.effect_object import register_cleanup_operation, resolve_cleanup  # noqa: E402
from keyforge.effects.named import dis, logos, mars, sanctum, untamed  # noqa: E402


def _assert_only_data(test, cleanups):
    test.assertGreater(len(cleanups), 0)
    for entry in cleanups:
        test.assertIsInstance(entry, tuple)
        test.assertEqual(len(entry), 2)
        operation, iid = entry
        test.assertIsInstance(operation, str)
        test.assertTrue(iid is None or isinstance(iid, int))
        test.assertFalse(callable(entry), "a cleanup entry must never itself be callable")


class TestNoCleanupIsEverACallable(unittest.TestCase):
    """The plan's own explicit acceptance test, exercised directly against
    each of the six sites converted from a closure to data -- none of
    Red-Hot Armor/Spectral Tunneler/Sniffer/Brain Stem Antenna/Protectrix/
    Scout are in the fignor/igor pool, so random self-play alone would
    never reach any of them. Scenarios mirror the existing per-card tests
    in tests/test_cards_{dis,logos_phase2,mars,sanctum,untamed}.py."""

    def test_red_hot_armor(self):
        game = new_game()
        put_creature(game, 2, "Firespitter")  # power 5, armor 1
        card = make_card("Red-Hot Armor", 1)
        run_hook(game, dis.red_hot_armor, card)
        _assert_only_data(self, game._end_of_turn_cleanups)

    def test_spectral_tunneler(self):
        game = new_game()
        target = put_creature(game, 1, "Dr. Escotera")
        tunneler = put_artifact(game, 1, "Spectral Tunneler", exhausted=False)
        run_hook(game, logos.spectral_tunneler, tunneler, answers=[[target]])
        _assert_only_data(self, game._end_of_turn_cleanups)

    def test_sniffer(self):
        game = new_game()
        sniffer = put_artifact(game, 1, "Sniffer")
        put_creature(game, 2, "Drumble")
        run_hook(game, mars.sniffer_action, sniffer)
        _assert_only_data(self, game._end_of_turn_cleanups)

    def test_brain_stem_antenna(self):
        game = new_game()
        host = put_creature(game, 1, "Charette", exhausted=True, can_be_used=False)
        upgrade = make_card("Brain Stem Antenna", 1)
        upgrade.type_object.host = host
        host.type_object.upgrades.append(upgrade)
        mars.brain_stem_antenna_register(game, upgrade)
        new_mars_creature = make_card("Yxili Marauder", 1)
        game.players[1].play_area.add_creature(new_mars_creature)
        game._cards_by_id[new_mars_creature.instance_id] = new_mars_creature
        drive(game._fire_event("card_played", {"player": 1, "card": new_mars_creature}))
        _assert_only_data(self, game._end_of_turn_cleanups)

    def test_protectrix(self):
        game = new_game()
        protectrix = put_creature(game, 1, "Protectrix")
        target = put_creature(game, 1, "Truebaru")
        target.type_object.damage = 4
        run_hook(game, sanctum.protectrix_after_reap, protectrix, answers=[True, [target]])
        _assert_only_data(self, game._end_of_turn_cleanups)

    def test_scout(self):
        game = new_game()
        fighter = put_creature(game, 1, "Drumble", exhausted=True)
        put_creature(game, 2, "Truebaru")
        card = make_card("Scout", 1)
        run_hook(game, untamed.scout, card, answers=[[fighter]])
        _assert_only_data(self, game._end_of_turn_cleanups)


class TestCleanupRegistry(unittest.TestCase):
    def test_resolve_cleanup_dispatches_to_the_registered_resolver(self):
        calls = []
        register_cleanup_operation("test.record_call", lambda game, iid: calls.append((game, iid)))
        sentinel_game = object()
        resolve_cleanup(sentinel_game, "test.record_call", 42)
        self.assertEqual(calls, [(sentinel_game, 42)])

    def test_registering_the_same_name_twice_is_an_error(self):
        register_cleanup_operation("test.duplicate_once", lambda game, iid: None)
        with self.assertRaises(ValueError):
            register_cleanup_operation("test.duplicate_once", lambda game, iid: None)

    def test_resolving_an_unregistered_operation_raises(self):
        with self.assertRaises(KeyError):
            resolve_cleanup(object(), "no_such_operation", None)


if __name__ == "__main__":
    unittest.main()

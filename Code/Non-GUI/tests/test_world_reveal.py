"""A hand revealed inside a determinized world shows the world's cards.

A re-deal isn't journaled, so the projection used to list the true game's
hand and zones: the search saw the opponent's real cards, and the chance
filter, told of cards the world's hand doesn't hold, raised "no consistent
world" (M11's random decks hold Psychic Bug and the like; Fignor and Igor
don't).
"""

from __future__ import annotations

import random
import unittest

from keyforge.config import GameConfig
from keyforge.determinize import chance_filter
from keyforge.game import Game
from sim.bc_corpus import EpsilonGreedy, restricted_random_deck


def _new_game(tag: str):
    rng = random.Random(tag)
    decks = (restricted_random_deck(rng, "R1"), restricted_random_deck(rng, "R2"))
    game = Game(GameConfig(decks=decks, seed=rng.getrandbits(31), max_turns=200))
    return game, rng, {1: EpsilonGreedy(rng.getrandbits(31), 0.2), 2: EpsilonGreedy(rng.getrandbits(31), 0.2)}


def _revealed(game, viewer, since):
    return {x[2] for x in game.projected(viewer)[since:] if x[0] == "reveal"}


class WorldReveal(unittest.TestCase):
    def test_saved_networks_signature_input_is_unchanged(self):
        """A new name in the state registry widens it, and no trained v2 network loads."""
        from agent.static_v2 import SIGNATURE_WIDTH

        self.assertEqual(SIGNATURE_WIDTH, 222)

    def _world(self):
        """A world whose opponent hand differs from the true game's."""
        for k in range(50):
            game, rng, _ = _new_game(f"world-reveal-{k}")
            viewer = game.pending_decision.player
            world = game.fork_determinized(viewer, rng, sampler="chance_exact")
            hand = {c.instance_id for c in world.players[3 - viewer].hand.cards()}
            if hand != {c.instance_id for c in game.players[3 - viewer].hand.cards()}:
                return world, viewer, hand
        self.fail("no world with a different hand")

    def test_the_viewer_sees_the_worlds_hand(self):
        world, viewer, hand = self._world()
        since = len(world.projected(viewer))
        world.reveal_hand(3 - viewer, viewer)
        self.assertEqual(_revealed(world, viewer, since), hand)

    def test_a_copy_and_a_rebuilt_projection_agree(self):
        world, viewer, hand = self._world()
        world.projected(viewer)
        if world.execution == "compiled":  # (native code can't be copied mid-decision)
            copied = world.copy()
            copied.reveal_hand(3 - viewer, viewer)
            self.assertEqual(_revealed(copied, viewer, 0), hand)
        world.reveal_hand(3 - viewer, viewer)
        del world.__dict__["_projectors"]
        self.assertEqual(_revealed(world, viewer, 0), hand)

    def test_an_options_zone_is_the_worlds(self):
        world, viewer, hand = self._world()
        o = 3 - viewer
        where = world.__dict__.setdefault("_projectors", {})
        world.projected(viewer)
        for iid in hand:
            self.assertEqual(where[viewer].where[iid], ("hand", o))

    def test_the_filter_survives_play_inside_worlds_on_random_decks(self):
        """These seeds raised before the fix (6 of the first 40 games)."""
        for i in range(40):
            game, rng, bots = _new_game(f"repro-{i}")
            while not game.is_over:
                d = game.pending_decision
                if rng.random() < 0.1:
                    world = game.fork_determinized(d.player, rng, sampler="chance_exact")
                    for _ in range(6):
                        if world.is_over:
                            break
                        chance_filter(world, d.player)
                        dd = world.pending_decision
                        world.submit(bots[dd.player].decide(world.view_for(dd.player), dd))
                game.submit(bots[d.player].decide(game.view_for(d.player), d))


if __name__ == "__main__":
    unittest.main()

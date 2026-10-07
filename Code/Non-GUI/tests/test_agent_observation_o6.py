"""Agent Observation Plan, Milestone O6: history encoding.

- History bytes are the same in every consistent world (I1; under sigma
  they follow from the projection's own invariance, O1).
- A world's prefix plus its suffix is byte-identical to encoding from
  scratch, and so are the turn tokens and summaries folded incrementally.
- A round trip through the shard form gives back the same rows.
"""

import random
import unittest

from agent.history import HistoryEncoder, history_for, summaries, turn_tokens
from bots.random_bot import RandomBot
from keyforge.enums import Resample
from keyforge.game import Game
from tools.o1_acceptance import POOLS, bots_for, game_config


class TestHistory(unittest.TestCase):
    def test_invariant_and_prefix_plus_suffix(self):
        checked = 0
        for pool in POOLS:
            for i in range(2):
                config = game_config(pool, i)
                game = Game(config)
                bots = bots_for(i, config.seed)
                n = 0
                while not game.is_over:
                    if n % 13 == 0 and game.copy_anywhere:
                        for v in (1, 2):
                            h = history_for(game, v)
                            h.freeze()
                            world = game.fork_determinized(v, random.Random(n), Resample.ALL, backend="copy",
                                                           sampler="chance_exact")
                            fresh = HistoryEncoder(world, v)
                            fresh.update(world.projected(v))
                            self.assertEqual(fresh.to_bytes(), h.to_bytes(), f"{pool} {i} decision {n} viewer {v}")
                            self.assertEqual(fresh.digest(), h.prefix_digest())
                            branch = h.copy()
                            sim = RandomBot(seed=n)
                            for _ in range(8):
                                if world.is_over:
                                    break
                                d = world.pending_decision
                                world.submit(sim.decide(world.view_for(d.player), d))
                            branch.update(world.projected(v))
                            scratch = HistoryEncoder(world, v)
                            scratch.update(world.projected(v))
                            self.assertEqual(branch.to_bytes(), scratch.to_bytes())
                            self.assertEqual(turn_tokens(branch), turn_tokens(scratch))
                            self.assertEqual(summaries(branch, world.turn_number), summaries(scratch, world.turn_number))
                            checked += 1
                    d = game.pending_decision
                    game.submit(bots[d.player].decide(game.view_for(d.player), d))
                    n += 1
        self.assertGreater(checked, 50)

    def test_shard_round_trip(self):
        game = Game(game_config("phase2", 3))
        bots = bots_for(3, game.config.seed)
        while not game.is_over:
            d = game.pending_decision
            game.submit(bots[d.player].decide(game.view_for(d.player), d))
        h = history_for(game, 2)
        ints, floats, pointers = HistoryEncoder.rows_from_bytes(h.to_bytes())
        self.assertEqual((ints, floats, pointers), (h.ints, h.floats, h.pointers))

    def test_hidden_identities_stay_hidden(self):
        game = Game(game_config("phase1", 0))
        bots = bots_for(0, game.config.seed)
        for _ in range(80):
            d = game.pending_decision
            game.submit(bots[d.player].decide(game.view_for(d.player), d))
        h = history_for(game, 1)
        opp_draws = 0
        for t in range(0, len(h.pointers), 3):
            row, role, ptr = h.pointers[t:t + 3]
            if ptr == -3:
                opp_draws += 1
            self.assertTrue(ptr < 72)
        self.assertGreater(opp_draws, 0)


if __name__ == "__main__":
    unittest.main()

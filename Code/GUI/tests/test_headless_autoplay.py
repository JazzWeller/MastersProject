"""Runs full bot-vs-bot games through Board + Director + Animator, headless,
and checks that the board never drifts out of sync with the engine."""

import unittest

import tests.helpers  # noqa: F401  (sets up sys.path + dummy SDL driver)

from bots.random_bot import RandomBot
from keyforge.config import GameConfig
from keyforge.game import Game

from gui.anim.animator import Animator
from gui.assets import AssetCache
from gui.board import Board
from gui.anim.director import Director
from gui.snapshot import build_snapshot

EPSILON = 0.5


class TestHeadlessAutoplay(unittest.TestCase):
    def test_games_run_and_stay_in_sync(self):
        self._run(("fignor", "igor"), range(20))

    def test_phase_3_presets_stay_in_sync(self):
        # Armor, capture, reveal, stun-on-entry and the rest of the four
        # Phase 3 houses, which fignor/igor never exercise.
        for seed, decks in enumerate([("stonewall", "starfall"), ("vigil", "thornwood"), ("starfall", "vigil"), ("thornwood", "stonewall")]):
            self._run(decks, [seed])

    def _run(self, decks, seeds):
        assets = AssetCache()
        for seed in seeds:
            game = Game(GameConfig(decks=decks, seed=seed, max_turns=30))
            bots = {1: RandomBot(seed=seed), 2: RandomBot(seed=seed + 500)}
            board = Board(assets, viewer=1)
            animator = Animator()

            before = build_snapshot(game, 1)
            board.snap_all_instant(before)

            while not game.is_over:
                d = game.pending_decision
                choice = bots[d.player].decide(game.view_for(d.player), d)
                start = len(game.log.events)
                game.submit(choice)
                events = game.log.events[start:]
                after = build_snapshot(game, 1)

                beat = Director.build(board, before, after, events)
                animator.enqueue(beat)
                animator.skip()  # resolve instantly; no real display loop in a headless test
                self.assertFalse(animator.is_busy)

                # every sprite must have settled exactly onto its slot
                for iid, cs in after.cards.items():
                    sprite = board.sprites[iid]
                    x, y, rot, w, h = board.slot(cs, after)
                    self.assertAlmostEqual(sprite.x, x, delta=EPSILON, msg=(seed, iid, cs.zone))
                    self.assertAlmostEqual(sprite.y, y, delta=EPSILON, msg=(seed, iid, cs.zone))
                    self.assertEqual(sprite.face_up, cs.face_up)

                for pid, ps in after.players.items():
                    self.assertAlmostEqual(board.hud_states[pid].displayed_aember, ps.aember, delta=EPSILON)

                before = after

            self.assertTrue(game.is_over)


if __name__ == "__main__":
    unittest.main()

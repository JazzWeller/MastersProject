"""Game history database and step-by-step replay (Code/PLAYTEST_FIX_PLAN.md H1-H5)."""

import os
import tempfile
import unittest
import zlib

import tests.helpers  # noqa: F401

import pygame

from bots.heuristic_bot import HeuristicBot
from keyforge.config import GameConfig
from keyforge.game import Game

from gui import settings as S
from gui.app import App
from gui.engine_bridge import MatchSettings, ReplayBridge
from gui.history import GameHistory, decode_record, encode_record
from gui.scenes.game_over_scene import GameOverScene
from gui.scenes.game_scene import GameScene
from gui.scenes.history_scene import HistoryScene
from gui.scenes.replay_scene import ReplayScene


def _bot_game(seed, max_turns=None):
    config = GameConfig(decks=("fignor", "igor"), seed=seed, max_turns=max_turns)
    game = Game(config)
    bots = {1: HeuristicBot(seed), 2: HeuristicBot(seed + 1)}
    while not game.is_over:
        d = game.pending_decision
        game.submit(bots[d.player].decide(game.view_for(d.player), d))
    return config, game


def _events(game):
    return [(e.kind, {k: v for k, v in e.data.items() if "iid" not in k}) for e in game.log.events]


class TestHistoryDatabase(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(), "h.sqlite3")
        self.history = GameHistory(self.path)

    def tearDown(self):
        self.history.close()

    def test_record_is_compressed_and_round_trips(self):
        config, game = _bot_game(7)
        blob = encode_record(config, game.choice_record)
        self.assertLess(len(blob), 4096)
        raw = zlib.decompress(blob)
        self.assertLess(len(blob), len(raw))
        config2, choices = decode_record(blob)
        self.assertEqual(choices, list(game.choice_record))
        self.assertEqual((config2.seed, tuple(config2.decks)), (config.seed, tuple(config.decks)))

    def test_store_list_load_replay_reproduces_the_game(self):
        for seed in range(5):
            config, game = _bot_game(seed)
            gid = self.history.start(config, "bot", "bot", game)
            self.history.update(gid, config, game)
            summary = self.history.get(gid)
            self.assertEqual(summary.status, "finished")
            self.assertEqual(summary.winner, game.result["winner"])
            self.assertEqual((summary.p1_keys, summary.p2_keys), (game.players[1].keys, game.players[2].keys))
            loaded_config, record = self.history.load(gid)
            bridge = ReplayBridge(loaded_config, record)
            bridge.rebuild_to(len(record))
            self.assertTrue(bridge.is_over)
            self.assertEqual(bridge.game.result, game.result)
            self.assertEqual(_events(bridge.game), _events(game))
        self.assertEqual(self.history.count(), 5)
        self.assertEqual([s.id for s in self.history.list(limit=2)], [5, 4])

    def test_unfinished_game_is_marked_abandoned_and_replays_to_where_it_stopped(self):
        config = GameConfig(seed=3)
        game = Game(config)
        bot = HeuristicBot(3)
        for _ in range(25):
            d = game.pending_decision
            game.submit(bot.decide(game.view_for(d.player), d))
        gid = self.history.start(config, "human", "bot", game)
        self.history.update(gid, config, game)
        self.history.close()
        self.history = GameHistory(self.path)  # a new session: in-progress rows become abandoned
        self.assertEqual(self.history.get(gid).status, "abandoned")
        _config, record = self.history.load(gid)
        bridge = ReplayBridge(_config, record)
        bridge.rebuild_to(len(record))
        self.assertEqual(bridge.position, 25)
        self.assertTrue(bridge.at_end)
        self.assertFalse(bridge.is_over)

    def test_delete(self):
        config, game = _bot_game(1, max_turns=4)
        gid = self.history.start(config, "bot", "bot", game)
        self.history.delete(gid)
        self.assertIsNone(self.history.get(gid))


class TestRecordingAndReplayScenes(unittest.TestCase):
    def setUp(self):
        self._think = S.T_BOT_THINK
        S.T_BOT_THINK = 1

    def tearDown(self):
        S.T_BOT_THINK = self._think

    def _finish(self, app):
        for _ in range(60000):
            top = app.scenes[-1]
            if isinstance(top, GameOverScene):
                return top
            top.update(60)
        self.fail("game did not finish")

    def test_a_played_game_is_recorded_and_replayable_from_game_over(self):
        app = App(window_size=(1600, 900))
        app.push(GameScene(MatchSettings(p1_seat="bot", p2_seat="bot", seed=12, max_turns=12)))
        game_scene = app.scenes[-1]
        go = self._finish(app)
        go.on_enter()
        app.draw_scenes()  # game over draws over the final board without error
        gid = go._history_id()
        self.assertIsNotNone(gid)
        summary = app.history.get(gid)
        self.assertEqual(summary.status, "finished")
        self.assertEqual(summary.decisions, len(game_scene.bridge.game.choice_record))
        self.assertIsNotNone(summary.seed)  # "random" games still get a concrete, replayable seed

        go.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=go.buttons["replay"].rect.center, button=1))
        replay = app.scenes[-1]
        self.assertIsInstance(replay, ReplayScene)
        self.assertEqual(replay.bridge.length, summary.decisions)

        replay.jump_to(replay.bridge.length)
        self.assertEqual(replay.bridge.game.result, game_scene.bridge.game.result)
        replay.jump_to(10)
        self.assertEqual(replay.bridge.position, 10)
        replay.step(1)
        for _ in range(200):
            replay.update(60)
        self.assertEqual(replay.bridge.position, 11)
        replay.step(-1)
        self.assertEqual(replay.bridge.position, 10)
        replay._turn_jump(1)
        self.assertGreater(replay.bridge.position, 10)
        replay.switch_perspective()
        app.draw_scenes()

    def test_history_scene_lists_and_opens_games(self):
        app = App(window_size=(1600, 900))
        app.push(GameScene(MatchSettings(p1_seat="bot", p2_seat="bot", seed=2, max_turns=6)))
        self._finish(app)
        app.scenes.clear()
        app.push(HistoryScene())
        hs = app.scenes[-1]
        app.draw_scenes()
        self.assertGreaterEqual(len(hs.rows), 1)
        _kind, replay_button = hs._row_buttons(0)[0]
        hs.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=replay_button.rect.center, button=1))
        self.assertIsInstance(app.scenes[-1], ReplayScene)
        app.draw_scenes()

    def test_replay_survives_stepping_through_every_decision(self):
        config, game = _bot_game(21, max_turns=10)
        app = App(window_size=(1600, 900))
        app.push(ReplayScene(config, list(game.choice_record)))
        rs = app.scenes[-1]
        rs.toggle_play()
        for _ in range(20000):
            if rs.bridge.at_end and not rs.animator.is_busy:
                break
            rs.update(700)
            if _ % 25 == 0:
                app.draw_scenes()
        self.assertTrue(rs.bridge.at_end)
        self.assertEqual(rs.bridge.game.result, game.result)


if __name__ == "__main__":
    unittest.main()

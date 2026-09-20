"""Bot-vs-bot playthroughs of the Match scene flow (Reversal/Adaptive GUI):
MatchGameScene -> MatchInterstitialScene -> FirstPlayerScene/BiddingScene ->
... -> MatchOverScene. See Code/PHASE_2_PLAN.md v1.2/v1.3."""

import unittest

import tests.helpers  # noqa: F401

import pygame

from bots.heuristic_bot import HeuristicBot

from gui import settings as S
from gui.app import App
from gui.engine_bridge import MatchSettings
from gui.scenes.history_scene import HistoryScene
from gui.scenes.match_scene import (
    BiddingScene,
    FirstPlayerScene,
    MatchGameScene,
    MatchInterstitialScene,
    MatchOverScene,
    MatchScene,
)

from keyforge.enums import DecisionKind


def _drive_to_match_over(app, frames=300000):
    seen_kinds = set()
    for _ in range(frames):
        top = app.scenes[-1]
        seen_kinds.add(type(top).__name__)
        top.update(30)
        if _ % 50 == 0:
            app.draw_scenes()
        if isinstance(top, MatchOverScene):
            return top, seen_kinds
    raise AssertionError("match did not reach MatchOverScene")


class TestMatchSceneFlow(unittest.TestCase):
    def setUp(self):
        self._think = S.T_BOT_THINK
        S.T_BOT_THINK = 1

    def tearDown(self):
        S.T_BOT_THINK = self._think

    def _run(self, fmt, seed):
        app = App(window_size=(1600, 900))
        settings = MatchSettings(p1_deck="fignor", p2_deck="igor", p1_seat="bot", p2_seat="bot", format=fmt, seed=seed)
        match_scene = MatchScene(settings)
        app.push(match_scene)
        over, seen = _drive_to_match_over(app)
        return app, match_scene, over, seen

    def test_archon_format_single_game_reaches_match_over(self):
        app, match_scene, over, seen = self._run("archon", seed=1)
        self.assertIn("MatchGameScene", seen)
        self.assertNotIn("FirstPlayerScene", seen)
        self.assertNotIn("BiddingScene", seen)
        self.assertEqual(len(match_scene.bridge.match.games), 1)
        self.assertIsNotNone(over.bridge.match.result["winner"])

    def test_reversal_format_swaps_decks(self):
        app, match_scene, over, seen = self._run("reversal", seed=1)
        match = match_scene.bridge.match
        self.assertEqual(len(match.games), 1)
        self.assertEqual(match.games[0].seat_decks, {1: "igor", 2: "fignor"})

    def test_adaptive_two_zero_skips_bidding(self):
        # Find a seed where the match ends 2-0 (no bidding needed) to
        # exercise the FirstPlayerScene without BiddingScene.
        for seed in range(30):
            app = App(window_size=(1600, 900))
            settings = MatchSettings(p1_deck="fignor", p2_deck="igor", p1_seat="bot", p2_seat="bot", format="adaptive", seed=seed)
            match_scene = MatchScene(settings)
            app.push(match_scene)
            over, seen = _drive_to_match_over(app)
            if len(match_scene.bridge.match.games) == 2:
                self.assertIn("FirstPlayerScene", seen)
                self.assertIn("MatchInterstitialScene", seen)
                self.assertNotIn("BiddingScene", seen)
                return
        self.skipTest("no 2-0 adaptive seed found in range")

    def test_adaptive_split_reaches_bidding_and_game_three(self):
        for seed in range(30):
            app = App(window_size=(1600, 900))
            settings = MatchSettings(p1_deck="fignor", p2_deck="igor", p1_seat="bot", p2_seat="bot", format="adaptive", seed=seed)
            match_scene = MatchScene(settings)
            app.push(match_scene)
            over, seen = _drive_to_match_over(app)
            match = match_scene.bridge.match
            if len(match.games) == 3:
                self.assertIn("BiddingScene", seen)
                self.assertIn("FirstPlayerScene", seen)
                self.assertIsNotNone(match.bid)
                self.assertGreaterEqual(match.bid.amount, 0)
                self.assertLessEqual(match.bid.amount, 24)
                return
        self.skipTest("no split (game-3) adaptive seed found in range")

    def test_match_is_recorded_to_history_and_replayable(self):
        app = App(window_size=(1600, 900))
        app._history = None  # force GameHistory() to open at a temp path (see tests/helpers.py env var)
        settings = MatchSettings(p1_deck="fignor", p2_deck="igor", p1_seat="bot", p2_seat="bot", format="adaptive", seed=2)
        match_scene = MatchScene(settings)
        app.push(match_scene)
        over, seen = _drive_to_match_over(app)
        history_id = match_scene.bridge.last_history_id
        self.assertIsNotNone(history_id)
        summary = app.history.get_match(history_id)
        self.assertEqual(summary.status, "finished")
        games = app.history.list(match_id=history_id)
        self.assertEqual(len(games), len(match_scene.bridge.match.games))


def _click(scene, button) -> None:
    scene.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=button.rect.center, button=1))


class TestHumanVsBotAdaptivePlaythrough(unittest.TestCase):
    """Clicks through a full human-vs-bot Adaptive match via real click
    events on every new screen (bid, first-player, interstitial), the same
    way test_playtest_fixes.py clicks through GameOverScene/HistoryScene."""

    def setUp(self):
        self._think = S.T_BOT_THINK
        S.T_BOT_THINK = 1

    def tearDown(self):
        S.T_BOT_THINK = self._think

    def _click_through(self, app, human_seat, frames=200000):
        """Drives every frame: per-game human decisions are answered via the
        DecisionPanel (the same entry point real clicks feed into), and the
        new match-level screens are answered with real button clicks."""
        bot = HeuristicBot(0)
        clicked_kinds = set()
        for _ in range(frames):
            top = app.scenes[-1]
            if isinstance(top, MatchOverScene):
                return top, clicked_kinds
            top.update(30)
            if isinstance(top, MatchGameScene):
                if top.animator.is_busy:
                    continue
                d = top.bridge.pending_decision
                if d is None or top.bridge.seat_is_bot(d.player) or top.panel.decision is not d:
                    continue
                choice = bot.decide(top.bridge.game.view_for(d.player), d)
                top.panel._submit(choice)
            elif isinstance(top, FirstPlayerScene):
                if top._bot_choice is not None:
                    continue  # the bot is "thinking"; update() will submit it
                clicked_kinds.add("FirstPlayerScene")
                _click(top, top.buttons["first"])
            elif isinstance(top, BiddingScene):
                if top._bot_choice is not None:
                    continue
                clicked_kinds.add("BiddingScene")
                _click(top, top.pass_button)  # a human happy to let the bot have it
            elif isinstance(top, MatchInterstitialScene):
                clicked_kinds.add("MatchInterstitialScene")
                _click(top, top.button)
        raise AssertionError("match did not reach MatchOverScene")

    def test_human_can_click_through_a_full_adaptive_match(self):
        # Search for a seed that reaches bidding, to exercise every new screen.
        for seed in range(30):
            app = App(window_size=(1600, 900))
            settings = MatchSettings(p1_deck="fignor", p2_deck="igor", p1_seat="human", p2_seat="bot", format="adaptive", seed=seed)
            match_scene = MatchScene(settings)
            app.push(match_scene)
            over, clicked = self._click_through(app, human_seat=1)
            if len(match_scene.bridge.match.games) == 3:
                self.assertIn("BiddingScene", clicked)
                self.assertIn("FirstPlayerScene", clicked)
                self.assertIn("MatchInterstitialScene", clicked)
                self.assertTrue(over.bridge.match.is_over)
                return
        self.skipTest("no split (game-3) seed found in range for a human-seat match")


class TestHistorySceneMatchesView(unittest.TestCase):
    def setUp(self):
        self._think = S.T_BOT_THINK
        S.T_BOT_THINK = 1

    def tearDown(self):
        S.T_BOT_THINK = self._think

    def test_toggle_to_matches_view_and_drill_into_its_games(self):
        app = App(window_size=(1600, 900))
        app._history = None
        settings = MatchSettings(p1_deck="fignor", p2_deck="igor", p1_seat="bot", p2_seat="bot", format="adaptive", seed=2)
        match_scene = MatchScene(settings)
        app.push(match_scene)
        _drive_to_match_over(app)
        expected_games = len(match_scene.bridge.match.games)

        app.scenes.clear()
        app.push(HistoryScene())
        hs = app.scenes[-1]
        app.draw_scenes()

        nav = hs._nav_buttons()
        hs.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=nav["mode"].rect.center, button=1))
        self.assertEqual(hs.mode, "matches")
        self.assertGreaterEqual(len(hs.rows), 1)
        app.draw_scenes()

        _kind, games_button = hs._row_buttons(0)[0]
        self.assertEqual(_kind, "games")
        match_id = hs.rows[0].id
        hs.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=games_button.rect.center, button=1))
        self.assertEqual(hs.mode, "games")
        self.assertEqual(hs.match_filter, match_id)
        self.assertEqual(len(hs.rows), expected_games)
        app.draw_scenes()

        # Replay still works unmodified on a match-linked game row.
        _kind, replay_button = hs._row_buttons(0)[0]
        self.assertEqual(_kind, "replay")
        hs.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=replay_button.rect.center, button=1))
        from gui.scenes.replay_scene import ReplayScene

        self.assertIsInstance(app.scenes[-1], ReplayScene)


if __name__ == "__main__":
    unittest.main()

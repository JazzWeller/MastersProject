"""Drives whole games through real pygame click events (MOUSEBUTTONDOWN +
MOUSEBUTTONUP, exactly like a real click) instead of calling engine/bot APIs
directly, so this exercises DecisionPanel, the card-click and Options-modal
paths, the action bar, hot-seat curtain switching, and scene navigation —
not just the animation pipeline (see test_headless_autoplay.py for that).
"""

import unittest

import tests.helpers  # noqa: F401

import pygame

from gui import settings as S
from gui.app import App
from gui.engine_bridge import MatchSettings
from gui.scenes.curtain_scene import CurtainScene
from gui.scenes.game_over_scene import GameOverScene
from gui.scenes.game_scene import GameScene
from gui.scenes.menu_scene import MenuScene

MAX_FRAMES = 20000


def _click(scene, pos):
    scene.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=1))
    scene.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=pos, button=1))


def _play_through(app, dt=30.0, max_frames=MAX_FRAMES):
    """Runs `app` by auto-clicking: the first modal row when a modal is
    open, otherwise the first clickable card; advances curtains; stops at
    GameOverScene. Returns the frame count it took."""
    frames = 0
    while app.scenes and frames < max_frames:
        frames += 1
        top = app.scenes[-1]
        if isinstance(top, GameOverScene):
            return frames
        if isinstance(top, CurtainScene):
            app.canvas.fill(S.BG_DEEP)
            top.draw(app.canvas)
            _click(top, (S.CANVAS_W // 2, S.CANVAS_H // 2 + 75))
            continue
        top.update(dt)
        app.canvas.fill(S.BG_DEEP)
        top.draw(app.canvas)

        gscene = top
        if not isinstance(gscene, GameScene) or gscene.animator.is_busy or gscene.bridge.is_over:
            continue
        d = gscene.bridge.pending_decision
        if d is None or gscene.bridge.seat_is_bot(d.player) or d.player != gscene.viewer:
            continue
        if gscene.panel.decision is not d:
            continue
        p = gscene.panel
        assets = app.assets
        if p.mulligan_open or p.archive_open:
            button, _opt = p._review_buttons()[0]
            _click(top, button.rect.center)
        elif p.chooser_iid is not None:
            rows = p._chooser_option_rows(p._chooser_rect(gscene.board))
            if rows:
                _click(top, rows[0].center)
        elif p.modal_open:
            rect = p._modal_rect(assets)
            cells = p._modal_cells(p._modal_body(rect, assets))
            if cells:
                _click(top, cells[0][0].center)
            if p.result is None and p.decision is d and d.max_n > 1 and d.min_n <= len(p.picked):
                p.confirm()
        elif p.flank_open:
            rect = p.flank_rects(gscene.board, gscene.board.layout)["right"]
            _click(top, rect.center)
        elif p.card_option_map:
            iid = next(iter(p.card_option_map))
            sprite = gscene.board.sprites[iid]
            _click(top, (sprite.x, sprite.y))
            if p.result is None and p.chooser_iid is None and d.max_n > 1 and d.min_n <= len(p.picked):
                p.confirm()
        else:
            # Nothing on the board: End Turn (or Confirm) lives on the action bar.
            for button, _cb in p.action_bar_buttons(gscene.board.layout):
                if button.text.startswith(("End Turn", "Click again", "Confirm", "Choose none")):
                    _click(top, button.rect.center)
                    break
    return frames


class TestUiPlaythrough(unittest.TestCase):
    def setUp(self):
        self._orig_think = S.T_BOT_THINK
        S.T_BOT_THINK = 1  # don't burn real test time on bot "thinking" pauses

    def tearDown(self):
        S.T_BOT_THINK = self._orig_think

    def _run_case(self, p1_seat, p2_seat, seed):
        app = App()
        app.push(GameScene(MatchSettings(p1_seat=p1_seat, p2_seat=p2_seat, seed=seed, max_turns=12)))
        frames = _play_through(app)
        self.assertLess(frames, MAX_FRAMES, "game did not reach GameOverScene in time")
        self.assertIsInstance(app.scenes[-1], GameOverScene)
        self.assertTrue(app.scenes[-1].result.get("reason") in ("turn limit", "3 keys"))

    def test_human_vs_bot(self):
        self._run_case("human", "bot", seed=11)

    def test_bot_vs_human(self):
        self._run_case("bot", "human", seed=22)

    def test_human_vs_human_hotseat(self):
        self._run_case("human", "human", seed=5)

    def test_menu_to_game_navigation(self):
        app = App()
        app.push(MenuScene())
        menu = app.scenes[-1]
        menu.update(16)
        before = menu.p1_deck_i
        b, _cb = menu.buttons["p1_deck"]
        _click(menu, b.rect.center)
        self.assertEqual(menu.p1_deck_i, (before + 1) % len(menu.decks))
        b, _cb = menu.buttons["start"]
        _click(menu, b.rect.center)
        self.assertEqual([type(s).__name__ for s in app.scenes], ["MenuScene", "GameScene"])

    def test_game_over_rematch_and_main_menu(self):
        app = App()
        app.push(MenuScene())
        app.push(GameScene(MatchSettings(p1_seat="bot", p2_seat="bot", seed=9, max_turns=8)))
        frames = _play_through(app)
        self.assertLess(frames, MAX_FRAMES)
        go = app.scenes[-1]
        go.update(0)

        _click(go, go.buttons["rematch"].rect.center)
        self.assertEqual([type(s).__name__ for s in app.scenes], ["MenuScene", "GameScene"])

        frames = _play_through(app)
        self.assertLess(frames, MAX_FRAMES)
        go2 = app.scenes[-1]
        go2.update(0)
        _click(go2, go2.buttons["menu"].rect.center)
        self.assertEqual([type(s).__name__ for s in app.scenes], ["MenuScene"])

    def test_game_over_main_menu_without_a_menu_underneath(self):
        app = App()
        app.push(GameScene(MatchSettings(p1_seat="bot", p2_seat="bot", seed=1, max_turns=8)))
        frames = _play_through(app)
        self.assertLess(frames, MAX_FRAMES)
        go = app.scenes[-1]
        go.update(0)
        _click(go, go.buttons["menu"].rect.center)
        self.assertEqual([type(s).__name__ for s in app.scenes], ["MenuScene"])


if __name__ == "__main__":
    unittest.main()

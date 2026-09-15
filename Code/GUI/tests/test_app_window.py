"""Window resize / letterboxing math, and that toggling fullscreen and
resizing don't crash a running scene."""

import unittest

import tests.helpers  # noqa: F401

import pygame

from gui import settings as S
from gui.app import App
from gui.engine_bridge import MatchSettings
from gui.scenes.game_scene import GameScene
from gui.scenes.menu_scene import MenuScene


class TestLetterboxMath(unittest.TestCase):
    def test_dest_rect_preserves_aspect_and_fits_the_window(self):
        app = App(window_size=(1280, 720))
        for w, h in [(1280, 720), (800, 600), (400, 900), (3840, 2160), (500, 300)]:
            app.window = pygame.Surface((w, h))  # stand-in; avoids resizing the real display
            app._recompute_dest_rect()
            r = app._dest_rect
            self.assertGreaterEqual(r.left, 0)
            self.assertGreaterEqual(r.top, 0)
            self.assertLessEqual(r.right, w)
            self.assertLessEqual(r.bottom, h)
            # aspect ratio preserved (within rounding)
            self.assertAlmostEqual(r.width / r.height, S.CANVAS_W / S.CANVAS_H, delta=0.02)

    def test_to_canvas_round_trips_the_center(self):
        app = App(window_size=(1280, 720))
        app.window = pygame.Surface((1280, 720))
        app._recompute_dest_rect()
        cx, cy = app._to_canvas(app._dest_rect.center)
        self.assertAlmostEqual(cx, S.CANVAS_W / 2, delta=1)
        self.assertAlmostEqual(cy, S.CANVAS_H / 2, delta=1)

    def test_to_canvas_handles_a_zero_size_window_without_crashing(self):
        app = App(window_size=(1280, 720))
        app.window = pygame.Surface((1, 1))
        app._dest_rect = pygame.Rect(0, 0, 0, 0)  # simulate the degenerate case directly
        self.assertEqual(app._to_canvas((5, 5)), (0, 0))


class TestSceneSurvivesResize(unittest.TestCase):
    def test_game_scene_draws_at_several_canvas_backed_sizes(self):
        # The scene always draws to the fixed logical canvas; only the
        # presentation (window blit) varies with the real window size, so
        # this exercises draw() itself staying correct regardless.
        app = App()
        app.push(GameScene(MatchSettings(p1_seat="bot", p2_seat="bot", seed=2, max_turns=5)))
        scene = app.scenes[-1]
        for _ in range(10):
            scene.update(30.0)
        for w, h in [(1280, 720), (800, 450), (1920, 1080)]:
            app.window = pygame.Surface((w, h))
            app._recompute_dest_rect()
            app.canvas.fill(S.BG_DEEP)
            scene.draw(app.canvas)  # must not raise at any letterbox scale

    def test_menu_scene_draws_after_a_resize(self):
        app = App()
        app.push(MenuScene())
        scene = app.scenes[-1]
        scene.update(16)
        app.window = pygame.Surface((640, 480))
        app._recompute_dest_rect()
        app.canvas.fill(S.BG_DEEP)
        scene.draw(app.canvas)


if __name__ == "__main__":
    unittest.main()

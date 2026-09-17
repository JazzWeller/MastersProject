"""Regression tests for UX_FIX_PLAN.md F1: every hover/click test must use
*canvas*-space coordinates (`App.mouse_canvas` / `Scene.mouse`), never raw
window pixels, or hovering drifts as soon as the window isn't exactly
1600x900 -- which it never is by default."""

import unittest

import tests.helpers  # noqa: F401

import pygame

from gui import settings as S
from gui.app import App
from gui.engine_bridge import MatchSettings
from gui.scenes.game_scene import GameScene


def _settle(scene, frames=10):
    for _ in range(frames):
        scene.update(0)


def _past_mulligans(scene):
    """Answer mulligans (keep) so the hand is on the board, not under the
    full-screen mulligan review."""
    from keyforge.enums import DecisionKind

    for _ in range(4000):
        scene.update(50)
        d = scene.bridge.pending_decision
        if scene.animator.is_busy or d is None:
            continue
        if d.kind != DecisionKind.MULLIGAN:
            if not scene.bridge.seat_is_bot(d.player) and scene.panel.decision is d:
                return
            continue
        if scene.panel.decision is d:
            scene.panel._submit(False)


class TestMouseCanvasCoordinates(unittest.TestCase):
    def test_to_canvas_scales_correctly_at_a_non_1to1_window_size(self):
        app = App(window_size=(1280, 720))  # 0.8x the 1600x900 canvas
        app.window = pygame.Surface((1280, 720))
        app._recompute_dest_rect()
        # The canvas center should map from the window center.
        cx, cy = app._to_canvas((640, 360))
        self.assertAlmostEqual(cx, S.CANVAS_W / 2, delta=1)
        self.assertAlmostEqual(cy, S.CANVAS_H / 2, delta=1)

    def test_hover_finds_the_right_card_when_window_is_scaled_down(self):
        """This is the exact bug: at the default (non-1:1) window size,
        `pygame.mouse.get_pos()` returns window pixels, but sprites live in
        canvas space. Simulating the mouse at a scaled window position must
        still resolve to the card actually under it, via `self.mouse`."""
        app = App(window_size=(1280, 720))
        app.window = pygame.Surface((1280, 720))
        app._recompute_dest_rect()
        app.push(GameScene(MatchSettings(p1_seat="human", p2_seat="bot", seed=3, max_turns=10)))
        scene = app.scenes[-1]
        _past_mulligans(scene)

        # Pick a face-up, visible sprite (one of the viewer's own hand
        # cards) and compute the WINDOW pixel that sits exactly on it, then
        # simulate the mouse being there.
        iid = next(cs.iid for cs in scene.last_snapshot.cards.values() if cs.face_up and cs.zone == "hand")
        sprite = scene.board.sprites[iid]
        r = app._dest_rect
        win_x = r.x + sprite.x / S.CANVAS_W * r.width
        win_y = r.y + sprite.y / S.CANVAS_H * r.height
        app.mouse_canvas = app._to_canvas((win_x, win_y))
        scene._update_hover()
        self.assertEqual(scene.hover_iid, iid)

    def test_hover_is_wrong_if_you_use_raw_window_pixels_directly(self):
        """Sanity check that the scenario above actually exercises a scaled
        window (i.e. this isn't a no-op at 1:1) -- raw window pixels used
        as canvas coordinates directly would NOT hit the same sprite."""
        app = App(window_size=(1280, 720))
        app.window = pygame.Surface((1280, 720))
        app._recompute_dest_rect()
        app.push(GameScene(MatchSettings(p1_seat="human", p2_seat="bot", seed=3, max_turns=10)))
        scene = app.scenes[-1]
        _past_mulligans(scene)

        iid = next(cs.iid for cs in scene.last_snapshot.cards.values() if cs.face_up and cs.zone == "hand")
        sprite = scene.board.sprites[iid]
        r = app._dest_rect
        win_x = r.x + sprite.x / S.CANVAS_W * r.width
        win_y = r.y + sprite.y / S.CANVAS_H * r.height
        # Using the raw window pixel as if it were already canvas space
        # (the old bug) should generally miss the card at this scale.
        app.mouse_canvas = (win_x, win_y)
        scene._update_hover()
        self.assertNotEqual(scene.hover_iid, iid)

    def test_pile_hover_highlight_uses_canvas_coordinates(self):
        app = App(window_size=(800, 450))  # 0.5x scale
        app.window = pygame.Surface((800, 450))
        app._recompute_dest_rect()
        app.push(GameScene(MatchSettings(p1_seat="bot", p2_seat="bot", seed=2, max_turns=5)))
        scene = app.scenes[-1]
        _settle(scene)

        rect = scene.board.layout.pile_rect(1, "discard")
        r = app._dest_rect
        win_x = r.x + rect.centerx / S.CANVAS_W * r.width
        win_y = r.y + rect.centery / S.CANVAS_H * r.height
        app.mouse_canvas = app._to_canvas((win_x, win_y))
        self.assertTrue(rect.collidepoint(scene.mouse))


if __name__ == "__main__":
    unittest.main()

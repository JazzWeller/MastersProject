"""A loose regression guard, not a benchmark: a populated board (several
creatures/artifacts a side, particles, floating text, a full hand) must
keep rendering comfortably fast. This won't catch small regressions, but
it will catch an accidental O(n^2) or a per-frame surface-recreation bug."""

import time
import unittest

import tests.helpers  # noqa: F401

from gui import settings as S
from gui.app import App
from gui.engine_bridge import MatchSettings
from gui.scenes.game_scene import GameScene

# Generous: real 60 FPS needs < 16.7 ms/frame; this only fails on something
# an order of magnitude off, so it stays stable across slower CI machines.
MAX_AVG_FRAME_MS = 60.0


class TestPerformance(unittest.TestCase):
    def test_populated_board_renders_comfortably_fast(self):
        app = App()
        app.push(GameScene(MatchSettings(p1_seat="bot", p2_seat="bot", seed=1, max_turns=40)))
        scene = app.scenes[-1]

        # Play forward until both sides have some board presence, so the
        # frame includes creature/artifact rows, not just two empty hands.
        frames = 0
        while frames < 20000 and not scene.bridge.is_over:
            frames += 1
            scene.update(20.0)
            app.canvas.fill(S.BG_DEEP)
            scene.draw(app.canvas)
            p1c = len(scene.bridge.game.players[1].play_area.creatures)
            p2c = len(scene.bridge.game.players[2].play_area.creatures)
            if p1c >= 1 and p2c >= 1 and not scene.animator.is_busy:
                break
        self.assertLess(frames, 20000, "never reached a populated board state")

        # A few particle bursts + floating numbers, for a "busy" frame.
        scene.board.particles.emit_sparks(700, 500, n=30)
        scene.board.particles.emit_embers(700, 500, n=30)

        n = 120
        t0 = time.perf_counter()
        for _ in range(n):
            scene.update(16.0)
            app.canvas.fill(S.BG_DEEP)
            scene.draw(app.canvas)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        avg_ms = elapsed_ms / n
        self.assertLess(avg_ms, MAX_AVG_FRAME_MS, f"average frame cost {avg_ms:.2f}ms")


if __name__ == "__main__":
    unittest.main()

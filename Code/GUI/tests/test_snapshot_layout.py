"""Layout: every named rect stays on-canvas and the bands don't collide.
Snapshot: hidden zones never carry face-up content for the wrong viewer."""

import math
import unittest

import tests.helpers  # noqa: F401

import pygame

from bots.random_bot import RandomBot
from keyforge.config import GameConfig
from keyforge.enums import DecisionKind
from keyforge.game import Game

from gui import settings as S
from gui.layout import Layout
from gui.snapshot import build_snapshot

# Card-count conservation only holds at these "boundary" points (and at
# game over) -- mid-resolution a card can be legitimately in transit, e.g.
# Wild Wormhole has pulled a creature off the deck top but a nested
# CHOOSE_FLANK decision is pending before it's placed. sim/simulate.py's
# own check_invariants observes the same restriction.
_BOUNDARY_KINDS = (DecisionKind.CHOOSE_ACTION, DecisionKind.CHOOSE_HOUSE, DecisionKind.TAKE_ARCHIVE)

CANVAS = pygame.Rect(0, 0, S.CANVAS_W, S.CANVAS_H)


class TestLayout(unittest.TestCase):
    def test_named_rects_stay_on_canvas(self):
        for viewer in (1, 2):
            L = Layout(viewer)
            rects = [
                L.hand_rect(1), L.hand_rect(2),
                L.hud_rect(1), L.hud_rect(2),
                L.creature_row_rect(1), L.creature_row_rect(2),
                L.artifact_row_rect(1), L.artifact_row_rect(2),
                L.prompt_rect(), L.action_bar_rect(),
                L.zoom_rect(), L.log_rect(),
            ]
            for kind in ("deck", "discard", "archive", "purged"):
                rects.append(L.pile_rect(1, kind))
                rects.append(L.pile_rect(2, kind))
            for r in rects:
                self.assertTrue(CANVAS.contains(r), (viewer, r))

    def test_your_and_opponent_bands_dont_overlap(self):
        L = Layout(1)
        self.assertLess(L.hand_rect(1).top, S.CANVAS_H)  # bottom hand is below center
        self.assertLess(L.hand_rect(2).bottom, L.creature_row_rect(2).top + 1)  # opp hand above opp creatures
        self.assertLessEqual(L.creature_row_rect(2).bottom, L.creature_row_rect(1).top)  # rows don't cross

    def test_fan_slots_stay_inside_the_rect_horizontally(self):
        L = Layout(1)
        rect = L.hand_rect(1)
        for n in (0, 1, 2, 7, 12):
            slots = L.fan_slots(n, rect, S.HAND_CARD_W, S.HAND_CARD_H)
            self.assertEqual(len(slots), n)
            for x, y, rot in slots:
                self.assertGreaterEqual(x, rect.left - S.HAND_CARD_W)
                self.assertLessEqual(x, rect.right + S.HAND_CARD_W)

    def test_hand_fan_never_goes_off_canvas(self):
        """Regression test for the hand-clipping bug in UX_FIX_PLAN.md F2:
        at every hand size, every card's *rotated* bounding box -- not just
        its center point -- must stay fully on the canvas, for both the
        bottom hand (arc_up=True) and the top hand (arc_up=False)."""
        L = Layout(1)
        for pid, arc_up in ((1, True), (2, False)):
            rect = L.hand_rect(pid)
            for n in range(1, 13):
                for x, y, rot in L.fan_slots(n, rect, S.HAND_CARD_W, S.HAND_CARD_H, arc_up=arc_up):
                    rad = math.radians(abs(rot))
                    half_h = (S.HAND_CARD_W * math.sin(rad) + S.HAND_CARD_H * math.cos(rad)) / 2
                    half_w = (S.HAND_CARD_W * math.cos(rad) + S.HAND_CARD_H * math.sin(rad)) / 2
                    self.assertGreaterEqual(y - half_h, 0, (pid, n, rot))
                    self.assertLessEqual(y + half_h, S.CANVAS_H, (pid, n, rot))
                    self.assertGreaterEqual(x - half_w, -1, (pid, n, rot))
                    self.assertLessEqual(x + half_w, S.CANVAS_W + 1, (pid, n, rot))

    def test_row_slots_dont_explode_for_many_cards(self):
        L = Layout(1)
        rect = L.creature_row_rect(1)
        slots = L.row_slots(12, rect, S.BOARD_CARD_W, S.BOARD_CARD_H)
        self.assertEqual(len(slots), 12)
        xs = [x for x, y in slots]
        self.assertLess(max(xs) - min(xs), S.PLAY_W + S.BOARD_CARD_W)


class TestSnapshotVisibility(unittest.TestCase):
    def test_upgrades_are_always_face_up(self):
        """An upgrade attached to a creature already in play is public
        information, same as the creature -- it must never render as a
        face-down card-back glued to its host (found via a real screenshot,
        see UX_FIX_PLAN.md's second-pass findings)."""
        found_one = False
        for seed in range(30):
            game = Game(GameConfig(decks=("fignor", "igor"), seed=seed, max_turns=25))
            bots = {1: RandomBot(seed=seed), 2: RandomBot(seed=seed + 1000)}
            while not game.is_over:
                d = game.pending_decision
                game.submit(bots[d.player].decide(game.view_for(d.player), d))
                for viewer in (1, 2):
                    snap = build_snapshot(game, viewer)
                    for cs in snap.cards.values():
                        if cs.zone == "upgrade":
                            found_one = True
                            self.assertTrue(cs.face_up, (seed, cs.name))
        self.assertTrue(found_one, "no upgrade ever entered play across 30 seeds -- test didn't exercise anything")

    def test_hidden_zones_never_leak_a_face(self):
        for seed in range(6):
            game = Game(GameConfig(decks=("fignor", "igor"), seed=seed, max_turns=20))
            bots = {1: RandomBot(seed=seed), 2: RandomBot(seed=seed + 1000)}
            while not game.is_over:
                d = game.pending_decision
                choice = bots[d.player].decide(game.view_for(d.player), d)
                game.submit(choice)
                at_boundary = game.is_over or game.pending_decision.kind in _BOUNDARY_KINDS
                for viewer in (1, 2):
                    snap = build_snapshot(game, viewer)
                    for cs in snap.cards.values():
                        if cs.zone == "deck":
                            self.assertFalse(cs.face_up)
                        elif cs.zone in ("hand", "archive"):
                            self.assertEqual(cs.face_up, cs.owner == viewer)
                        else:
                            self.assertTrue(cs.face_up)
                    if at_boundary:
                        total = {1: 0, 2: 0}
                        for cs in snap.cards.values():
                            total[cs.owner] += 1
                        self.assertEqual(total, {1: 36, 2: 36})


if __name__ == "__main__":
    unittest.main()

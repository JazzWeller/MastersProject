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
from gui.layout import Layout, rotated_half_extents
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

    def test_bands_never_overlap(self):
        for viewer in (1, 2):
            bands = Layout(viewer).all_band_rects()
            for i, (name_a, a) in enumerate(bands):
                self.assertTrue(CANVAS.contains(a), name_a)
                for name_b, b in bands[i + 1:]:
                    self.assertFalse(a.colliderect(b), f"{name_a} overlaps {name_b}")
            layout = Layout(viewer)
            for pid in (1, 2):
                c, a = layout.creature_row_rect(pid), layout.artifact_row_rect(pid)
                self.assertFalse(c.colliderect(a), "creature and artifact lanes overlap")
                self.assertTrue(layout.board_rect(pid).contains(c) and layout.board_rect(pid).contains(a))

    def test_every_card_stays_inside_its_own_band(self):
        """Code/PLAYTEST_FIX_PLAN.md L3: for 1-12 cards in every zone, each
        card's rotated bounding box lies entirely inside its own band -- so no
        row can overlap another, which is what the old layout got wrong."""
        for viewer in (1, 2):
            L = Layout(viewer)
            for pid in (1, 2):
                for n in range(1, 13):
                    band = L.hand_rect(pid)
                    w, h = L.hand_card_size(pid)
                    for x, y, rot in L.hand_slots(pid, n):
                        hw, hh = rotated_half_extents(w, h, rot)
                        self._inside(pygame.Rect(x - hw, y - hh, 2 * hw, 2 * hh), band, ("hand", viewer, pid, n))
                    for lane in (L.creature_row_rect(pid), L.artifact_row_rect(pid)):
                        for x, y in L.row_slots(n, lane, S.BOARD_CARD_W, S.BOARD_CARD_H):
                            card = pygame.Rect(0, 0, S.BOARD_CARD_W, S.BOARD_CARD_H)
                            card.center = (round(x), round(y))
                            self._inside(card, lane, ("lane", viewer, pid, n))
                    for side, (x, y) in L.flank_ghost_slots(pid, n - 1).items():
                        card = pygame.Rect(0, 0, S.BOARD_CARD_W, S.BOARD_CARD_H)
                        card.center = (round(x), round(y))
                        self._inside(card, L.creature_row_rect(pid), ("flank", side, n))

    def test_upgrade_tabs_sit_inside_their_host(self):
        L = Layout(1)
        for index in range(3):
            x, y, w, h = L.upgrade_slot(500, 300, index)
            host = pygame.Rect(0, 0, S.BOARD_CARD_W, S.BOARD_CARD_H)
            host.center = (500, 300)
            tab = pygame.Rect(0, 0, w, h)
            tab.center = (round(x), round(y))
            self.assertTrue(host.contains(tab), index)

    def test_piles_stay_in_the_left_column_without_overlapping(self):
        L = Layout(1)
        rects = [L.pile_rect(pid, kind) for pid in (1, 2) for kind in ("deck", "discard", "archive", "purged")]
        for i, a in enumerate(rects):
            self.assertTrue(CANVAS.contains(a))
            self.assertLessEqual(a.right, S.PLAY_X)
            labelled = a.inflate(0, 2 * 14)
            for b in rects[i + 1:]:
                self.assertFalse(labelled.colliderect(b))

    def _inside(self, card: pygame.Rect, band: pygame.Rect, what) -> None:
        self.assertGreaterEqual(card.left, band.left - 1, what)
        self.assertLessEqual(card.right, band.right + 1, what)
        self.assertGreaterEqual(card.top, band.top - 1, what)
        self.assertLessEqual(card.bottom, band.bottom + 1, what)

    def test_row_slots_dont_explode_for_many_cards(self):
        L = Layout(1)
        rect = L.creature_row_rect(1)
        slots = L.row_slots(12, rect, S.BOARD_CARD_W, S.BOARD_CARD_H)
        self.assertEqual(len(slots), 12)


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

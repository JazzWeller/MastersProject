"""Small-feature tests: the decklist viewer, the help overlay, spectate's
"reveal hands" toggle, and Board's you/opponent-vs-Player-N phrasing."""

import unittest

import tests.helpers  # noqa: F401

import pygame

from gui import settings as S
from gui.app import App
from gui.board import Board
from gui.engine_bridge import MatchSettings
from gui.scenes.game_scene import GameScene


def _key(scene, key):
    scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=key, mod=0))


def _click(scene, pos):
    scene.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=1))
    scene.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=pos, button=1))


def _settle(scene, frames=5):
    for _ in range(frames):
        scene.update(0)


class TestBoardPerspective(unittest.TestCase):
    def test_direct_labels_when_not_spectating(self):
        b = Board(None, viewer=1, spectating=False)
        self.assertEqual(b.player_label(1), "You")
        self.assertEqual(b.player_label(2), "Your opponent")
        self.assertTrue(b.is_second_person(1))
        self.assertFalse(b.is_second_person(2))

    def test_third_person_labels_when_spectating(self):
        b = Board(None, viewer=1, spectating=True)
        self.assertEqual(b.player_label(1), "Player 1")
        self.assertEqual(b.player_label(2), "Player 2")
        self.assertFalse(b.is_second_person(1))
        self.assertFalse(b.is_second_person(2))


class TestDecklistViewer(unittest.TestCase):
    def setUp(self):
        self.app = App()
        self.app.push(GameScene(MatchSettings(p1_seat="bot", p2_seat="bot", seed=1, max_turns=5)))
        self.scene = self.app.scenes[-1]
        _settle(self.scene)

    def test_d_opens_the_viewers_own_deck_by_default(self):
        self.assertIsNone(self.scene.decklist_pid)
        _key(self.scene, pygame.K_d)
        self.assertEqual(self.scene.decklist_pid, self.scene.viewer)

    def test_tabs_switch_player_and_escape_closes(self):
        _key(self.scene, pygame.K_d)
        w, h = 780, 600
        rect = pygame.Rect((S.CANVAS_W - w) // 2, (S.CANVAS_H - h) // 2, w, h)
        tab2 = pygame.Rect(rect.left + 132, rect.top + 10, 110, 28)
        _click(self.scene, tab2.center)
        self.assertEqual(self.scene.decklist_pid, 2)

        self.scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0))
        self.assertIsNone(self.scene.decklist_pid)

    def test_decklist_contents_match_the_players_full_pool(self):
        pid = self.scene.viewer
        expected = {c.instance_id for c in self.scene.bridge.game.players[pid].all_cards}
        self.assertEqual(len(expected), 36)
        # Rendering shouldn't crash and should draw from exactly that pool.
        _key(self.scene, pygame.K_d)
        canvas = pygame.Surface((S.CANVAS_W, S.CANVAS_H))
        self.scene.draw(canvas)  # smoke test: no exception


class TestHelpOverlay(unittest.TestCase):
    def setUp(self):
        self.app = App()
        self.app.push(GameScene(MatchSettings(p1_seat="bot", p2_seat="bot", seed=1, max_turns=5)))
        self.scene = self.app.scenes[-1]
        _settle(self.scene)

    def test_h_and_slash_toggle_help(self):
        self.assertFalse(self.scene.show_help)
        _key(self.scene, pygame.K_h)
        self.assertTrue(self.scene.show_help)
        _key(self.scene, pygame.K_h)
        self.assertFalse(self.scene.show_help)
        _key(self.scene, pygame.K_SLASH)
        self.assertTrue(self.scene.show_help)

    def test_help_blocks_pile_clicks(self):
        _key(self.scene, pygame.K_h)
        self.assertTrue(self.scene.show_help)
        pile_rect = self.scene.board.layout.pile_rect(1, "deck")
        _click(self.scene, pile_rect.center)
        self.assertIsNone(self.scene.browsing)  # click was swallowed by the help overlay
        self.assertTrue(self.scene.show_help)

    def test_escape_closes_help_without_leaving_the_scene(self):
        _key(self.scene, pygame.K_h)
        self.scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0))
        self.assertFalse(self.scene.show_help)
        self.assertEqual(len(self.app.scenes), 1)  # still on the GameScene


class TestRevealHands(unittest.TestCase):
    def _hand_face_states(self, scene, pid):
        states = set()
        for cs in scene.last_snapshot.zone_cards(pid, "hand"):
            sprite = scene.board.sprites[cs.iid]
            states.add(sprite.face_up)
        return states

    def test_reveal_hands_only_works_with_no_human_seat(self):
        app = App()
        app.push(GameScene(MatchSettings(p1_seat="human", p2_seat="bot", seed=1, max_turns=5)))
        scene = app.scenes[-1]
        _settle(scene)
        self.assertFalse(scene.reveal_hands)
        _key(scene, pygame.K_r)
        self.assertFalse(scene.reveal_hands, "R should be a no-op with a human seated")

    def test_reveal_hands_shows_the_opponents_hand_face_up(self):
        app = App()
        app.push(GameScene(MatchSettings(p1_seat="bot", p2_seat="bot", seed=1, max_turns=5)))
        scene = app.scenes[-1]
        _settle(scene)
        opp = 3 - scene.viewer

        # Before: the opponent's hand is face down (normal visibility rules).
        self.assertEqual(self._hand_face_states(scene, opp), {False})

        _key(scene, pygame.K_r)
        self.assertTrue(scene.reveal_hands)
        self.assertEqual(self._hand_face_states(scene, opp), {True})
        for cs in scene.last_snapshot.zone_cards(opp, "hand"):
            sprite = scene.board.sprites[cs.iid]
            self.assertEqual(sprite.image_path, cs.image)

        # Toggle back off: reverts to the true (hidden) visibility.
        _key(scene, pygame.K_r)
        self.assertFalse(scene.reveal_hands)
        self.assertEqual(self._hand_face_states(scene, opp), {False})

    def test_own_hand_is_unaffected_by_reveal_hands(self):
        app = App()
        app.push(GameScene(MatchSettings(p1_seat="bot", p2_seat="bot", seed=1, max_turns=5)))
        scene = app.scenes[-1]
        _settle(scene)
        _key(scene, pygame.K_r)
        self.assertEqual(self._hand_face_states(scene, scene.viewer), {True})  # already public to the viewer


if __name__ == "__main__":
    unittest.main()

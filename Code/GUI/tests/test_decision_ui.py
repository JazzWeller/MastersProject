"""Regression tests for UX_FIX_PLAN.md A3/B3/D7: the action chooser never
shows the same card's art twice, MULLIGAN always gets the dedicated review
screen (not the generic modal), and a Discard / premature End Turn always
requires a second click before it actually submits."""

import unittest

import tests.helpers  # noqa: F401

from bots.random_bot import RandomBot
from keyforge.actions import DiscardCard, EndTurn
from keyforge.config import GameConfig
from keyforge.enums import DecisionKind
from keyforge.game import Game

from gui.board import Board
from gui.decision.panel import DecisionPanel


class TestNoDuplicateThumbnails(unittest.TestCase):
    """Fuzzes real games, driving every human-reachable decision through the
    panel exactly as the GameScene does, and checks the invariant that
    broke before (two cells showing the same physical card with nothing to
    tell them apart)."""

    def test_chooser_and_modal_never_duplicate_a_cards_art(self):
        for seed in range(10):
            game = Game(GameConfig(decks=("fignor", "igor"), seed=seed, max_turns=15))
            bots = {1: RandomBot(seed=seed), 2: RandomBot(seed=seed + 500)}
            board = Board(_FakeAssets(), viewer=1)
            panel = DecisionPanel()

            while not game.is_over:
                d = game.pending_decision
                from gui.snapshot import build_snapshot

                board.snapshot = build_snapshot(game, 1)
                for cs in board.snapshot.cards.values():
                    board.sprite_for(cs.iid)  # ensure a sprite exists for every card
                view = game.view_for(d.player)
                panel.on_enter(d, view, board)

                # The action chooser: every option's card, if any, must be
                # the SAME single card the chooser is scoped to (that's the
                # whole point) -- never a grid of duplicates.
                if panel.chooser_iid is not None:
                    for opt in panel.chooser_opts:
                        card = getattr(opt, "card", opt)
                        self.assertEqual(getattr(card, "instance_id", None), panel.chooser_iid)

                # The generic modal: if it renders as a card grid, every
                # cell must be a distinct physical card.
                if panel.modal_open and panel._modal_is_grid():
                    ids = [r.card.instance_id for r in panel.modal_rows]
                    self.assertEqual(len(ids), len(set(ids)), (seed, d.kind))

                choice = bots[d.player].decide(view, d)
                game.submit(choice)

    def test_mulligan_always_opens_the_dedicated_screen(self):
        game = Game(GameConfig(decks=("fignor", "igor"), seed=1, max_turns=5))
        self.assertEqual(game.pending_decision.kind, DecisionKind.MULLIGAN)
        board = Board(_FakeAssets(), viewer=1)
        from gui.snapshot import build_snapshot

        board.snapshot = build_snapshot(game, 1)
        panel = DecisionPanel()
        panel.on_enter(game.pending_decision, game.view_for(1), board)
        self.assertTrue(panel.mulligan_open)
        self.assertFalse(panel.modal_open)


class TestConfirmBeforeSubmit(unittest.TestCase):
    def _panel_at_choose_action(self, seed=1):
        game = Game(GameConfig(decks=("fignor", "igor"), seed=seed, max_turns=8))
        bot1, bot2 = RandomBot(seed=seed), RandomBot(seed=seed + 1)
        # Fast-forward through mulligan/house-choice into a CHOOSE_ACTION.
        while game.pending_decision.kind != DecisionKind.CHOOSE_ACTION:
            d = game.pending_decision
            game.submit((bot1 if d.player == 1 else bot2).decide(game.view_for(d.player), d))
        board = Board(_FakeAssets(), viewer=game.pending_decision.player)
        from gui.snapshot import build_snapshot

        board.snapshot = build_snapshot(game, board.viewer)
        for cs in board.snapshot.cards.values():
            board.sprite_for(cs.iid)
        panel = DecisionPanel()
        panel.on_enter(game.pending_decision, game.view_for(board.viewer), board)
        return panel

    def test_discard_needs_two_picks(self):
        for seed in range(15):
            panel = self._panel_at_choose_action(seed)
            discard_opt = next((o for o in panel.decision.options if isinstance(o, DiscardCard)), None)
            if discard_opt is None:
                continue
            panel._pick(discard_opt)
            self.assertIsNone(panel.result, "a single click must not submit a discard")
            self.assertIs(panel._armed_action, discard_opt)
            panel._pick(discard_opt)
            self.assertIsNotNone(panel.result)
            return
        self.skipTest("no DiscardCard option turned up in the fuzzed seeds")

    def test_end_turn_needs_two_picks_when_other_actions_exist(self):
        for seed in range(15):
            panel = self._panel_at_choose_action(seed)
            end_turn = next((o for o in panel.decision.options if isinstance(o, EndTurn)), None)
            if end_turn is None or len(panel.decision.options) <= 1:
                continue
            panel._pick(end_turn)
            self.assertIsNone(panel.result, "ending a turn with other actions available must arm, not submit")
            panel._pick(end_turn)
            self.assertIsNotNone(panel.result)
            return
        self.skipTest("no CHOOSE_ACTION with EndTurn + other options turned up in the fuzzed seeds")

    def test_end_turn_submits_immediately_when_its_the_only_option(self):
        game = Game(GameConfig(decks=("fignor", "igor"), seed=1, max_turns=1))
        bot1, bot2 = RandomBot(seed=1), RandomBot(seed=2)
        while game.pending_decision.kind != DecisionKind.CHOOSE_ACTION or len(game.pending_decision.options) > 1:
            d = game.pending_decision
            game.submit((bot1 if d.player == 1 else bot2).decide(game.view_for(d.player), d))
            if game.is_over:
                self.skipTest("game ended before an End-Turn-only decision turned up")
        board = Board(_FakeAssets(), viewer=game.pending_decision.player)
        from gui.snapshot import build_snapshot

        board.snapshot = build_snapshot(game, board.viewer)
        panel = DecisionPanel()
        panel.on_enter(game.pending_decision, game.view_for(board.viewer), board)
        end_turn = panel.decision.options[0]
        panel._pick(end_turn)
        self.assertIsNotNone(panel.result)


class _FakeAssets:
    """A minimal stand-in so DecisionPanel/Board never need a real display
    or font system for these logic-only tests."""

    def play(self, *a, **k):
        pass


if __name__ == "__main__":
    unittest.main()

"""Regression tests for UX_FIX_PLAN.md A3/B3/D7: the action chooser never
shows the same card's art twice, MULLIGAN always gets the dedicated review
screen (not the generic modal), and a Discard / premature End Turn always
requires a second click before it actually submits."""

import unittest

import tests.helpers  # noqa: F401

from bots.random_bot import RandomBot
from keyforge.actions import DiscardCard, EndTurn, PlayCard
from keyforge.cards.card import Card
from keyforge.cards.card_data import get_card_def
from keyforge.config import GameConfig
from keyforge.enums import DecisionKind, House
from keyforge.game import Game

from gui.board import Board
from gui.decision.panel import DecisionPanel
from gui.snapshot import build_snapshot


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


class TestReadyAndFightSubPrompts(unittest.TestCase):
    """Phase 3 Milestone E: cards like Anger play out as two sequential,
    generic CHOOSE_CARDS decisions (choose a friendly creature, then --
    from inside Game.ready_and_fight -- choose its fight target). Confirms
    the panel needs no special "sub-prompt" widget for this: both steps
    are answerable the same way any other on-board card choice is,
    by picking the option scoped to the relevant sprite."""

    def _make(self, name, owner):
        return Card(get_card_def(name), owner)

    def _advance_to_choose_action(self, game) -> None:
        while game.pending_decision.kind != DecisionKind.CHOOSE_ACTION:
            d = game.pending_decision
            if d.kind == DecisionKind.TAKE_ARCHIVE:
                game.submit(False)
            else:
                game.submit(d.options[0])

    def test_anger_two_step_decision_resolves_through_the_panel(self):
        game = Game(GameConfig(decks=("fignor", "igor"), seed=1, max_turns=8))
        game.submit(False)
        game.submit(False)
        self._advance_to_choose_action(game)  # player 1's first CHOOSE_ACTION, turn 1

        # First-turn rule restricts only the FIRST player's very first turn
        # to one play-or-discard -- sidestep it entirely (rather than fight
        # it) by ending that turn unused and doing the actual setup on
        # player 2's first CHOOSE_ACTION instead.
        end_turn = next(o for o in game.pending_decision.options if isinstance(o, EndTurn))
        game.submit(end_turn)
        self._advance_to_choose_action(game)

        stale = game.pending_decision  # options computed from the pre-mutation state
        pid = stale.player
        other = 3 - pid
        player = game.players[pid]

        # Mutate play areas/hand/house *before* forcing a fresh CHOOSE_ACTION
        # (a Decision's `.options` are a frozen snapshot -- mutating state
        # after one is issued never changes it, so a throwaway Discard from
        # the stale decision is used purely to make the main loop recompute
        # `_legal_actions` against the new state below).
        player.selected_house = House.BROBNAR
        # Two friendly creatures and two enemy creatures so neither of
        # Anger's two decisions auto-skips as a single-legal-option choice.
        attacker1 = self._make("Charette", pid)
        attacker2 = self._make("Snudge", pid)
        player.play_area.add_creature(attacker1)
        player.play_area.add_creature(attacker2)
        target1 = self._make("Snudge", other)
        target2 = self._make("Mother", other)  # not Truebaru: it has taunt, which would protect target1
        game.players[other].play_area.add_creature(target1)
        game.players[other].play_area.add_creature(target2)
        anger = self._make("Anger", pid)
        player.hand.add(anger)

        throwaway = next(o for o in stale.options if isinstance(o, DiscardCard))
        game.submit(throwaway)  # resolves under the old house; refreshes CHOOSE_ACTION under the new one

        board = Board(_FakeAssets(), viewer=pid)
        board.snapshot = build_snapshot(game, pid)
        for cs in board.snapshot.cards.values():
            board.sprite_for(cs.iid)
        panel = DecisionPanel()

        fresh = game.pending_decision
        self.assertEqual(fresh.kind, DecisionKind.CHOOSE_ACTION)
        game.submit(next(o for o in fresh.options if isinstance(o, PlayCard) and o.card is anger))

        d1 = game.pending_decision
        self.assertEqual(d1.kind, DecisionKind.CHOOSE_CARDS)
        self.assertCountEqual(d1.options, [attacker1, attacker2])
        board.snapshot = build_snapshot(game, pid)
        panel.on_enter(d1, game.view_for(pid), board)
        self.assertIn(attacker1.instance_id, panel.card_option_map)
        panel._pick(panel.card_option_map[attacker1.instance_id][0])
        self.assertIsNotNone(panel.result)
        game.submit(panel.result)

        d2 = game.pending_decision
        self.assertEqual(d2.kind, DecisionKind.CHOOSE_CARDS)
        self.assertCountEqual(d2.options, [target1, target2])
        board.snapshot = build_snapshot(game, pid)
        panel.on_enter(d2, game.view_for(pid), board)
        self.assertIn(target1.instance_id, panel.card_option_map)
        panel._pick(panel.card_option_map[target1.instance_id][0])
        self.assertIsNotNone(panel.result)
        game.submit(panel.result)

        self.assertTrue(attacker1.Exhausted)  # readied, then fought -> exhausted again
        self.assertEqual(game.pending_decision.kind, DecisionKind.CHOOSE_ACTION)


class _FakeAssets:
    """A minimal stand-in so DecisionPanel/Board never need a real display
    or font system for these logic-only tests."""

    def play(self, *a, **k):
        pass


if __name__ == "__main__":
    unittest.main()

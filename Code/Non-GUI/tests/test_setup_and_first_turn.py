import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bots.random_bot import RandomBot
from keyforge.actions import DiscardCard, EndTurn, PlayCard
from keyforge.config import GameConfig
from keyforge.enums import DecisionKind
from keyforge.game import Game


def auto_answer_setup(game, mulligan1=False, mulligan2=False):
    # first mulligan decision
    game.submit(mulligan1)
    game.submit(mulligan2)


class TestSetup(unittest.TestCase):
    def test_deal_and_hand_sizes(self):
        game = Game(GameConfig(decks=("fignor", "igor"), first_player=1, seed=1))
        self.assertEqual(game.pending_decision.kind, DecisionKind.MULLIGAN)
        self.assertEqual(len(game.players[1].hand), 7)
        self.assertEqual(len(game.players[2].hand), 6)
        auto_answer_setup(game)
        self.assertEqual(game.pending_decision.kind, DecisionKind.CHOOSE_HOUSE)

    def test_mulligan_draws_one_fewer(self):
        game = Game(GameConfig(decks=("fignor", "igor"), first_player=1, seed=2))
        game.submit(True)  # p1 mulligans
        self.assertEqual(len(game.players[1].hand), 6)
        game.submit(False)  # p2 does not
        self.assertEqual(len(game.players[2].hand), 6)

    def test_first_turn_rule_limits_to_one_hand_play(self):
        game = Game(GameConfig(decks=("fignor", "igor"), first_player=1, seed=3))
        auto_answer_setup(game)
        self.assertEqual(game.pending_decision.kind, DecisionKind.CHOOSE_HOUSE)
        house = game.pending_decision.options[0]
        game.submit(house)
        # no archive yet, so next is CHOOSE_ACTION
        self.assertEqual(game.pending_decision.kind, DecisionKind.CHOOSE_ACTION)
        options = game.pending_decision.options
        play_or_discard = [o for o in options if isinstance(o, (PlayCard, DiscardCard))]
        self.assertTrue(len(play_or_discard) >= 1)
        # pick the first available play/discard action of our chosen house
        chosen = play_or_discard[0]
        game.submit(chosen)
        # auto-resolve any nested decisions (flank choice, targets, ...) caused by
        # that play, using a bot, until we're back at a CHOOSE_ACTION for player 1
        bot = RandomBot(seed=0)
        guard = 0
        while game.pending_decision is not None and game.pending_decision.kind != DecisionKind.CHOOSE_ACTION:
            d = game.pending_decision
            game.submit(bot.decide(game.view_for(d.player), d))
            guard += 1
            if guard > 20:
                self.fail("too many nested decisions")
        remaining_hand_actions = [
            o for o in game.pending_decision.options if isinstance(o, (PlayCard, DiscardCard))
        ]
        self.assertEqual(remaining_hand_actions, [])


class TestOpeningHandWithChains(unittest.TestCase):
    """Starting chains (Reversal/Adaptive bid winners) shrink the opening
    hand exactly like the normal draw-step penalty: 1 fewer card per 6
    chains (rounded up), then shed one chain -- see PHASE_2_PLAN.md."""

    def _hand_and_chains(self, chains, base_size=7):
        game = Game(GameConfig(
            decks=("fignor", "igor"), first_player=1, seed=1, starting_chains={1: chains, 2: 0},
        ))
        return len(game.players[1].hand), game.players[1].chains

    def test_zero_chains_is_unaffected(self):
        hand, chains = self._hand_and_chains(0)
        self.assertEqual((hand, chains), (7, 0))

    def test_one_chain(self):
        hand, chains = self._hand_and_chains(1)
        self.assertEqual((hand, chains), (6, 0))

    def test_six_chains_exact_boundary(self):
        hand, chains = self._hand_and_chains(6)
        self.assertEqual((hand, chains), (6, 5))

    def test_seven_chains_past_boundary(self):
        hand, chains = self._hand_and_chains(7)
        self.assertEqual((hand, chains), (5, 6))

    def test_thirteen_chains(self):
        hand, chains = self._hand_and_chains(13)
        self.assertEqual((hand, chains), (4, 12))

    def test_max_24_chains(self):
        hand, chains = self._hand_and_chains(24)
        self.assertEqual((hand, chains), (3, 23))

    def test_second_player_base_size_six(self):
        game = Game(GameConfig(
            decks=("fignor", "igor"), first_player=1, seed=1, starting_chains={1: 0, 2: 7},
        ))
        self.assertEqual(len(game.players[2].hand), 4)  # 6 - ceil(7/6)=2
        self.assertEqual(game.players[2].chains, 6)

    def test_shed_chain_logged_with_opening_hand_source(self):
        game = Game(GameConfig(
            decks=("fignor", "igor"), first_player=1, seed=1, starting_chains={1: 6, 2: 0},
        ))
        ev = [e for e in game.log.events if e.kind == "shed_chain"]
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0].data["source"], "opening hand")
        self.assertEqual((ev[0].data["fewer"], ev[0].data["total"]), (1, 5))

    def test_mulligan_draws_one_fewer_and_does_not_shed_again(self):
        game = Game(GameConfig(
            decks=("fignor", "igor"), first_player=1, seed=1, starting_chains={1: 7, 2: 0},
        ))
        self.assertEqual(len(game.players[1].hand), 5)
        self.assertEqual(game.players[1].chains, 6)
        game.submit(True)  # p1 mulligans
        self.assertEqual(len(game.players[1].hand), 4)  # one fewer than the chain-reduced hand
        self.assertEqual(game.players[1].chains, 6)  # no second shed
        shed_events = [e for e in game.log.events if e.kind == "shed_chain"]
        self.assertEqual(len(shed_events), 1)


if __name__ == "__main__":
    unittest.main()

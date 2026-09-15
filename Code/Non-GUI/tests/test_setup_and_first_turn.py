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


if __name__ == "__main__":
    unittest.main()

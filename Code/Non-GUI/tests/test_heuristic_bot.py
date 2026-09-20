"""Targeted tests for HeuristicBot's handling of the newer decision kinds
(Code/PHASE_2_PLAN.md Milestone C/E: bots must handle every decision kind)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.helpers import new_game, put_creature

from bots.heuristic_bot import HeuristicBot
from keyforge.decision import Decision
from keyforge.enums import DecisionKind


class TestHeuristicBotChooseNumber(unittest.TestCase):
    def test_prefers_the_power_that_destroys_more_enemies_than_friendlies(self):
        game = new_game()
        put_creature(game, 1, "Charette")  # power 4, friendly
        put_creature(game, 2, "Doc Bookton")  # power 5, enemy
        put_creature(game, 2, "Truebaru")  # power 7, enemy
        bot = HeuristicBot(seed=1)
        view = game.view_for(1)
        decision = Decision(player=1, kind=DecisionKind.CHOOSE_NUMBER, prompt="Dance of Doom: choose a number", options=[4, 5, 7])
        choice = bot.decide(view, decision)
        # 4 kills 1 friendly for 0 enemies (score -1); 5 and 7 each kill 1
        # enemy for 0 friendlies (score +1) -- either is an equally good pick.
        self.assertIn(choice, (5, 7))

    def test_does_not_crash_with_no_matching_creatures_in_play(self):
        game = new_game()
        bot = HeuristicBot(seed=1)
        view = game.view_for(1)
        decision = Decision(player=1, kind=DecisionKind.CHOOSE_NUMBER, prompt="Dance of Doom: choose a number", options=[3, 6])
        choice = bot.decide(view, decision)
        self.assertIn(choice, (3, 6))


if __name__ == "__main__":
    unittest.main()

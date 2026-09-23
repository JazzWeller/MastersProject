"""Regression for the Milestone L `_legal_actions`/`why_not_playable` gate
refactor (Code/AGENT_INTERFACE_PLAN.md, "one play gate per decision, not
per card"): `_legal_actions` now decides PlayCard legality via a fast,
gate-based boolean check instead of calling `why_not_playable` per card,
and `why_not_playable` itself now delegates its own "yes" case to that same
fast check. `why_not_playable`'s own docstring guarantee -- that legality
and its explanation can never disagree -- is exactly what this checks,
across many random games rather than by construction alone.
"""

import unittest

from bots.random_bot import RandomBot
from keyforge.actions import PlayCard
from keyforge.config import GameConfig
from keyforge.game import Game


class TestLegalActionsAgreesWithWhyNotPlayable(unittest.TestCase):
    def test_every_hand_card_agrees_at_every_choose_action_decision(self):
        for seed in range(8):
            game = Game(GameConfig(decks=("fignor", "igor"), seed=seed, max_turns=80))
            bots = {1: RandomBot(seed=seed), 2: RandomBot(seed=seed + 1)}
            checked = 0
            while not game.is_over:
                d = game.pending_decision
                if d.kind.name == "CHOOSE_ACTION":
                    pid = d.player
                    player = game.players[pid]
                    playable_via_actions = {a.card for a in d.options if isinstance(a, PlayCard)}
                    for card in player.hand.cards():
                        is_playable = game.why_not_playable(pid, card) is None
                        self.assertEqual(
                            card in playable_via_actions, is_playable,
                            f"seed {seed}: {card.name} disagreement between _legal_actions and why_not_playable",
                        )
                        checked += 1
                game.submit(bots[d.player].decide(game.view_for(d.player), d))
            self.assertGreater(checked, 0, f"seed {seed}: no CHOOSE_ACTION decisions with cards in hand were seen")

    def test_why_not_playable_never_raises_across_many_random_games(self):
        """The AssertionError path in why_not_playable (its own internal
        consistency check) must never actually fire."""
        for seed in range(20):
            game = Game(GameConfig(decks=("fignor", "igor"), seed=seed, max_turns=60))
            bots = {1: RandomBot(seed=seed), 2: RandomBot(seed=seed + 1)}
            while not game.is_over:
                d = game.pending_decision
                if d.kind.name == "CHOOSE_ACTION":
                    pid = d.player
                    for card in game.players[pid].hand.cards():
                        game.why_not_playable(pid, card)  # must not raise
                game.submit(bots[d.player].decide(game.view_for(d.player), d))


if __name__ == "__main__":
    unittest.main()

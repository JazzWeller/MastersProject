"""Regression for `Game.submit_index`/`Match.submit_index` (Code/
AGENT_INTERFACE_PLAN.md, Milestone L: "submit_index(i) fast path for the
driver, replay and apply"). `submit()` (used for untrusted/human input)
and `submit_index()` (used for trusted, already-encoded choices) must
always agree on where a game/match ends up.
"""

import unittest

from bots.random_bot import RandomBot
from keyforge.config import GameConfig
from keyforge.enums import DecisionKind
from keyforge.game import Game
from keyforge.match import Match, MatchConfig, match_replay
from keyforge.replay import encode_choice, replay


def _play_random(config, n_decisions=None):
    game = Game(config)
    bots = {1: RandomBot(seed=config.seed), 2: RandomBot(seed=(config.seed or 0) + 1)}
    count = 0
    while not game.is_over:
        if n_decisions is not None and count >= n_decisions:
            break
        d = game.pending_decision
        game.submit(bots[d.player].decide(game.view_for(d.player), d))
        count += 1
    return game


class TestGameSubmitIndex(unittest.TestCase):
    def test_submit_index_matches_submit_for_a_single_choice_decision(self):
        config = GameConfig(decks=("fignor", "igor"), seed=1, max_turns=20)
        via_submit = _play_random(config, n_decisions=15)
        via_index = replay(config, via_submit.choice_record)
        self.assertEqual(via_submit.choice_record, via_index.choice_record)
        self.assertEqual(via_submit.state_hash(), via_index.state_hash())

    def test_submit_index_matches_submit_for_multi_select_decisions(self):
        """Finds and exercises at least one CHOOSE_CARDS/ORDER_EFFECTS
        decision (encoded as a list of indices, not a bare int)."""
        config = GameConfig(decks=("fignor", "igor"), seed=3, max_turns=100)
        game = _play_random(config)
        list_kinds = {DecisionKind.CHOOSE_CARDS, DecisionKind.ORDER_EFFECTS}
        # decode_choice's own record shape: a list entry only for these kinds.
        saw_a_list_choice = any(isinstance(e, list) for e in game.choice_record)
        self.assertTrue(saw_a_list_choice, "this seed never hit a CHOOSE_CARDS/ORDER_EFFECTS decision")
        via_index = replay(config, game.choice_record)
        self.assertEqual(game.state_hash(), via_index.state_hash())

    def test_submit_index_stores_an_independent_copy_of_a_list_choice(self):
        """`choice_record` must never alias the caller's own record list --
        mutating the input after the fact must not corrupt the replay."""
        config = GameConfig(decks=("fignor", "igor"), seed=3, max_turns=100)
        original = _play_random(config)
        record = list(original.choice_record)
        game = replay(config, record)
        for entry in record:
            if isinstance(entry, list) and entry:
                entry.append(999)
                break
        self.assertNotIn(999, [x for e in game.choice_record if isinstance(e, list) for x in e])

    def test_apply_uses_submit_index_and_agrees_with_a_manual_replay(self):
        config = GameConfig(decks=("fignor", "igor"), seed=5, max_turns=30)
        game = _play_random(config, n_decisions=20)
        fork = game.fork()
        # fork() already replayed the record; re-`apply`ing an empty slice
        # is a no-op, so instead check a partial apply matches a partial replay.
        partial_record = game.choice_record[:10]
        via_apply = Game(config).apply(partial_record)
        via_replay = replay(config, game.choice_record, upto=10)
        self.assertEqual(via_apply.state_hash(), via_replay.state_hash())

    def test_submit_index_raises_like_submit_when_the_game_is_over(self):
        config = GameConfig(decks=("fignor", "igor"), seed=1, max_turns=1)
        game = _play_random(config)
        while not game.is_over:
            d = game.pending_decision
            game.submit(RandomBot(seed=1).decide(game.view_for(d.player), d))
        with self.assertRaises(RuntimeError):
            game.submit_index(0)


class TestMatchSubmitIndex(unittest.TestCase):
    def test_match_replay_uses_submit_index_and_matches_a_live_match(self):
        config = MatchConfig(format="adaptive", decks=("fignor", "igor"), seed=11, max_turns=60)
        match = Match(config)
        bots = {1: RandomBot(seed=1), 2: RandomBot(seed=2)}
        guard = 0
        while not match.is_over:
            d = match.pending_decision
            match.submit(bots[d.player].decide(match.view_for(d.player), d))
            guard += 1
            self.assertLess(guard, 5000)

        replayed = match_replay(config, match.choice_record)
        self.assertEqual(match.result, replayed.result)
        self.assertEqual(match.choice_record, replayed.choice_record)

    def test_match_submit_index_matches_submit_for_a_match_level_decision(self):
        config = MatchConfig(format="archon", decks=("fignor", "igor"), seed=13, max_turns=20)
        match = Match(config)
        d = match.pending_decision
        encoded = encode_choice(d, d.options[0])
        match.submit_index(encoded)
        self.assertEqual(match.choice_record, [encoded])


if __name__ == "__main__":
    unittest.main()

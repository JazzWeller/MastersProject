"""Tests for the Reversal/Adaptive match formats (Code/PHASE_2_PLAN.md v1.2/v1.3)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bots.heuristic_bot import HeuristicBot
from bots.random_bot import RandomBot
from keyforge.enums import DecisionKind
from keyforge.match import GameRecord, Match, MatchConfig, match_config_from_dict, match_config_to_dict, match_replay


class _ScriptedMatch(Match):
    """A Match whose games are pre-scripted outcomes instead of real
    gameplay, so match-level orchestration (scoring, first-player and bid
    flow) can be tested independently of full game simulation. Each call to
    `_play_game` still records the *actual* decks/first_player/chains it was
    given, so callers can assert on what the real orchestration computed."""

    def __init__(self, config, winners):
        self._winners = list(winners)
        self.calls = []
        super().__init__(config)

    def _play_game(self, decks, first_player, starting_chains=None):
        self.calls.append(
            {"decks": tuple(decks), "first_player": first_player, "starting_chains": dict(starting_chains or {})}
        )
        winner = self._winners.pop(0)
        record = GameRecord(
            config=None,
            winner=winner,
            turns=10,
            reason="3 keys" if winner else "turn limit",
            seat_decks={1: decks[0], 2: decks[1]},
            final_chains={1: (starting_chains or {}).get(1, 0), 2: (starting_chains or {}).get(2, 0)},
            final_keys={1: 3 if winner == 1 else 0, 2: 3 if winner == 2 else 0},
        )
        self.games.append(record)
        self.current_game = None
        self._score_game(record)
        return
        yield  # pragma: no cover - makes this a generator function


class TestReversal(unittest.TestCase):
    def test_single_game_with_swapped_decks_and_ownership(self):
        match = Match(MatchConfig(format="reversal", decks=("fignor", "igor"), seed=1))
        bots = {1: HeuristicBot(seed=1), 2: HeuristicBot(seed=2)}
        guard = 0
        while not match.is_over:
            d = match.pending_decision
            match.submit(bots[d.player].decide(match.view_for(d.player), d))
            guard += 1
            self.assertLess(guard, 20000, "match did not terminate")
        self.assertEqual(len(match.games), 1)
        g = match.games[0]
        self.assertEqual(g.seat_decks, {1: "igor", 2: "fignor"})
        self.assertEqual(match.result["format"], "reversal")
        self.assertIn(match.result["winner"], (1, 2))
        self.assertIsNone(match.result["bid"])


class TestAdaptiveScoring(unittest.TestCase):
    def _drive_first_player(self, match, answer="first"):
        d = match.pending_decision
        self.assertEqual(d.kind, DecisionKind.CHOOSE_FIRST_PLAYER)
        match.submit(answer)

    def test_two_zero_ends_after_two_games_no_bid(self):
        match = _ScriptedMatch(MatchConfig(format="adaptive", decks=("fignor", "igor"), seed=1), winners=[1, 1])
        self._drive_first_player(match)  # game 1's loser (P2) chooses
        self.assertTrue(match.is_over)
        self.assertEqual(len(match.games), 2)
        self.assertEqual(match.result["winner"], 1)
        self.assertEqual(match.result["games"], [1, 1])
        self.assertIsNone(match.result["bid"])
        self.assertEqual(match.score, {1: 2, 2: 0})
        # game 2 used swapped decks
        self.assertEqual(match.calls[1]["decks"], ("igor", "fignor"))

    def test_loser_of_previous_game_chooses_first_player(self):
        match = _ScriptedMatch(MatchConfig(format="adaptive", decks=("fignor", "igor"), seed=1), winners=[1, 1])
        d = match.pending_decision
        self.assertEqual(d.player, 2)  # P2 lost game 1
        match.submit("first")
        self.assertEqual(match.calls[1]["first_player"], 2)

        match2 = _ScriptedMatch(MatchConfig(format="adaptive", decks=("fignor", "igor"), seed=1), winners=[1, 1])
        match2.submit("second")
        self.assertEqual(match2.calls[1]["first_player"], 1)  # loser chose second -> the other seat goes first

    def test_split_result_goes_to_bidding(self):
        # Game 1: seat 1 (deck "fignor") wins. Game 2 decks are swapped, and
        # seat 2 (holding "fignor" in game 2) wins -- "fignor" won both games.
        match = _ScriptedMatch(MatchConfig(format="adaptive", decks=("fignor", "igor"), seed=1), winners=[1, 2])
        self._drive_first_player(match)  # between games 1 and 2
        d = match.pending_decision
        self.assertEqual(d.kind, DecisionKind.BID_CHAINS)
        self.assertEqual(d.player, 2)  # non-owner (owner of "fignor" is seat 1) bids first

    def test_bid_owner_keeps_deck_at_zero_if_opponent_passes(self):
        match = _ScriptedMatch(MatchConfig(format="adaptive", decks=("fignor", "igor"), seed=1), winners=[1, 2, 1])
        self._drive_first_player(match)
        d = match.pending_decision
        self.assertIn("pass", d.options)
        match.submit("pass")  # opponent (P2) passes immediately
        self._drive_first_player(match)  # between games 2 and 3
        self.assertTrue(match.is_over)
        self.assertEqual(match.bid.owner, 1)
        self.assertEqual(match.bid.winner, 1)
        self.assertEqual(match.bid.amount, 0)
        self.assertEqual(match.bid.deck, "fignor")
        self.assertEqual(match.calls[2]["starting_chains"], {1: 0, 2: 0})
        self.assertEqual(match.calls[2]["decks"], ("fignor", "igor"))  # winner(1) keeps fignor, loser(2) gets igor

    def test_bid_alternates_and_winner_pays_final_bid(self):
        match = _ScriptedMatch(MatchConfig(format="adaptive", decks=("fignor", "igor"), seed=1), winners=[1, 2, 2])
        self._drive_first_player(match)
        d = match.pending_decision
        self.assertEqual(d.player, 2)
        match.submit(5)  # P2 raises to 5
        d = match.pending_decision
        self.assertEqual(d.kind, DecisionKind.BID_CHAINS)
        self.assertEqual(d.player, 1)  # owner's turn to respond
        self.assertEqual(d.options[1], 6)  # must raise past the current bid
        match.submit("pass")  # owner passes: P2 wins the bid at 5
        self._drive_first_player(match)
        self.assertEqual(match.bid.winner, 2)
        self.assertEqual(match.bid.amount, 5)
        self.assertEqual(match.calls[2]["starting_chains"], {2: 5, 1: 0})
        self.assertEqual(match.calls[2]["decks"], ("igor", "fignor"))  # seat1=loser->igor, seat2=winner->fignor

    def test_bid_cap_at_24_ends_immediately(self):
        match = _ScriptedMatch(MatchConfig(format="adaptive", decks=("fignor", "igor"), seed=1), winners=[1, 2])
        self._drive_first_player(match)
        d = match.pending_decision
        self.assertEqual(d.options[-1], 24)
        match.submit(24)  # P2 jumps straight to the cap
        self.assertEqual(match.bid.winner, 2)
        self.assertEqual(match.bid.amount, 24)

    def test_undecided_game_aborts_the_match(self):
        match = _ScriptedMatch(MatchConfig(format="adaptive", decks=("fignor", "igor"), seed=1), winners=[None])
        self.assertTrue(match.is_over)
        self.assertIsNone(match.result["winner"])
        self.assertEqual(len(match.games), 1)


class TestFullAdaptiveMatchAndReplay(unittest.TestCase):
    def _play_out(self, match, bots):
        guard = 0
        while not match.is_over:
            d = match.pending_decision
            match.submit(bots[d.player].decide(match.view_for(d.player), d))
            guard += 1
            self.assertLess(guard, 50000, "match did not terminate")

    def test_real_adaptive_match_replays_exactly(self):
        config = MatchConfig(format="adaptive", decks=("fignor", "igor"), seed=7)
        match = Match(config)
        bots = {1: HeuristicBot(seed=1), 2: HeuristicBot(seed=2)}
        self._play_out(match, bots)
        self.assertTrue(match.is_over)
        original_result = dict(match.result)

        replayed = match_replay(config, match.choice_record)
        self.assertEqual(replayed.result["winner"], original_result["winner"])
        self.assertEqual(replayed.result["games"], original_result["games"])
        self.assertEqual(len(replayed.games), len(match.games))
        for a, b in zip(replayed.games, match.games):
            self.assertEqual(a.winner, b.winner)
            self.assertEqual(a.seat_decks, b.seat_decks)
            self.assertEqual(a.final_chains, b.final_chains)

    def test_random_bots_fuzz_reversal_and_adaptive(self):
        for fmt in ("reversal", "adaptive"):
            for seed in range(5):
                config = MatchConfig(format=fmt, decks=("fignor", "igor"), seed=1000 + seed)
                match = Match(config)
                bots = {1: RandomBot(seed=seed), 2: RandomBot(seed=seed + 1)}
                self._play_out(match, bots)
                self.assertTrue(match.is_over)
                self.assertIsNotNone(match.result)

    def test_match_config_roundtrips_through_dict(self):
        config = MatchConfig(format="adaptive", decks=("fignor", "igor"), first_player=1, seed=9, max_turns=100)
        data = match_config_to_dict(config)
        restored = match_config_from_dict(data)
        # The round-trip embeds each preset's full decklist (so a replay
        # survives that preset being edited or deleted later), so `decks`
        # comes back as resolved `Deck` objects rather than the original
        # bare names -- everything else round-trips unchanged.
        self.assertEqual(restored.format, config.format)
        self.assertEqual(restored.first_player, config.first_player)
        self.assertEqual(restored.seed, config.seed)
        self.assertEqual(restored.max_turns, config.max_turns)
        self.assertEqual([d.name for d in restored.decks], ["Fignor", "Igor"])
        from keyforge.cards.decks import resolve_deck

        self.assertEqual(restored.decks[0].pods, resolve_deck("fignor").pods)
        self.assertEqual(restored.decks[1].pods, resolve_deck("igor").pods)


if __name__ == "__main__":
    unittest.main()

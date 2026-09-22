"""Milestone A (Code/AGENT_INTERFACE_PLAN.md): determinism and identity --
per-game instance ids, `state_hash`, `outcome_for`, the engine version stamp,
and the portable/keyed RNG it all rests on."""

import unittest

from bots.random_bot import RandomBot
from helpers import default_chooser
from keyforge.config import GameConfig
from keyforge.enums import DecisionKind
from keyforge.game import Game
from keyforge.keyed_random import derive_rng, portable_choice, portable_shuffle
from keyforge.match import MatchConfig, match_config_from_dict, match_config_to_dict
from keyforge.replay import config_from_dict, config_to_dict, replay
from keyforge.version import ENGINE_VERSION, RULES_HASH, EngineVersionMismatch, check_version_stamp, version_stamp


def _play_out(config: GameConfig):
    game = Game(config)
    bots = {1: RandomBot(seed=config.seed), 2: RandomBot(seed=(config.seed or 0) + 1)}
    while not game.is_over:
        d = game.pending_decision
        choice = bots[d.player].decide(game.view_for(d.player), d)
        game.submit(choice)
    return game


class TestPerGameInstanceIds(unittest.TestCase):
    def test_ids_are_assigned_in_deck_build_order_starting_at_one(self):
        game = Game(GameConfig(decks=("fignor", "igor"), seed=1, max_turns=5))
        p1_ids = sorted(c.instance_id for c in game.players[1].all_cards)
        p2_ids = sorted(c.instance_id for c in game.players[2].all_cards)
        self.assertEqual(p1_ids, list(range(1, 37)))
        self.assertEqual(p2_ids, list(range(37, 73)))

    def test_same_seed_gives_identical_ids_in_a_fresh_process_state(self):
        # Two independent Game() builds, not sharing any module-level counter
        # state, must agree on every card's id -- this is what makes ids
        # stable across processes and forks.
        g1 = Game(GameConfig(decks=("fignor", "igor"), seed=1, max_turns=5))
        g2 = Game(GameConfig(decks=("fignor", "igor"), seed=1, max_turns=5))
        ids1 = {c.name: c.instance_id for c in g1.players[1].all_cards + g1.players[2].all_cards}
        ids2 = {c.name: c.instance_id for c in g2.players[1].all_cards + g2.players[2].all_cards}
        self.assertEqual(ids1, ids2)

    def test_ids_are_independent_of_how_many_games_ran_before_in_this_process(self):
        for _ in range(5):
            Game(GameConfig(decks=("fignor", "igor"), seed=99, max_turns=1))
        fresh = Game(GameConfig(decks=("fignor", "igor"), seed=1, max_turns=5))
        self.assertEqual(sorted(c.instance_id for c in fresh.players[1].all_cards), list(range(1, 37)))


class TestStateHash(unittest.TestCase):
    def test_independent_replays_of_the_same_record_hash_equal_throughout(self):
        original = _play_out(GameConfig(decks=("fignor", "igor"), seed=7, max_turns=60))
        config, record = original.config, original.choice_record
        for frac in (0.25, 0.5, 0.75, 1.0):
            upto = max(1, round(len(record) * frac))
            a = replay(config, record, upto=upto)
            b = replay(config, record, upto=upto)
            self.assertEqual(a.state_hash(), b.state_hash(), f"diverged at {frac:.0%} of the game")

    def test_hash_is_stable_across_repeated_calls_with_no_mutation(self):
        game = _play_out(GameConfig(decks=("fignor", "igor"), seed=3, max_turns=30))
        self.assertEqual(game.state_hash(), game.state_hash())

    def test_a_different_choice_produces_a_different_hash(self):
        config = GameConfig(decks=("fignor", "igor"), seed=13, max_turns=60)
        game = Game(config)
        # Drive to the first CHOOSE_ACTION with a fixed script, then branch:
        # one submits option 0 every time, the other prefers the last option.
        prefix = []
        while game.pending_decision.kind != DecisionKind.CHOOSE_ACTION:
            d = game.pending_decision
            choice = d.options[0]
            prefix.append(choice)
            game.submit(choice)

        def finish(choose):
            g = Game(config)
            for c in prefix:
                g.submit(c)
            for _ in range(40):
                if g.is_over:
                    break
                d = g.pending_decision
                if d.kind == DecisionKind.CHOOSE_ACTION:
                    g.submit(choose(d.options))
                else:
                    g.submit(default_chooser(d))
            return g

        first = finish(lambda opts: opts[0])
        last = finish(lambda opts: opts[-1])
        if first.choice_record != last.choice_record:
            self.assertNotEqual(first.state_hash(), last.state_hash())

    def test_fork_equivalence_diverging_one_leaves_the_other_untouched(self):
        original = _play_out(GameConfig(decks=("fignor", "igor"), seed=21, max_turns=60))
        upto = max(1, len(original.choice_record) // 2)
        fork = replay(original.config, original.choice_record, upto=upto)
        baseline = replay(original.config, original.choice_record, upto=upto)
        self.assertEqual(fork.state_hash(), baseline.state_hash())
        # Advance the fork one more decision; the untouched baseline must
        # still match a fresh replay to the *original* upto point.
        if not fork.is_over:
            d = fork.pending_decision
            fork.submit(d.options[0])
        still = replay(original.config, original.choice_record, upto=upto)
        self.assertEqual(baseline.state_hash(), still.state_hash())


class TestOutcomeFor(unittest.TestCase):
    def test_matches_result_winner_from_both_perspectives(self):
        game = _play_out(GameConfig(decks=("fignor", "igor"), seed=5, max_turns=80))
        winner = game.result.get("winner")
        if winner is None:
            self.assertEqual(game.outcome_for(1), 0)
            self.assertEqual(game.outcome_for(2), 0)
        else:
            self.assertEqual(game.outcome_for(winner), 1)
            self.assertEqual(game.outcome_for(3 - winner), -1)

    def test_raises_before_the_game_is_over(self):
        game = Game(GameConfig(decks=("fignor", "igor"), seed=1, max_turns=5))
        with self.assertRaises(RuntimeError):
            game.outcome_for(1)


class TestVersionStamp(unittest.TestCase):
    def test_config_round_trip_carries_a_matching_stamp(self):
        config = GameConfig(decks=("fignor", "igor"), seed=1)
        data = config_to_dict(config)
        self.assertEqual(data["engine_version"], ENGINE_VERSION)
        self.assertEqual(data["rules_hash"], RULES_HASH)
        config_from_dict(data)  # must not raise

    def test_mismatched_engine_version_is_refused(self):
        data = config_to_dict(GameConfig(decks=("fignor", "igor"), seed=1))
        data["engine_version"] = "0.0.0-stale"
        with self.assertRaises(EngineVersionMismatch):
            config_from_dict(data)

    def test_mismatched_rules_hash_is_refused(self):
        data = config_to_dict(GameConfig(decks=("fignor", "igor"), seed=1))
        data["rules_hash"] = "0" * 64
        with self.assertRaises(EngineVersionMismatch):
            config_from_dict(data)

    def test_a_record_with_no_stamp_at_all_is_refused(self):
        data = config_to_dict(GameConfig(decks=("fignor", "igor"), seed=1))
        del data["engine_version"]
        del data["rules_hash"]
        with self.assertRaises(EngineVersionMismatch):
            config_from_dict(data)

    def test_match_config_round_trip_is_stamped_the_same_way(self):
        config = MatchConfig(format="archon", decks=("fignor", "igor"), seed=1)
        data = match_config_to_dict(config)
        self.assertEqual(data["engine_version"], ENGINE_VERSION)
        match_config_from_dict(data)  # must not raise
        data["rules_hash"] = "bad"
        with self.assertRaises(EngineVersionMismatch):
            match_config_from_dict(data)

    def test_check_version_stamp_accepts_a_freshly_made_stamp(self):
        check_version_stamp(version_stamp())


class TestKeyedRandomness(unittest.TestCase):
    def test_derive_rng_is_deterministic_and_key_sensitive(self):
        a = derive_rng(1, 1, "reshuffle", 0)
        b = derive_rng(1, 1, "reshuffle", 0)
        self.assertEqual([a.random() for _ in range(5)], [b.random() for _ in range(5)])
        c = derive_rng(1, 1, "reshuffle", 1)
        self.assertNotEqual(derive_rng(1, 1, "reshuffle", 0).random(), c.random())

    def test_event_rng_counters_advance_independently_per_key(self):
        game = Game(GameConfig(decks=("fignor", "igor"), seed=1, max_turns=5))
        r1 = game.event_rng("some_kind", 1)
        r2 = game.event_rng("some_kind", 2)
        r3 = game.event_rng("some_kind", 1)
        self.assertEqual(game._rng_counters[(1, "some_kind")], 2)
        self.assertEqual(game._rng_counters[(2, "some_kind")], 1)
        self.assertNotEqual(r1.random(), r3.random())

    def test_portable_shuffle_is_a_permutation_and_deterministic(self):
        items = list(range(20))
        shuffled = list(items)
        portable_shuffle(derive_rng(1, None, "k", 0), shuffled)
        self.assertEqual(sorted(shuffled), items)
        again = list(items)
        portable_shuffle(derive_rng(1, None, "k", 0), again)
        self.assertEqual(shuffled, again)

    def test_portable_choice_picks_from_the_sequence(self):
        items = ["a", "b", "c"]
        self.assertIn(portable_choice(derive_rng(1, None, "k", 0), items), items)
        with self.assertRaises(IndexError):
            portable_choice(derive_rng(1, None, "k", 0), [])

    def test_two_games_diverging_only_in_an_extra_mulligan_stay_dealt_alike(self):
        """The point of keying by (player, kind) rather than one shared
        stream: player 2's opening deal is unaffected by whether player 1
        mulligans, since a mulligan only consumes player 1's own 'reshuffle'
        draws, not a shared stream both players pull from."""
        seed = 1234

        def opening_p2_hand(mulligan_p1: bool):
            game = Game(GameConfig(decks=("fignor", "igor"), seed=seed, first_player=1, max_turns=5))
            game.submit(mulligan_p1)  # player 1's mulligan decision
            game.submit(False)  # player 2 never mulligans
            return sorted(c.name for c in game.players[2].hand.cards())

        self.assertEqual(opening_p2_hand(False), opening_p2_hand(True))


if __name__ == "__main__":
    unittest.main()

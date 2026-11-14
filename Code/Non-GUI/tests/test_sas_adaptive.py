"""SAS-gap evaluation and Adaptive matches in the arena: deck pairs by SAS gap, a network's between-game decisions
(first or second, chain bids), match records, and the report's fits."""

from __future__ import annotations

import json
import math
import random
import unittest
from types import SimpleNamespace

import run
from keyforge.enums import DecisionKind
from keyforge.match import MAX_BID, Match, MatchConfig
from sim.decklists import decklist, pair_in_gap, sas_of


class _FlatClient:
    """Every position worth exactly an even game."""

    def __init__(self):
        self.calls = 0

    def predict_many(self, reqs):
        self.calls += len(reqs)
        return [([], 0.0) for _ in reqs]


class GapPairs(unittest.TestCase):
    def test_pairs_fall_in_their_band_and_are_fixed(self):
        for lo, hi in ((0, 5), (10, 20)):
            for k in range(20):
                a, b = pair_in_gap("phase3", "test", k, lo, hi)
                self.assertTrue(lo <= abs(sas_of("phase3", a.name) - sas_of("phase3", b.name)) < hi)
                self.assertEqual(pair_in_gap("phase3", "test", k, lo, hi)[0].name, a.name)

    def test_only_test_decks(self):
        test = {d.name for d in decklist("phase3", "test")}
        for k in range(50):
            a, b = pair_in_gap("phase3", "test", k, 20, 60)
            self.assertTrue({a.name, b.name} <= test)


class MatchPolicy(unittest.TestCase):
    def _match_at_bid(self, seed=0):
        """A match driven by HeuristicBots up to game 3's first bid (or None if it ended 2-0)."""
        from bots.registry import make_agent

        a, b = pair_in_gap("phase3", "test", seed, 0, 5)
        m = Match(MatchConfig(format="adaptive", decks=(a, b), seed=seed, max_turns=60))
        bots = {1: make_agent("heuristic", seed=1), 2: make_agent("heuristic", seed=2)}
        while not m.is_over:
            d = m.pending_decision
            if d.kind == DecisionKind.BID_CHAINS:
                return m
            m.submit(bots[d.player].decide(m.view_for(d.player), d))
        return None

    def test_even_values_mean_no_bid_and_curves_cover_every_bid(self):
        from agent.match_policy import NetMatchPolicy

        for seed in range(12):
            m = self._match_at_bid(seed)
            if m is not None:
                break
        self.assertIsNotNone(m, "no match reached a bid in 12 seeds")
        d = m.pending_decision
        pol = NetMatchPolicy(_FlatClient(), deals=2, bid="value")
        self.assertEqual(pol.decide(m.view_for(d.player), d), "pass")
        p_pilot, p_face = pol.curves(m.view_for(d.player), d.player)
        self.assertEqual(len(p_pilot), MAX_BID + 1)
        self.assertEqual(pol.client.calls, (MAX_BID + 1) * 2 * 2 * 2)
        pol.decide(m.view_for(d.player), d)  # cached: no new positions
        self.assertEqual(pol.client.calls, (MAX_BID + 1) * 2 * 2 * 2)

    def test_bids_up_to_the_indifference_point(self):
        from agent.match_policy import NetMatchPolicy

        pol = NetMatchPolicy(_FlatClient(), bid="value")
        # piloting is worth 0.7 - 0.02 c, facing 0.3 + 0.02 c: indifferent at c = 10
        pol.curves = lambda view, me: ([0.7 - 0.02 * c for c in range(MAX_BID + 1)],
                                       [0.3 + 0.02 * c for c in range(MAX_BID + 1)])
        for r, want in ((1, 1), (9, 9), (10, "pass"), (15, "pass")):
            d = SimpleNamespace(kind=DecisionKind.BID_CHAINS, player=1, options=["pass"] + list(range(r, MAX_BID + 1)))
            self.assertEqual(pol.decide(None, d), want)

    def test_rollout_bidding_finds_the_indifference_point(self):
        from agent.match_policy import NetMatchPolicy

        view = SimpleNamespace(games=[SimpleNamespace(config=SimpleNamespace(seed=s)) for s in (1, 2)])
        for p0, want in ((0.7, 9), (0.45, -1), (0.99, MAX_BID)):
            pol = NetMatchPolicy(_FlatClient())
            calls = []
            pol._rollout = lambda v, c, p0=p0: calls.append(c) or (p0 - 0.02 * c)
            self.assertEqual(pol.indifference(view), want)
            self.assertLessEqual(len(calls), 7)  # bisection, not a sweep
            pol.indifference(view)
            self.assertLessEqual(len(calls), 7)  # cached
        pol = NetMatchPolicy(_FlatClient())
        pol._rollout = lambda v, c: 0.7 - 0.02 * c
        for r, want in ((1, 1), (9, 9), (10, "pass")):
            d = SimpleNamespace(kind=DecisionKind.BID_CHAINS, player=1, options=["pass"] + list(range(r, MAX_BID + 1)))
            self.assertEqual(pol.decide(view, d), want)

    def test_noisy_curves_are_made_monotone(self):
        from agent.match_policy import NetMatchPolicy

        for seed in range(12):
            m = self._match_at_bid(seed)
            if m is not None:
                break
        d = m.pending_decision
        rng = random.Random(3)

        class Noisy(_FlatClient):
            def predict_many(self, reqs):
                return [([], rng.uniform(-1, 1)) for _ in reqs]

        p_pilot, p_face = NetMatchPolicy(Noisy(), deals=1, bid="value").curves(m.view_for(d.player), d.player)
        self.assertTrue(all(x >= y for x, y in zip(p_pilot, p_pilot[1:])))
        self.assertTrue(all(x <= y for x, y in zip(p_face, p_face[1:])))

    def test_a_whole_match_with_network_decisions(self):
        from agent.match_policy import MATCH_KINDS, NetMatchPolicy
        from bots.registry import make_agent

        played3 = False
        for seed in range(8):
            a, b = pair_in_gap("phase3", "test", seed, 0, 5)
            m = Match(MatchConfig(format="adaptive", decks=(a, b), seed=seed, max_turns=60))
            bots = {1: make_agent("heuristic", seed=1), 2: make_agent("heuristic", seed=2)}
            pols = {1: NetMatchPolicy(_FlatClient(), deals=1, seed=1, bid="value"),
                    2: NetMatchPolicy(_FlatClient(), deals=1, seed=2, bid="value")}
            while not m.is_over:
                d = m.pending_decision
                who = pols if d.kind in MATCH_KINDS else bots
                m.submit(who[d.player].decide(m.view_for(d.player), d))
            played3 |= len(m.games) == 3
            if len(m.games) == 3:
                self.assertEqual(m.bid.amount, 0)  # flat values: nobody bids
        self.assertTrue(played3, "no match went to game 3 in 8 seeds")


class ArenaAdaptive(unittest.TestCase):
    def test_bots_play_paired_matches_and_records_hold_them(self):
        from ml.arena import Player, play_paired

        a, b = pair_in_gap("phase3", "test", 0, 10, 20)
        h = Player(name="h", kind="bot", bot="heuristic")
        rep = play_paired(h, Player(name="r", kind="bot", bot="random"), [11, 12], a, b, max_turns=60, fmt="adaptive")
        self.assertEqual(rep.games, 8)
        self.assertEqual(len(rep.records), 8)
        for r in rep.records:
            self.assertIn(len(r["games"]), (1, 2, 3))
            self.assertEqual("bid" in r, len(r["games"]) == 3)
            if r["winner"] is not None:
                self.assertEqual(r["winner"], r["games"][-1]["winner"])
            self.assertEqual(r["games"][0]["a_deck"], r["a_deck"])
            if len(r["games"]) > 1:
                self.assertEqual(r["games"][1]["a_deck"], r["b_deck"])
        self.assertGreater(rep.score, 0.6)  # heuristic beats random

    def test_archon_records_one_per_game(self):
        from ml.arena import Player, play_paired

        h = Player(name="h", kind="bot", bot="heuristic")
        rep = play_paired(h, h, [5], max_turns=60)
        self.assertEqual([("games" in r) for r in rep.records], [False] * 4)
        self.assertEqual({r["a_deck"] for r in rep.records}, {"fignor", "igor"})
        self.assertEqual({r["a_seat"] for r in rep.records}, {1, 2})


class Report(unittest.TestCase):
    def test_the_fit_recovers_a_known_slope(self):
        from tools.sas_report import ELO, fit_logistic

        rng = random.Random(0)
        xs = [rng.uniform(-30, 30) for _ in range(20000)]
        ys = [1.0 if rng.random() < 1 / (1 + math.exp(-(100 + 8 * x) / ELO)) else 0.0 for x in xs]
        e, k, se_e, se_k = fit_logistic(xs, ys)
        self.assertAlmostEqual(e, 100, delta=4 * se_e)
        self.assertAlmostEqual(k, 8, delta=4 * se_k)

    def test_a_mirror_where_the_higher_deck_always_wins(self):
        from tools.sas_report import analyse

        decks = {"A": 80.0, "B": 60.0}
        units = [{"records": [{"a_deck": "A", "b_deck": "B", "winner": "a"}, {"a_deck": "B", "b_deck": "A", "winner": "b"}]}
                 for _ in range(5)]
        res, rows = analyse({"_file": "x", "a": "p", "b": "p", "sas_gap": "10-30", "units": units}, decks.__getitem__)
        self.assertEqual(res["higher_deck_score"], 1.0)
        self.assertEqual(len(rows), 10)


class Jobs(unittest.TestCase):
    def test_eval_sas_covers_every_band_both_opponents_and_formats(self):
        for method, fmt in (("eval-sas", "archon"), ("eval-sas-adaptive", "adaptive")):
            steps = run.METHODS[method]["steps"]
            self.assertEqual(len(steps), len(run.SAS_GAPS) * len(run.SAS_PAIRS))
            for st in steps:
                self.assertEqual(st[st.index("--format") + 1], fmt)
                self.assertEqual(st[st.index("--deck-split") + 1], "test")
                self.assertEqual(st[st.index("--unit") + 1], "1")
            self.assertEqual({st[st.index("--sas-gap") + 1] for st in steps}, set(run.SAS_GAPS))
        self.assertIn("export-sas", run.METHODS)

    def test_output_names_keep_old_matches_and_tell_new_ones_apart(self):
        from tools.eval_v2 import out_name, parse_gap

        self.assertEqual(out_name("a", "b", 100, 100, 1), "a__b__100sims_100x1.json")
        self.assertTrue(out_name("a", "b", 100, 100, 1, deck_list="phase3", sas_gap="0-5", fmt="adaptive")
                        .endswith("_phase3-test_gap0-5_adaptive.json"))
        self.assertEqual(parse_gap("10-20"), (10.0, 20.0))
        self.assertIsNone(parse_gap(""))


if __name__ == "__main__":
    unittest.main()

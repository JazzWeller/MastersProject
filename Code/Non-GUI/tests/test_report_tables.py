"""Report names (tools/report_names.py), the four tables (tools/report_tables.py) and the jobs that fill them (run.py)."""

from __future__ import annotations

import json
import math
import os
import tempfile
import unittest

import run
from tools import report_tables as rt
from tools.report_names import RUNS, name


class Names(unittest.TestCase):
    def test_specs_get_report_names(self):
        self.assertEqual(name("heuristic"), "HeuristicBot")
        self.assertEqual(name("v1:real-v1"), "Real-deck agent")
        self.assertEqual(name("v1:real-v1#sims=1000"), "Real-deck agent · 1,000 sims")
        self.assertEqual(name("v1:plateau-ctrl@final"), "Real-deck agent + 20k games")
        self.assertEqual(name("v1:m9-wt-1#regime=full_game,leaf=belief_oracle"),
                         "Specialist, within-turn training, seed 1 · full-game search · oracle leaf")
        self.assertEqual(name("v1:m9-wt-1#regime=within_turn,leaf=student"), "Specialist, within-turn training, seed 1")
        self.assertEqual(name("v1:tier2-v1-s3#resample=exact"), "Specialist (compact) · sees hidden cards")
        self.assertEqual(name("v1:unknown-run@p3"), "unknown-run @p3")

    def test_every_player_in_a_table_or_a_ladder_job_has_a_name(self):
        specs = [p for p, _, _ in rt.T1_AGENTS] + [b for _, b, _ in rt.T2_NETS]
        specs += [x for _, _, real, fixed in rt.T3_ROWS for _, a, b in real + fixed for x in (a, b)]
        for m in run.METHODS.values():
            for step in m["steps"]:
                if "tools.ladder" in step:
                    i = step.index("--players") + 1
                    while i < len(step) and not step[i].startswith("--"):
                        specs.append(step[i])
                        i += 1
        for spec in specs:
            run_id = spec.partition("#")[0].split(":", 1)[-1].partition("@")[0]
            self.assertIn(run_id, RUNS, spec)


def _ladder(name, players, scores, games=400):
    """A ladder as tools.ladder writes it, from {(a, b): a's score}; players[0] is the anchor."""
    from tools.ladder import fit

    idx = {p: i for i, p in enumerate(players)}
    results = {(idx[a], idx[b]): (s * games, games) for (a, b), s in scores.items()}
    elo, se = fit(players, results)
    return {"name": name, "anchor": players[0],
            "matches": [{"a": a, "b": b, "a_score": s, "games": games} for (a, b), s in scores.items()],
            "ratings": [{"player": p, "elo": e, "se": s} for p, e, s in zip(players, elo, se)]}


class Tables(unittest.TestCase):
    def test_reanchoring_differences_and_the_copy_with_most_matches(self):
        H, R1, R2, C = "heuristic", "v1:real-v1", "v2:real-v2", "v1:plateau-ctrl@final"
        with tempfile.TemporaryDirectory() as root:
            def put(sub, d):
                os.makedirs(os.path.join(root, sub, "evals"), exist_ok=True)
                with open(os.path.join(root, sub, "evals", f"ladder-{d['name']}.json"), "w", encoding="utf-8") as f:
                    json.dump(d, f)
            put("old", _ladder("real-test", [H, R1], {(H, R1): 0.1}))
            put("new", _ladder("real-test", [H, R1, R2], {(H, R1): 0.08, (H, R2): 0.09, (R1, R2): 0.53}))
            put("new", _ladder("plateau-sasw", ["v1:m11-v1", H, R1, C],
                               {("v1:m11-v1", H): 0.92, ("v1:m11-v1", R1): 0.49, ("v1:m11-v1", C): 0.51,
                                (H, R1): 0.075, (H, C): 0.08, (R1, C): 0.52}))
            L = rt.Ladders(rt.load_ladders(root))
            t = L.l["real-test"]["ratings"]
            self.assertEqual(len(t), 3)  # the copy with more matches
            r1 = [r for r in t if r["player"] == R1][0]
            v, se = L.elo("real-test", R1)
            self.assertAlmostEqual(v, r1["elo"])
            self.assertAlmostEqual(se, r1["se"], places=6)  # anchored on HeuristicBot: the ladder's own error
            # anchored elsewhere: shifted to HeuristicBot = 0, its error from the fit's covariance
            p = {r["player"]: r for r in L.l["plateau-sasw"]["ratings"]}
            v, se = L.elo("plateau-sasw", R1)
            self.assertAlmostEqual(v, p[R1]["elo"] - p[H]["elo"])
            self.assertLess(se, (p[R1]["se"] ** 2 + p[H]["se"] ** 2) ** 0.5)  # they share opponents
            # a difference: smaller error than its ratings' combined, and no larger than the direct match's alone
            d, dse = L.diff([("plateau-sasw", C, R1)])
            self.assertAlmostEqual(d, p[C]["elo"] - p[R1]["elo"])
            direct = 400 / math.log(10) / math.sqrt(400 * 0.52 * 0.48)
            self.assertLess(dse, direct + 1e-9)
            self.assertLess(dse, (p[C]["se"] ** 2 + p[R1]["se"] ** 2) ** 0.5)
            # a mean of the same difference twice has the same error
            self.assertAlmostEqual(L.diff([("plateau-sasw", C, R1)] * 2)[1], dse)
            self.assertIsNone(L.diff([("plateau-sasw", "v1:plateau-nogate@final", R1)]))  # not rated: empty
            t1 = rt.table1(L)
            row = [r for r in t1[3] if r[0] == "Real-deck agent"][0]
            self.assertAlmostEqual(row[5][0], r1["elo"])
            self.assertIsNone(row[4])  # no random-deck ladder here
            self.assertIn(f"{r1['elo']:.0f} ± {r1['se']:.0f}", rt.markdown([t1, rt.table2(L), rt.table3(L), rt.table4(None)]))


class Jobs(unittest.TestCase):
    def test_extended_ladders_keep_every_earlier_player(self):
        # the matches already played are found by file name, so no earlier player may be renamed or dropped
        for earlier in (["heuristic", "v1:tier2-v1-s3#sims=10", "v1:tier2-v1-s3#sims=30", "v1:tier2-v1-s3",
                         "v1:tier2-v1-s3#sims=300", "v1:tier2-v1-s3#resample=exact"],):
            self.assertTrue(set(earlier) <= set(run.DIAG_LADDER))
        self.assertTrue({"heuristic", "v1:tier0b-a0", "v2:tier0b2-a7", "v1:tier2-v1-s3", "v2:tier2-v2-s3", "v1:m11-v1",
                         "v2:m11-v2"} <= set(run.M11_LADDER))
        self.assertTrue({"heuristic", "v1:m11-v1", "v2:m11-v2", "v1:real-v1", "v2:real-v2"} <= set(run.REAL_TRAIN_LADDER))
        for m in ("eval-table1", "eval-diag", "eval-anatomy-bc", "eval-table3", "export-tables"):
            self.assertIn(m, run.METHODS)
        names = [s[s.index("--name") + 1] for m in ("eval-table1", "eval-diag", "eval-anatomy-bc", "eval-table3")
                 for s in run.METHODS[m]["steps"]]
        self.assertEqual(names, ["m11", "real-train", "diag", "anatomy-bc", "design-real", "size-fi", "m9-real"])
        # every ladder a table reads is one a job writes
        read = {lad for lad, _ in rt.T1_LADDERS} | {lad for lad, _, _ in rt.T2_NETS}
        read |= {lad for _, _, real, fixed in rt.T3_ROWS for lad, _, _ in real + fixed}
        written = {s[s.index("--name") + 1] for m in run.METHODS.values() for s in m["steps"] if "tools.ladder" in s}
        self.assertEqual(read - written, set())


if __name__ == "__main__":
    unittest.main()

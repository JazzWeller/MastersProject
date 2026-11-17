"""Report names (tools/report_names.py), the four tables (tools/report_tables.py) and the jobs that fill them (run.py)."""

from __future__ import annotations

import json
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


def _ladder(name, anchor, ratings):
    return {"name": name, "anchor": anchor, "matches": [{}] * len(ratings),
            "ratings": [{"player": p, "elo": e, "se": s} for p, e, s in ratings]}


class Tables(unittest.TestCase):
    def test_reanchoring_differences_and_the_copy_with_most_matches(self):
        with tempfile.TemporaryDirectory() as root:
            def put(sub, d):
                os.makedirs(os.path.join(root, sub, "evals"), exist_ok=True)
                with open(os.path.join(root, sub, "evals", f"ladder-{d['name']}.json"), "w", encoding="utf-8") as f:
                    json.dump(d, f)
            put("old", _ladder("real-test", "heuristic", [("heuristic", 0, 0), ("v1:real-v1", 400, 10)]))
            put("new", _ladder("real-test", "heuristic", [("heuristic", 0, 0), ("v1:real-v1", 404, 11),
                                                          ("v2:real-v2", 386, 11)]))
            put("new", _ladder("plateau-sasw", "v1:m11-v1", [("v1:m11-v1", 0, 0), ("heuristic", -427.5, 12.8),
                                                             ("v1:real-v1", 9.4, 8.5), ("v1:plateau-ctrl@final", -5.8, 8.5)]))
            L = rt.Ladders(rt.load_ladders(root))
            self.assertEqual(L.elo("real-test", "v1:real-v1"), (404, 11))  # the copy with more matches
            v, se = L.elo("plateau-sasw", "v1:real-v1")
            self.assertAlmostEqual(v, 436.9)
            self.assertAlmostEqual(se, (8.5 ** 2 + 12.8 ** 2) ** 0.5)  # HeuristicBot's own error added
            d, dse = L.diff([("plateau-sasw", "v1:plateau-ctrl@final", "v1:real-v1")])
            self.assertAlmostEqual(d, -15.2)
            self.assertIsNone(L.diff([("plateau-sasw", "v1:plateau-nogate@final", "v1:real-v1")]))  # not rated: empty
            t1 = rt.table1(L)
            row = [r for r in t1[3] if r[0] == "Real-deck agent"][0]
            self.assertEqual(row[5], (404, 11))
            self.assertIsNone(row[4])  # no random-deck ladder here
            self.assertIn("404 ± 11", rt.markdown([t1, rt.table2(L), rt.table3(L), rt.table4(None)]))


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

"""Agent Training Plan, Milestone M0 (Code/AGENT_TRAINING_PLAN.md): the
torch-free `agent` package, and config/run provenance -- plus M10's
telemetry primitives, which land with it."""

import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest

from agent import config as config_mod
from agent.telemetry import MetricsWriter, Run, read_jsonl

_NON_GUI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_AGENT_DIR = os.path.join(_NON_GUI, "agent")


def _agent_modules():
    for root, dirs, files in os.walk(_AGENT_DIR):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in files:
            if name.endswith(".py"):
                yield os.path.join(root, name)


class TestTorchFree(unittest.TestCase):
    def test_no_module_under_agent_imports_torch(self):
        """Static: no `import torch`/`from torch ...` (or numpy, which is
        slow under PyPy) anywhere in agent/, at any nesting level."""
        offenders = []
        for path in _agent_modules():
            with open(path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=path)
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for n in names:
                    top = n.split(".")[0]
                    if top in ("torch", "numpy", "ml"):
                        offenders.append(f"{os.path.relpath(path, _NON_GUI)}: imports {n}")
        self.assertEqual(offenders, [])

    def test_agent_imports_with_torch_absent_from_sys_modules(self):
        """Dynamic: import every agent module in a fresh interpreter where
        `torch` is made unimportable, and check it never got loaded."""
        mods = []
        for path in _agent_modules():
            rel = os.path.relpath(path, _NON_GUI)[:-3].replace(os.sep, ".")
            mods.append(rel[: -len(".__init__")] if rel.endswith(".__init__") else rel)
        code = (
            "import sys\n"
            "class _Block:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] in ('torch', 'numpy'):\n"
            "            raise ImportError('blocked: ' + name)\n"
            "        return None\n"
            "sys.meta_path.insert(0, _Block())\n"
            f"for m in {sorted(mods)!r}:\n"
            "    __import__(m)\n"
            "assert 'torch' not in sys.modules and 'numpy' not in sys.modules\n"
            "print('ok')\n"
        )
        out = subprocess.run([sys.executable, "-c", code], cwd=_NON_GUI, capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("ok", out.stdout)


class TestConfig(unittest.TestCase):
    def test_every_bundled_config_resolves(self):
        for name in sorted(os.listdir(config_mod.CONFIG_DIR)):
            if name.endswith(".json"):
                with self.subTest(config=name):
                    resolved = config_mod.resolve(name)
                    self.assertIn("code", resolved)

    def test_resolution_is_deterministic_and_hash_is_reproducible(self):
        a = config_mod.resolve({"tier": 1, "search": {"simulations": 200}})
        b = config_mod.resolve({"search": {"simulations": 200}, "tier": 1})
        self.assertEqual(config_mod.canonical_bytes(a), config_mod.canonical_bytes(b))
        self.assertEqual(config_mod.config_hash(a), config_mod.config_hash(b))

    def test_differences_are_enumerable(self):
        a = config_mod.resolve({"search": {"simulations": 200}})
        b = config_mod.resolve({"search": {"simulations": 50}, "tier": 2})
        diffs = {path for path, _a, _b in config_mod.diff_configs(a, b)}
        self.assertEqual(diffs, {"search.simulations", "tier"})
        self.assertNotEqual(config_mod.config_hash(a), config_mod.config_hash(b))

    def test_unknown_top_level_key_is_refused(self):
        with self.assertRaises(ValueError):
            config_mod.resolve({"serach": {"simulations": 5}})

    def test_resolved_config_carries_the_code_stamps(self):
        from agent import spec

        code = config_mod.resolve()["code"]
        self.assertEqual(code["feature_version"], spec.FEATURE_VERSION)
        for key in ("vocab_hash", "engine_version", "rules_hash", "shape_hash"):
            self.assertIn(key, code)


class TestRunProvenance(unittest.TestCase):
    def test_run_artifacts_carry_the_config_hash_and_re_resolve_identically(self):
        with tempfile.TemporaryDirectory() as root:
            run = Run.create("smoke", {"tier": 0}, root=root)
            self.assertEqual(run.config_hash, config_mod.config_hash(config_mod.resolve({"tier": 0})))
            metrics = run.metrics("actor-0", interval_seconds=0.0)
            metrics.count("decisions", 10)
            metrics.flush()
            run.journal("lr_change", game_index=1234, old=1e-3, new=5e-4)
            for rec in run.metric_records() + run.journal_entries():
                self.assertEqual(rec["config_hash"], run.config_hash)
            path = run.write_artifact("checkpoints", b"weights")
            self.assertTrue(path.endswith(__import__("hashlib").sha256(b"weights").hexdigest()))
            # Re-opening reads back the identical config and hash.
            again = Run.open("smoke", root=root)
            self.assertEqual(again.config_hash, run.config_hash)
            self.assertEqual(again.config, run.config)
            # Same id, different config: refused rather than silently mixed.
            with self.assertRaises(ValueError):
                Run.create("smoke", {"tier": 3}, root=root)
            entries = [e["event"] for e in run.journal_entries()]
            self.assertEqual(entries, ["run_created", "lr_change"])
            self.assertEqual(run.journal_entries()[1]["game_index"], 1234)


class TestResume(unittest.TestCase):
    def test_reopening_keeps_the_original_config_and_journals_drift(self):
        with tempfile.TemporaryDirectory() as root:
            first = Run.resume_or_create("r", {"tier": 1}, root=root)
            again = Run.resume_or_create("r", {"tier": 1}, root=root)
            self.assertEqual(again.config_hash, first.config_hash)
            drifted = Run.resume_or_create("r", {"tier": 2}, root=root)
            self.assertEqual(drifted.config_hash, first.config_hash)  # still the run it started as
            drift = [e for e in drifted.journal_entries() if e["event"] == "config_drift"]
            self.assertEqual(len(drift), 1)
            self.assertIn("tier", drift[0]["paths"])


class TestMetricsWriter(unittest.TestCase):
    def test_rate_limited_flush_with_rates_means_and_gauges(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.jsonl")
            w = MetricsWriter(path, "learner", "abc", interval_seconds=60.0)
            t0 = w._t0
            w.count("positions", 600)
            w.observe("loss", 2.0)
            w.observe("loss", 4.0)
            w.gauge("lr", 1e-3)
            self.assertIsNone(w.tick(now=t0 + 30))  # not yet
            rec = w.tick(now=t0 + 60)
            self.assertIsNotNone(rec)
            self.assertAlmostEqual(rec["rates"]["positions"], 10.0)
            self.assertAlmostEqual(rec["means"]["loss"], 3.0)
            self.assertEqual(rec["maxes"]["loss"], 4.0)
            self.assertEqual(rec["gauges"]["lr"], 1e-3)
            self.assertEqual(len(read_jsonl(path)), 1)
            # Counters reset after a flush.
            self.assertEqual(w.flush(now=t0 + 120)["counts"], {})


if __name__ == "__main__":
    unittest.main()

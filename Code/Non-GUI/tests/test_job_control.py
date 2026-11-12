"""Jobs never requeue themselves, a time limit's signal reaches the step (which
saves), a run whose actors keep crashing stops early, and the crash a run
survived (an order of effects that share a key) is gone."""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import run
from agent.search.core import legal_actions, map_choice
from keyforge.decision import Decision
from keyforge.effects.effect_object import TriggerEffect
from keyforge.enums import DecisionKind

try:
    from ml.selfplay_train import CRASH_STALL_SECONDS, ActorWatch
except ImportError:  # no torch or numpy
    ActorWatch = None

HERE = os.path.dirname(os.path.abspath(run.__file__))


class BatchScript(unittest.TestCase):
    def setUp(self):
        self.script = run.sbatch_script("eval-m9", run.METHODS["eval-m9"], run.settings(), "eval-m9")

    def test_a_job_does_not_requeue(self):
        self.assertIn("#SBATCH --no-requeue", self.script)
        self.assertNotIn("#SBATCH --requeue", self.script)
        self.assertFalse(hasattr(run, "requeue_if_paused"))

    def test_the_signal_reaches_run_py(self):
        self.assertIn("exec python3 run.py eval-m9", self.script)

    @unittest.skipUnless(os.name == "posix", "bash and POSIX signals")
    def test_a_signal_to_the_batch_shell_saves_and_ends(self):
        step = ("import signal,sys,time; signal.signal(signal.SIGTERM, lambda *a: sys.exit(75)); "
                "print('up', flush=True); time.sleep(60)")
        job = (f"import sys; sys.path.insert(0, {HERE!r}); import run; "
               f"run.METHODS['t'] = dict(cpus=1, mem_gb=1, gpu=False, hours=1, steps=[[sys.executable, '-c', {step!r}]]); "
               "sys.exit(run.main(['run.py', 't']))")
        shell = f"export KF_IN_JOB=1\nexec {shlex.quote(sys.executable)} -c {shlex.quote(job)}\n"
        with subprocess.Popen(["bash", "-c", shell], stdout=subprocess.PIPE, text=True, cwd=HERE) as p:
            for line in p.stdout:  # (the step's own "up": the handlers are in place)
                if line.strip() == "up":
                    break
            p.send_signal(signal.SIGTERM)
            out = p.stdout.read()
            self.assertEqual(p.wait(timeout=30), run.EXIT_PAUSED)
        self.assertIn("run.py t resumes it", out)


class Baseline(unittest.TestCase):
    def test_it_plays_fignor_and_igor_and_reuses_the_ladders_matches(self):
        step = run.METHODS["eval-baseline"]["steps"][0]
        self.assertNotIn("--deck-set", step)
        self.assertEqual(step[step.index("--seeds") + 1], run.METHODS["eval-ladder"]["steps"][0][run.METHODS["eval-ladder"]["steps"][0].index("--seeds") + 1])
        for player in ("heuristic", "v1:m11-v1", "v2:m11-v2", "v1:tier2-v1-s3", "v2:tier2-v2-s3"):
            self.assertIn(player, step)


class SizeTest(unittest.TestCase):
    """Only the size differs between the two arms: same corpus, same games."""

    @staticmethod
    def _arg(step, flag):
        return step[step.index(flag) + 1]

    def test_the_arms_differ_only_in_size(self):
        small, big = (run.METHODS[f"size-bc-{d}"]["steps"][0] for d in (128, 256))
        self.assertEqual(self._arg(small, "--data"), self._arg(big, "--data"))
        self.assertEqual(self._arg(small, "--config"), self._arg(big, "--config"))
        self.assertEqual(json.loads(self._arg(big, "--override")), {"network": {"d_model": 256, "layers": 8, "ff": 1024}})
        self.assertEqual(json.loads(self._arg(small, "--override")), {"network": {"d_model": 128, "layers": 4, "ff": 512}})
        for d in (128, 256):
            step = run.METHODS[f"size-{d}"]["steps"][0]
            self.assertEqual(self._arg(step, "--init-run"), f"size-bc-{d}")
            self.assertEqual(self._arg(step, "--replay-from"), "real-v1")

    def test_the_corpus_is_the_train_split(self):
        from agent import config

        pool = config.resolve("size_bc_v1.json")["pool"]
        self.assertEqual((pool["deck_source"], pool["deck_list"], pool["split"]), ("list", "phase3", "train"))

    def test_the_ladder_holds_out_the_test_split(self):
        step = run.METHODS["eval-size"]["steps"][0]
        self.assertEqual(self._arg(step, "--deck-split"), "test")
        self.assertIn("v1:size-256@final", step)


def _double(x):
    return 2 * x


def _die(x):
    if x == 3:
        os._exit(1)  # as a CUDA fault can end a process
    return x


class _Unpicklable(Exception):
    def __init__(self, a, b):  # what torch.AcceleratorError does to multiprocessing: unpickles with one argument
        super().__init__(a)


class EvalWorkers(unittest.TestCase):
    """eval-size hung 17 hours after a GPU memory fault in one evaluation worker; now the evaluation ends with an error."""

    def _run(self, work, tasks):
        import threading

        from tools.eval_v2 import run_units

        out, err = [], []

        def go():
            try:
                run_units(2, tasks, None, (), work, out.append)
            except BaseException as e:
                err.append(e)

        t = threading.Thread(target=go, daemon=True)
        t.start()
        t.join(120)
        self.assertFalse(t.is_alive(), "run_units waited on a dead worker")
        return out, err

    def test_results_arrive(self):
        out, err = self._run(_double, [1, 2, 3])
        self.assertEqual((sorted(out), err), ([2, 4, 6], []))

    def test_a_dead_worker_ends_it(self):
        _out, err = self._run(_die, [1, 2, 3, 4])
        self.assertEqual(len(err), 1)

    @unittest.skipUnless(os.name == "posix", "POSIX signals")
    def test_a_dead_worker_ends_it_when_the_rest_ignore_sigterm(self):
        """Workers inherit an ignored SIGTERM (a paused run earlier in the same process): the suite hung here."""
        previous = signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            _out, err = self._run(_die, [1, 2, 3, 4])
        finally:
            signal.signal(signal.SIGTERM, previous)
        self.assertEqual(len(err), 1)

    def test_a_failure_reaches_the_parent_as_text(self):
        import pickle

        from tools import eval_v2

        with mock.patch.object(eval_v2, "_work", side_effect=_Unpicklable("ECC", 1)):
            with self.assertRaises(RuntimeError) as cm:
                eval_v2._safe(None)
        again = pickle.loads(pickle.dumps(cm.exception))
        self.assertIn("_Unpicklable: ECC", str(again))


class ExcludeNodes(unittest.TestCase):
    def test_an_excluded_node_reaches_the_batch_script(self):
        cfg = dict(run.settings(), exclude="bad-node-1")
        script = run.sbatch_script("eval-size", run.METHODS["eval-size"], cfg, "eval-size")
        self.assertIn("#SBATCH --exclude=bad-node-1", script)
        self.assertNotIn("--exclude", run.sbatch_script("eval-size", run.METHODS["eval-size"], dict(run.settings(), exclude=""), "eval-size"))


class PlateauArms(unittest.TestCase):
    """Round 1: three continuations of real-v1 that differ only in the one setting each tests."""

    @staticmethod
    def _over(arm):
        step = run.METHODS[f"plateau-{arm}"]["steps"][0]
        return step, json.loads(step[step.index("--override") + 1])

    def test_each_arm_differs_from_the_control_in_one_setting(self):
        _s, ctrl = self._over("ctrl")
        for arm, extra in (("sims300", {"playout_cap_full": 300, "playout_cap_small": 75}), ("nogate", {"gate_mode": "always"})):
            step, over = self._over(arm)
            self.assertEqual(over["selfplay"], {**ctrl["selfplay"], **extra})
            self.assertEqual(over["pool"], ctrl["pool"])
            self.assertEqual(step[step.index("--init-run") + 1], "real-v1")
        self.assertEqual(ctrl["pool"], {"deck_source": "list", "deck_list": "phase3", "split": "train"})

    def test_the_default_still_gates(self):
        from agent.config import DEFAULTS

        self.assertEqual(DEFAULTS["selfplay"]["gate_mode"], "sprt")

    def test_arms_are_rated_by_their_last_network_on_held_out_decks(self):
        real, fignor = run.METHODS["eval-plateau"]["steps"]
        for arm in run.PLATEAU_ARMS:
            self.assertIn(f"v1:plateau-{arm}@final", real)
            self.assertIn(f"v1:plateau-{arm}@final", fignor)
        self.assertEqual(real[real.index("--deck-split") + 1], "test")
        self.assertNotIn("--deck-list", fignor)

    def test_the_anatomy_ladder_varies_only_real_v1s_search(self):
        step = run.METHODS["eval-anatomy"]["steps"][0]
        players = step[step.index("--players") + 1:step.index("--sims")]
        self.assertEqual(players[0], "heuristic")
        self.assertTrue(all(p.split("#")[0] == "v1:real-v1" for p in players[1:]))
        self.assertEqual(step[step.index("--deck-split") + 1], "test")


class LadderPlayers(unittest.TestCase):
    """A ladder skips a run still training, whose result would be reused as the finished run's."""

    def _run(self, root, name, events):
        run_dir = os.path.join(root, "runs", name)
        os.makedirs(os.path.join(run_dir, "checkpoints"))
        with open(os.path.join(run_dir, "checkpoints", "abc.kfc"), "w"):
            pass
        with open(os.path.join(run_dir, "best.json"), "w") as f:
            json.dump({"checkpoint": "abc"}, f)
        with open(os.path.join(run_dir, "journal.jsonl"), "w") as f:
            for e in events:
                print(json.dumps({"event": e}), file=f)

    def test_only_a_finished_run_is_available(self):
        from tools.ladder import _available

        with tempfile.TemporaryDirectory() as root, mock.patch.dict(os.environ, {"KEYFORGE_DATA": root}):
            self._run(root, "training", ["run_created", "promoted"])
            self._run(root, "finished", ["run_created", "run_finished"])
            self.assertFalse(_available("v1:training"))
            self.assertTrue(_available("v1:finished"))
            self.assertFalse(_available("v1:missing"))
            self.assertTrue(_available("heuristic"))


class MapChoice(unittest.TestCase):
    """Worker 14's crash in m11-v2: 7 triggers, 6 of one card on one event, are
    past `enumerate_cap`, and the fixed policy's order was mapped back by key."""

    def _decision(self, source=None):
        def effect(event):
            return TriggerEffect(source, 1, event, lambda game, ev: None)

        options = [effect("play")] + [effect("reap") for _ in range(6)]
        return Decision(1, DecisionKind.ORDER_EFFECTS, "order", options, 7, 7)

    def test_an_order_over_the_cap_with_shared_keys_is_valid(self):
        real = self._decision()
        fork = self._decision()
        self.assertIsNone(legal_actions(real, 1024))
        choice = map_choice(real, list(fork.options))
        self.assertTrue(real.validate(choice))
        self.assertEqual(choice, real.options)

    def test_a_reversed_order_keeps_its_keys(self):
        real = self._decision()
        choice = map_choice(real, list(reversed(self._decision().options)))
        self.assertTrue(real.validate(choice))
        self.assertEqual(choice[-1], real.options[0])


@unittest.skipIf(ActorWatch is None, "needs torch and numpy")
class Crashes(unittest.TestCase):
    def test_most_actors_abandoned_stops(self):
        w = ActorWatch(26, 0.0)
        w.abandoned = 13
        self.assertIsNone(w.stop_reason(60.0))
        w.abandoned = 14
        self.assertIn("14 of 26", w.stop_reason(60.0))

    def test_slow_crashes_with_no_games_stop(self):
        w = ActorWatch(2, 0.0)
        w.crashes = 1
        self.assertIsNone(w.stop_reason(CRASH_STALL_SECONDS))
        self.assertIn("no new game", w.stop_reason(CRASH_STALL_SECONDS + 1))

    def test_a_crash_while_games_arrive_does_not_stop(self):
        w = ActorWatch(28, 0.0)
        for t in range(0, 20_000, 1000):
            w.crashes += 1
            w.progress(t + 1, float(t))
            self.assertIsNone(w.stop_reason(t + 1.0))

    def test_no_games_without_a_crash_does_not_stop(self):
        self.assertIsNone(ActorWatch(28, 0.0).stop_reason(1e6))


if __name__ == "__main__":
    unittest.main()

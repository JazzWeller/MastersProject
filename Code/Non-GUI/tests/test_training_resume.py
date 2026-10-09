"""Pausable training (ml/resume.py).

- A packed corpus's `skip` leaves out exactly the first batches of the
  shuffled epoch, segments skipped whole included.
- (WSL, torch) BC v2 training paused twice -- mid-epoch and again in the
  next epoch -- and resumed from its state file ends with the same weights
  and the same logged history as the same run uninterrupted.
"""

import contextlib
import io
import os
import shutil
import tempfile
import unittest

try:
    import numpy as np
    import torch
except ImportError:  # PyPy has neither
    np = torch = None

from tests.test_agent_observation_o9 import _corpus_dir, _records


def _packed():
    from ml.dataset_v2 import pack_corpus_v2

    root = _corpus_dir()
    packs = os.path.join(root, "packed_resume")
    if not os.path.isdir(packs):
        pack_corpus_v2(root, packs, workers=1, chunk=7)
    return packs


@unittest.skipIf(torch is None, "needs torch")
class TestSkip(unittest.TestCase):
    def test_skip_drops_exactly_the_first_batches(self):
        from ml.dataset_v2 import PackedCorpusV2

        for form in ("rows", "none"):
            corpus = PackedCorpusV2(_packed(), history=form, segment_chunks=2)
            sel = corpus.indices()
            full = list(corpus.iterate(sel, 16, shuffle=True, seed=3))
            self.assertGreater(len(full), 6)
            for k in (1, 3, len(full) // 2, len(full) - 1, len(full)):
                with self.subTest(form=form, skip=k):
                    tail = list(corpus.iterate(sel, 16, shuffle=True, seed=3, skip=k))
                    self.assertEqual(_records(tail), _records(full[k:]))


@unittest.skipIf(torch is None, "needs torch")
class TestPauseResume(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="resume_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _train(self, state_path=None):
        from agent.config import resolve
        from ml.bc_train_v2 import train
        from ml.dataset_v2 import PackedCorpusV2

        cfg = resolve("tier0b_bc.json", {"network": {"d_model": 32, "heads": 4, "ff": 64, "layers": 1,
                                                     "history_arch": "stream", "attention_kernel": "math"},
                                         "bc": {"batch": 16, "epochs": 2, "compile": False}})
        corpus = PackedCorpusV2(_packed(), history="rows", segment_chunks=2)
        every = corpus.indices()
        with contextlib.redirect_stdout(io.StringIO()):
            return train(cfg, corpus, device=torch.device("cpu"), log_every=2, train_idx=every, val_idx=every,
                         state_path=state_path)

    def test_paused_twice_ends_where_the_uninterrupted_run_does(self):
        from unittest import mock

        from ml import resume

        torch.use_deterministic_algorithms(True)
        try:
            model_a, rep_a = self._train()
            steps = rep_a["training"]["steps"]
            self.assertGreater(steps, 8)
            self.assertEqual(rep_a["training"]["epochs"], 2)
            state = os.path.join(self.tmp, "train_state.pt")
            pauses = iter([3, steps * 3 // 4 - 3])  # mid-epoch 0, then in epoch 1
            calls = {"n": 0, "at": next(pauses)}

            class Scripted(resume.Pauser):
                def requested(self):
                    calls["n"] += 1
                    return calls["n"] == calls["at"]

            with mock.patch.object(resume, "Pauser", Scripted):
                for _ in range(2):
                    with self.assertRaises(resume.Paused) as cm:
                        self._train(state)
                    self.assertEqual(cm.exception.code, resume.PAUSED)
                    self.assertTrue(os.path.exists(state))
                    calls["at"] = next(pauses, -1)
                model_b, rep_b = self._train(state)
        finally:
            torch.use_deterministic_algorithms(False)
        self.assertEqual(rep_b["training"]["steps"], steps)
        self.assertEqual([h["step"] for h in rep_b["training"]["history"]],
                         [h["step"] for h in rep_a["training"]["history"]])
        for (name, a), (_n, b) in zip(model_a.state_dict().items(), model_b.state_dict().items()):
            self.assertTrue(torch.allclose(a, b, atol=1e-6), name)
        self.assertEqual(rep_a["value"]["logloss"], rep_b["value"]["logloss"])


if __name__ == "__main__":
    unittest.main()

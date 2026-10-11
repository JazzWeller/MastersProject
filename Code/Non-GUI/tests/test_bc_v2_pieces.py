"""The v2 trainer's memory cap (`bc.max_tokens`, ml/bc_train_v2.py).

- (WSL, torch) a batch trained in pieces gives the whole batch's losses and
  gradient, for every history architecture;
- the evaluation in pieces gives the same report.
"""

import unittest

try:
    import numpy as np
    import torch
except ImportError:  # PyPy has neither
    np = torch = None

from tests.test_agent_observation_o9 import _corpus_dir


@unittest.skipIf(torch is None, "needs torch")
class TestPieces(unittest.TestCase):
    def _setup(self, arch, form):
        from agent.config import resolve
        from ml.dataset_v2 import CorpusV2
        from ml.model_v2 import KeyForgeNetV2

        cfg = resolve("tier0b_bc.json", {"network": {"d_model": 32, "heads": 4, "ff": 64, "layers": 1, "dropout": 0.0,
                                                     "history_arch": arch, "attention_kernel": "math"}})
        torch.manual_seed(0)
        model = KeyForgeNetV2(cfg["network"])
        corpus = CorpusV2.from_dir(_corpus_dir(), history=form)
        return cfg, model, corpus

    def test_pieces_give_the_whole_batch_gradient(self):
        from ml.bc_train_v2 import DEFAULT_WEIGHTS, backward_step, pieces_of

        autocast = lambda: torch.autocast("cpu", enabled=False)
        for arch, form in (("none", "none"), ("turn_tokens", "folded"), ("joint", "rows"), ("stream", "rows")):
            with self.subTest(arch=arch):
                cfg, model, corpus = self._setup(arch, form)
                batch, tg = corpus.batch(np.arange(min(corpus.size, 96)))
                budget = batch.tokens() * 20  # pieces of 20 rows
                self.assertGreater(len(pieces_of(batch, budget)), 3)
                runs = []
                for max_tokens in (None, budget):
                    model.zero_grad(set_to_none=True)
                    losses = backward_step(model, batch, tg, max_tokens=max_tokens, autocast=autocast,
                                           weights=DEFAULT_WEIGHTS, cap=1024, heads=tuple(DEFAULT_WEIGHTS))
                    grads = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]
                    runs.append(({k: float(v.detach()) for k, v in losses.items()}, grads))
                (la, ga), (lb, gb) = runs
                self.assertEqual(set(la), set(lb))
                for k in la:
                    self.assertAlmostEqual(la[k], lb[k], places=5, msg=k)
                for a, b in zip(ga, gb):
                    self.assertEqual(a is None, b is None)
                    if a is not None:
                        self.assertTrue(torch.allclose(a, b, atol=1e-6, rtol=1e-4))

    def test_evaluation_in_pieces_is_the_same(self):
        from ml.bc_train_v2 import evaluate

        cfg, model, corpus = self._setup("stream", "rows")
        idx = np.arange(corpus.size)
        whole = evaluate(model, corpus, idx, cap=1024, batch_size=64)
        split = evaluate(model, corpus, idx, cap=1024, batch_size=64, max_tokens=1)  # one row per piece
        self.assertEqual(whole["top1_by_kind"], split["top1_by_kind"])
        self.assertEqual(whole["multi_select_exact"], split["multi_select_exact"])
        self.assertAlmostEqual(whole["value"]["logloss"], split["value"]["logloss"], places=4)
        self.assertAlmostEqual(whole["belief"]["zone_logloss"], split["belief"]["zone_logloss"], places=4)


if __name__ == "__main__":
    unittest.main()

"""Agent Observation Plan, Milestone O7: network v2 (torch; WSL).

- Shapes are right for every history architecture, every mix of tokens
  and every decision kind met in fuzz games.
- Padding invariance: padding a batch (another, longer item beside it, or
  `bucket_by_length`'s order) never changes an item's outputs.
- The checkpoint stamp carries feature version 2 and the vocabularies.
"""

import unittest

try:
    import torch
except ImportError:  # the torch half runs in WSL (see tests/test_agent_training_ml.py)
    torch = None


def _items(n_games=3, every=7):
    from agent.features_v2 import encode_v2
    from agent.history import history_for
    from keyforge.game import Game
    from tools.o1_acceptance import POOLS, bots_for, game_config

    items, kinds = [], set()
    for pool in POOLS:
        for i in range(n_games):
            config = game_config(pool, i)
            game = Game(config)
            bots = bots_for(i, config.seed)
            n = 0
            while not game.is_over:
                d = game.pending_decision
                if n % every == 0:
                    h = history_for(game, d.player).copy()
                    items.append((encode_v2(game, d.player), h, game.turn_number))
                    kinds.add(d.kind)
                game.submit(bots[d.player].decide(game.view_for(d.player), d))
                n += 1
    return items, kinds


CFG = {"d_model": 64, "heads": 4, "ff": 128, "layers": 2, "dropout": 0.0, "attention_kernel": "math"}


@unittest.skipIf(torch is None, "needs torch")
class TestNetworkV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items, cls.kinds = _items()

    def _net(self, arch, **kw):
        from ml.model_v2 import KeyForgeNetV2

        torch.manual_seed(0)
        return KeyForgeNetV2(dict(CFG, history_arch=arch, **kw)).eval()

    def test_shapes_for_every_architecture(self):
        from ml.encode_v2 import collate_v2

        batch = collate_v2(self.items[:16])
        for arch in ("none", "summary", "turn_tokens", "joint", "stream"):
            for in_trunk in (True, False):
                with self.subTest(arch=arch, options_in_trunk=in_trunk):
                    net = self._net(arch, options_in_trunk=in_trunk)
                    with torch.no_grad():
                        out = net.encode_state(batch)
                        logits = net.policy_logits(out)
                        zones, draw = net.belief_v2(out, batch)
                    K = batch.blocks["option"].mask.shape[1]
                    self.assertEqual(tuple(logits.shape), (16, K))
                    self.assertEqual(tuple(net.value(out).shape), (16,))
                    self.assertEqual(tuple(zones.shape), (16, 72, 3))
                    self.assertTrue(torch.isfinite(net.value(out)).all())
                    real = batch.blocks["option"].mask
                    self.assertTrue(torch.isfinite(logits[real]).all())
        self.assertGreaterEqual(len(self.kinds), 6)

    def test_padding_never_changes_an_output(self):
        from ml.encode_v2 import collate_v2

        short = min(self.items, key=lambda it: it[1].n)
        long = max(self.items, key=lambda it: it[1].n)
        for arch in ("none", "turn_tokens", "joint", "stream"):
            with self.subTest(arch=arch):
                net = self._net(arch)
                with torch.no_grad():
                    alone = net.encode_state(collate_v2([short]))
                    padded = net.encode_state(collate_v2([short, long]))
                    v1 = net.value(alone)[0]
                    v2 = net.value(padded)[0]
                    p1 = net.policy_logits(alone)[0]
                    p2 = net.policy_logits(padded)[0, :p1.shape[0]]
                self.assertTrue(torch.allclose(v1, v2, atol=1e-5), (arch, v1, v2))
                real = torch.isfinite(p1)
                self.assertTrue(torch.allclose(p1[real], p2[real], atol=1e-4))

    def test_the_stamp_names_v2_and_the_vocabularies(self):
        from ml.model_v2 import checkpoint_stamp

        s = checkpoint_stamp()
        self.assertEqual(s["feature_version"], 2)
        self.assertTrue(s["vocab_v2_hash"])


if __name__ == "__main__":
    unittest.main()

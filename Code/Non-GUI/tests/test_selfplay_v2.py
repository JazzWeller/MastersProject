"""Self-play with network v2 (Agent Observation Plan, after O10): the v2
actor's records, the learner's replay encoding, its buffer and its losses.

The plan's requirement for the replay design: a position the learner encodes
by replaying the record is exactly the position the actor searched.
"""

import random
import unittest

try:
    import torch
except ImportError:  # pragma: no cover - torch-free environments
    torch = None

NET = {"d_model": 32, "layers": 1, "heads": 2, "ff": 64, "dropout": 0.0, "history_arch": "none",
       "enumerate_cap": 1024, "options_in_trunk": False}


def _settings(**over):
    from agent.selfplay import SelfPlaySettings

    s = SelfPlaySettings(leaf="student", sims_full=4, sims_small=2, full_fraction=0.5, concurrency=2, max_turns=6,
                         leaves_in_flight=2, observation_version=2, history="none")
    for k, v in over.items():
        setattr(s, k, v)
    return s


@unittest.skipIf(torch is None, "needs torch")
class TestSelfPlayV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from agent import selfplay
        from agent.features_v2 import encode_v2
        from agent.selfplay import Actor
        from bots.inference_client import InProcessInferenceClient
        from ml.infer_v2 import TorchModelV2
        from ml.model_v2 import KeyForgeNetV2

        torch.manual_seed(0)
        cls.net = KeyForgeNetV2(dict(NET))
        client = InProcessInferenceClient(TorchModelV2(cls.net, "cpu", compile=False))
        cls.live = {}
        original = selfplay._position_v2

        def recording(game, pid, decision, **kw):  # what the actor saw, to compare with the replay
            p = original(game, pid, decision, **kw)
            cls.live[(game.config.seed, p["decision"], pid)] = encode_v2(game, pid).to_bytes()
            return p

        selfplay._position_v2 = recording
        try:
            actor = Actor(client, _settings(), run_seed=3, worker=0)
            cls.records = []
            actor.play(range(3), cls.records.append)
        finally:
            selfplay._position_v2 = original

    def test_records_are_compact_and_carry_what_replay_needs(self):
        for rec in self.records:
            self.assertIn("config", rec)
            self.assertTrue(rec["positions"])
            for p in rec["positions"]:
                self.assertNotIn("enc", p)
                self.assertLessEqual(p["decision"], len(rec["choice_record"]))
                if p["target"] is not None:
                    self.assertAlmostEqual(sum(p["target"]), 1.0, places=5)

    def test_replay_encodes_exactly_the_positions_the_actor_searched(self):
        from ml import selfplay_v2

        seen = []
        original = selfplay_v2.encode_v2_of

        def capture(game, pid):
            e = original(game, pid)
            seen.append(((game.config.seed, len(game.choice_record), pid), e.to_bytes()))
            return e

        selfplay_v2.encode_v2_of = capture
        try:
            arrays = [selfplay_v2.encode_selfplay_game(r) for r in self.records]
        finally:
            selfplay_v2.encode_v2_of = original
        self.assertEqual(sum(a["n"] for a in arrays), sum(len(r["positions"]) for r in self.records))
        self.assertEqual(len(seen), len(self.live))
        for key, enc in seen:
            self.assertEqual(enc, self.live[key], f"replayed encoding differs at {key}")
        for rec, a in zip(self.records, arrays):  # outcome and seat land on every position
            for j, p in enumerate(sorted(rec["positions"], key=lambda q: q["decision"])):
                self.assertEqual(int(a["z"][j]), rec["outcome"][int(a["seat"][j])])

    def _buffer(self):
        from ml.selfplay_v2 import BufferV2, EncodingPool

        pool = EncodingPool(0)
        pool.submit(self.records)
        buf = BufferV2(10_000, 100, "none")
        for arrays, _info in pool.wait_all():
            buf.add(arrays)
        return buf

    def test_losses_are_finite_and_train(self):
        from ml.selfplay_v2 import DEFAULT_WEIGHTS, active_heads, losses_v2

        batch, tg, pis, cands = self._buffer().sample(48, random.Random(0))
        total, L, _st = losses_v2(self.net, batch, tg, pis, cands, DEFAULT_WEIGHTS, active_heads(DEFAULT_WEIGHTS))
        self.assertTrue(torch.isfinite(total))
        self.assertTrue({"value", "oracle"} <= set(L))
        self.assertTrue({"policy", "multi"} & set(L))
        total.backward()

    def test_training_in_pieces_gives_the_whole_batch_gradient(self):
        from contextlib import nullcontext

        from ml.selfplay_v2 import DEFAULT_WEIGHTS, backward_step_v2

        sample = self._buffer().sample(40, random.Random(1))
        grads = []
        for max_tokens in (None, sample[0].tokens() * 7):
            self.net.zero_grad(set_to_none=True)
            backward_step_v2(self.net, sample, weights=DEFAULT_WEIGHTS, max_tokens=max_tokens, autocast=nullcontext)
            grads.append([p.grad.clone() for p in self.net.parameters() if p.grad is not None])
        self.assertEqual(len(grads[0]), len(grads[1]))
        for a, b in zip(*grads):
            self.assertTrue(torch.allclose(a, b, atol=1e-5, rtol=1e-4))

    def test_the_server_swaps_in_a_promoted_network(self):
        import json
        import os
        import tempfile

        from agent.agents.requests import HEAD_VALUE, RequestV2
        from agent.features_v2 import encode_v2
        from keyforge.config import GameConfig
        from keyforge.game import Game
        from ml.checkpoints import CheckpointStore, save_model
        from ml.infer_v2 import ReloadingTorchModelV2
        from ml.model_v2 import KeyForgeNetV2, checkpoint_stamp

        game = Game(GameConfig(decks=("fignor", "igor"), seed=1))
        req = RequestV2(enc=encode_v2(game, 1), head=HEAD_VALUE, history=None, turn=game.turn_number, digest="")
        with tempfile.TemporaryDirectory() as d:
            store = CheckpointStore(os.path.join(d, "checkpoints"))
            other = KeyForgeNetV2(dict(NET))
            with torch.no_grad():
                for p in other.parameters():
                    p.add_(0.5)
            first, _ = save_model(self.net, store, extra=checkpoint_stamp())
            second, _ = save_model(other, store, extra=checkpoint_stamp())
            best = os.path.join(d, "best.json")
            with open(best, "w") as f:
                json.dump({"checkpoint": first}, f)
            model = ReloadingTorchModelV2(KeyForgeNetV2(dict(NET)), "cpu", best, poll_seconds=0.0, compile=False)
            model.net.load_state_dict(self.net.state_dict())
            v1 = model.predict_many([req])[0][1]
            with open(best, "w") as f:
                json.dump({"checkpoint": second}, f)
            v2 = model.predict_many([req])[0][1]
        self.assertEqual(model.reloads, 2)
        self.assertNotAlmostEqual(v1, v2, places=4)

    def test_v2_checkpoints_round_trip_with_their_optimizer(self):
        import os
        import tempfile

        from ml.checkpoints import load_model_v2_state, save_model
        from ml.model_v2 import checkpoint_stamp

        opt = torch.optim.AdamW(self.net.parameters(), lr=1e-3)
        sample = self._buffer().sample(16, random.Random(2))
        from contextlib import nullcontext

        from ml.selfplay_v2 import DEFAULT_WEIGHTS, backward_step_v2

        opt.zero_grad()
        backward_step_v2(self.net, sample, weights=DEFAULT_WEIGHTS, max_tokens=None, autocast=nullcontext)
        opt.step()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.kfc")
            save_model(self.net, path, optimizer=opt, extra={"step": 1, **checkpoint_stamp()})
            model, meta, opt_state = load_model_v2_state(path)
        self.assertEqual(meta["extra"]["step"], 1)
        self.assertTrue(opt_state["state"])
        opt2 = torch.optim.AdamW(model.parameters(), lr=1e-3)
        opt2.load_state_dict({"state": opt_state["state"], "param_groups": opt_state["param_groups"]})
        for a, b in zip(model.state_dict().values(), self.net.state_dict().values()):
            self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()

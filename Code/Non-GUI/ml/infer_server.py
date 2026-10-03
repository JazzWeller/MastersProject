"""The network behind `bots.inference_client.InferenceClient` (Agent
Training Plan, Milestones M2/M4/M7).

`TorchModel.predict_many(requests)` is the one entry point, usable directly
in-process (`InProcessInferenceClient` wraps any callable; this class also
exposes `predict_many`, so `InferenceServer` batches across many worker
connections into a single GPU call). A request is an
`agent.agents.requests.Request` -- plain data (an `Encoded` plus which head
to evaluate), so it pickles cleanly across the worker/server pipe, and the
engine-side code that builds it never imports torch.

`serve(checkpoint, address, authkey)` runs the dedicated inference-server
process the plan calls for: the only process that ever initializes CUDA.
"""

from __future__ import annotations

import argparse
from typing import List, Sequence, Tuple

import numpy as np
import torch

from agent.agents.requests import HEAD_BELIEF, HEAD_FIXED, HEAD_ORACLE, HEAD_POLICY, HEAD_Q, HEAD_SEQUENTIAL, HEAD_SUBSET, HEAD_TOPK, HEAD_VALUE, Request
from agent.features import Encoded

from .encode import collate
from .model import KeyForgeNet, candidates_tensor


class TorchModel:
    def __init__(self, net: KeyForgeNet, device: str = "cuda", *, amp: bool = True):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.net = net.to(self.device).eval()
        self.amp = amp and self.device.type == "cuda"
        self.evaluations = 0

    def __call__(self, request):
        return self.predict_many([request])[0]

    @torch.no_grad()
    def predict_many(self, requests: Sequence) -> List[Tuple[List[float], float]]:
        """Answers every request in one batched pass. Each head computes on
        the device; then each head's results come back to the CPU in one
        copy. Copying every request's slice on its own (a `.tolist()` or
        `float()` per request) synchronized with the device once per request:
        16 of 22 ms on a 229-request batch (Agent Observation Plan, O8)."""
        if not requests:
            return []
        reqs = [r if isinstance(r, Request) else Request(enc=r) for r in requests]
        self.evaluations += len(reqs)
        batch = collate([r.enc for r in reqs], self.device)
        by_head = {}
        for i, r in enumerate(reqs):
            by_head.setdefault(r.head, []).append(i)
        on_device = {}
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
            out = self.net.encode_state(batch)
            values_t = self.net.value(out).float()
            for head, idxs in by_head.items():
                sel = torch.tensor(idxs, device=self.device)
                if head in (HEAD_POLICY, HEAD_FIXED, HEAD_TOPK, HEAD_Q):
                    fn = {
                        HEAD_POLICY: lambda o: self.net.policy_logits(o),
                        HEAD_FIXED: lambda o: self.net.fixed_logits(o, batch),
                        HEAD_TOPK: lambda o: self.net.topk_logits(o),
                        HEAD_Q: lambda o: self.net.q_values(o),
                    }[head]
                    scores = fn(out).float()[sel]
                    if head in (HEAD_POLICY, HEAD_FIXED):
                        scores = torch.softmax(scores, dim=-1)
                    on_device[head] = scores
                elif head == HEAD_VALUE:
                    pass
                elif head == HEAD_BELIEF:
                    on_device[head] = torch.sigmoid(self.net.belief_logits(out).float()[sel])
                elif head == HEAD_ORACLE:
                    # Count vectors built on the CPU and written in one
                    # accumulate (as the learner's targets); -1 = no card.
                    V = self.net.vocab_size
                    rows, cols = [], []
                    for j, i in enumerate(idxs):
                        hand, draws = reqs[i].hidden
                        ids = reqs[i].enc.card_ids
                        for ent in hand:
                            if ent >= 0:
                                rows.append(j)
                                cols.append(ids[ent])
                        for ent in draws:
                            if ent >= 0:
                                rows.append(j)
                                cols.append(V + ids[ent])
                    hidden = torch.zeros(len(idxs), 2 * V, device=self.device)
                    if rows:
                        at = (torch.tensor(rows, device=self.device), torch.tensor(cols, device=self.device))
                        hidden.index_put_(at, torch.ones(len(rows), device=self.device), accumulate=True)
                    sub = type(out)(g=out.g[sel], h=out.h[sel], e_opt=out.e_opt[sel], option_mask=out.option_mask[sel])
                    on_device[head] = self.net.oracle_value(sub, hidden).float()
                elif head == HEAD_SUBSET:
                    rows, members, ordered, spans = [], [], [], []
                    for i in idxs:
                        c = reqs[i].candidates
                        spans.append((i, len(members), len(c)))
                        rows.extend([i] * len(c))
                        members.extend(c)
                        ordered.extend([reqs[i].ordered] * len(c))
                    width = max((n for _i, _s, n in spans), default=0)
                    if width == 0:
                        on_device[head] = (None, spans)
                        continue
                    scores = self.net.subset_scores(
                        out, torch.tensor(rows, device=self.device), candidates_tensor(members, self.device),
                        torch.tensor(ordered, device=self.device),
                    ).float()
                    # One softmax per request over its own candidates: the rows
                    # of a padded matrix, -inf where a request has none.
                    counts = torch.tensor([n for _i, _s, n in spans], device=self.device)
                    starts = torch.tensor([s for _i, s, _n in spans], device=self.device)
                    row_of = torch.repeat_interleave(torch.arange(len(spans), device=self.device), counts)
                    col_of = torch.arange(len(members), device=self.device) - torch.repeat_interleave(starts, counts)
                    padded = torch.full((len(spans), width), float("-inf"), device=self.device)
                    padded[row_of, col_of] = scores
                    on_device[head] = (torch.softmax(padded, dim=-1), spans)
                elif head == HEAD_SEQUENTIAL:
                    K = batch.options.shape[1]
                    prefix_np = np.zeros((len(idxs), K), dtype=np.float32)
                    legal_np = np.zeros((len(idxs), K + 1), dtype=bool)
                    for j, i in enumerate(idxs):
                        r = reqs[i]
                        prefix_np[j, list(r.prefix)] = 1.0
                        legal_np[j, [K if l == -1 else l for l in r.legal]] = True
                    prefix = torch.from_numpy(prefix_np).to(self.device)
                    legal = torch.from_numpy(legal_np).to(self.device)
                    logits = self.net.sequential_logits(out, sel, prefix, legal).float()
                    on_device[head] = (torch.softmax(logits, dim=-1), K)
                else:
                    raise ValueError(f"unknown head {head!r}")
        values = values_t.cpu().tolist()
        results: List = [None] * len(reqs)
        for head, idxs in by_head.items():
            if head == HEAD_VALUE:
                for i in idxs:
                    results[i] = ([], values[i])
            elif head in (HEAD_POLICY, HEAD_FIXED, HEAD_TOPK, HEAD_Q):
                rows = on_device[head].cpu().numpy()
                for j, i in enumerate(idxs):
                    results[i] = (rows[j, : reqs[i].enc.n_options].tolist(), values[i])
            elif head == HEAD_BELIEF:
                rows = on_device[head].cpu().numpy()
                for j, i in enumerate(idxs):
                    results[i] = (rows[j].tolist(), values[i])
            elif head == HEAD_ORACLE:
                ov = on_device[head].cpu().tolist()
                for j, i in enumerate(idxs):
                    results[i] = ([], ov[j])
            elif head == HEAD_SUBSET:
                probs_t, spans = on_device[head]
                probs = probs_t.cpu().numpy() if probs_t is not None else None
                for r, (i, _start, n) in enumerate(spans):
                    results[i] = (probs[r, :n].tolist() if n else [], values[i])
            elif head == HEAD_SEQUENTIAL:
                probs_t, K = on_device[head]
                probs = probs_t.cpu().numpy()
                for j, i in enumerate(idxs):
                    k = reqs[i].enc.n_options
                    results[i] = (probs[j, :k].tolist() + [float(probs[j, K])], values[i])
        return results


class ReloadingTorchModel(TorchModel):
    """A `TorchModel` that follows a run's `best.json`: at most every
    `poll_seconds` (checked between batches -- `InferenceServer` calls the
    model from one thread only), a changed checkpoint is loaded and swapped
    in, so actors pick up a promotion at their next request without
    restarting. The checkpoint path is resolved next to `best.json`."""

    def __init__(self, net, device, best_json: str, *, amp: bool = True, poll_seconds: float = 5.0):
        super().__init__(net, device, amp=amp)
        self.best_json = best_json
        self.poll_seconds = poll_seconds
        self._last_poll = 0.0
        self._current = None
        self.reloads = 0

    def _maybe_reload(self) -> None:
        import json as _json
        import os as _os
        import time as _time

        now = _time.time()
        if now - self._last_poll < self.poll_seconds:
            return
        self._last_poll = now
        try:
            with open(self.best_json, "r", encoding="utf-8") as f:
                ckpt = _json.load(f)["checkpoint"]
        except (OSError, ValueError, KeyError):
            return
        if ckpt == self._current:
            return
        from .checkpoints import CheckpointStore, load_model

        store = CheckpointStore(_os.path.join(_os.path.dirname(self.best_json), "checkpoints"))
        net, _meta, _opt = load_model(store.path_of(ckpt), device="cpu")
        self.net = net.to(self.device).eval()
        self._current = ckpt
        self.reloads += 1

    def predict_many(self, requests):
        self._maybe_reload()
        return super().predict_many(requests)


def load_torch_model(checkpoint: str, device: str = "cuda", *, amp: bool = True) -> TorchModel:
    from .checkpoints import load_model

    net, _meta, _opt = load_model(checkpoint, device="cpu")
    return TorchModel(net, device, amp=amp)


def serve(checkpoint: str, address, authkey: bytes, device: str = "cuda", max_batch_size: int = 512, window: float = 0.002,
          watch: str = None):
    from bots.inference_client import InferenceServer

    if watch:
        from .checkpoints import load_model

        net, _meta, _opt = load_model(checkpoint, device="cpu")
        model = ReloadingTorchModel(net, device, watch)
    else:
        model = load_torch_model(checkpoint, device)
    server = InferenceServer(address, authkey, model, max_batch_size=max_batch_size, batch_window_seconds=window)
    print(f"inference server on {address} ({model.device})", flush=True)
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description="Run the dedicated inference server process.")
    parser.add_argument("checkpoint")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6150)
    parser.add_argument("--authkey", default="keyforge")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--watch", default=None, help="a run's best.json: hot-reload promoted checkpoints")
    args = parser.parse_args()
    if args.watch:  # started by a run's learner: never outlive it (an orphan keeps the port)
        from agent.lifecycle import exit_with_parent

        exit_with_parent()
    serve(args.checkpoint, (args.host, args.port), args.authkey.encode(), args.device, watch=args.watch)


if __name__ == "__main__":
    main()

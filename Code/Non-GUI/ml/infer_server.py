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

import torch

from agent.agents.requests import HEAD_POLICY, HEAD_Q, HEAD_SEQUENTIAL, HEAD_SUBSET, HEAD_TOPK, HEAD_VALUE, HEAD_FIXED, HEAD_BELIEF, Request
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
        if not requests:
            return []
        reqs = [r if isinstance(r, Request) else Request(enc=r) for r in requests]
        self.evaluations += len(reqs)
        batch = collate([r.enc for r in reqs], self.device)
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
            out = self.net.encode_state(batch)
            values = self.net.value(out).float()
            results: List = [None] * len(reqs)
            by_head = {}
            for i, r in enumerate(reqs):
                by_head.setdefault(r.head, []).append(i)
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
                    for j, i in enumerate(idxs):
                        k = reqs[i].enc.n_options
                        results[i] = (scores[j, :k].tolist(), float(values[i]))
                elif head == HEAD_VALUE:
                    for i in idxs:
                        results[i] = ([], float(values[i]))
                elif head == HEAD_BELIEF:
                    logits = self.net.belief_logits(out).float()
                    for i in idxs:
                        results[i] = (torch.sigmoid(logits[i]).tolist(), float(values[i]))
                elif head == HEAD_SUBSET:
                    rows, members, ordered, spans = [], [], [], []
                    for i in idxs:
                        c = reqs[i].candidates
                        spans.append((i, len(members), len(c)))
                        rows.extend([i] * len(c))
                        members.extend(c)
                        ordered.extend([reqs[i].ordered] * len(c))
                    scores = self.net.subset_scores(
                        out, torch.tensor(rows, device=self.device), candidates_tensor(members, self.device),
                        torch.tensor(ordered, device=self.device),
                    ).float()
                    for i, start, n in spans:
                        results[i] = (torch.softmax(scores[start : start + n], dim=0).tolist(), float(values[i]))
                elif head == HEAD_SEQUENTIAL:
                    K = batch.options.shape[1]
                    prefix = torch.zeros(len(idxs), K, device=self.device)
                    legal = torch.zeros(len(idxs), K + 1, dtype=torch.bool, device=self.device)
                    for j, i in enumerate(idxs):
                        r = reqs[i]
                        for p in r.prefix:
                            prefix[j, p] = 1.0
                        for l in r.legal:
                            legal[j, K if l == -1 else l] = True
                    logits = self.net.sequential_logits(out, sel, prefix, legal).float()
                    probs = torch.softmax(logits, dim=-1)
                    for j, i in enumerate(idxs):
                        k = reqs[i].enc.n_options
                        results[i] = (probs[j, :k].tolist() + [float(probs[j, K])], float(values[i]))
                else:
                    raise ValueError(f"unknown head {head!r}")
        return results


def load_torch_model(checkpoint: str, device: str = "cuda", *, amp: bool = True) -> TorchModel:
    from .checkpoints import load_model

    net, _meta, _opt = load_model(checkpoint, device="cpu")
    return TorchModel(net, device, amp=amp)


def serve(checkpoint: str, address, authkey: bytes, device: str = "cuda", max_batch_size: int = 512, window: float = 0.002):
    from bots.inference_client import InferenceServer

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
    args = parser.parse_args()
    serve(args.checkpoint, (args.host, args.port), args.authkey.encode(), args.device)


if __name__ == "__main__":
    main()

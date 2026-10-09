"""The v2 network behind the inference pipe (Agent Observation Plan, O8).

`TorchModelV2.predict_many(requests)` answers `RequestV2`s as
`ml.infer_server.TorchModel` answers v1's `Request`s, one `(scores, value)`
pair each, in one batched pass.

**The prefix cache.** A request's history is a `HistoryRef`: the key of a
prefix (a search's history up to its real decision) plus the request's own
suffix (what its world added). The server keeps an LRU of prefixes:
- their rows (CPU), from which `joint` rebuilds the whole history;
- for `stream`, each layer's keys and values (on the device), computed once
  per prefix: a leaf then runs only its state tokens and its suffix rows
  (`KeyForgeNetV2.encode_state(batch, past)`), with the outputs the whole
  sequence gives.

**Compiled** (`compile`, default on CUDA): the input embedding goes
through `torch.compile(dynamic=True)` (`KeyForgeNetV2.compile_inputs`) -- one
compile serves every batch shape. The eager v2 forward is launch-bound
(~350 kernels, most of them in the embedding).

A request naming a prefix the server doesn't hold is answered
`PREFIX_MISSING`; the client resends it with the prefix's rows. A request
carrying the whole history as `HistoryRows` bytes (the naive path) is
answered from it directly, and is the reference the cache is tested
against.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch._dynamo

from agent import spec_v2 as S
from agent.agents.requests import (HEAD_BELIEF, HEAD_ORACLE, HEAD_POLICY, HEAD_Q, HEAD_SEQUENTIAL, HEAD_SUBSET,
                                   HEAD_TOPK, HEAD_VALUE, PREFIX_MISSING, HistoryRef)
from agent.history import N_RF, N_RI, HistoryRows

from .encode_v2 import collate_v2
from .model import candidates_tensor
from .model_v2 import VOCAB, KeyForgeNetV2, PrefixPast

_CARD = S.ENTITY.i["card"]


def _concat(a: HistoryRows, b: HistoryRows) -> HistoryRows:
    return HistoryRows(a.n + b.n, a.ints + b.ints, a.floats + b.floats, a.pointers + b.pointers, a.first)


class _Prefix:
    __slots__ = ("rows", "kv", "nbytes")

    def __init__(self, rows: HistoryRows):
        self.rows = rows
        self.kv = None  # stream: per layer (k, v) [1, heads, n, d / heads]
        self.nbytes = len(rows.ints) * 8 + len(rows.floats) * 4 + len(rows.pointers) * 8


class PrefixStore:
    """An LRU of prefixes, bounded by bytes (rows plus cached keys and
    values)."""

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self.data: "OrderedDict[str, _Prefix]" = OrderedDict()
        self.bytes = 0
        self.hits = 0
        self.misses = 0
        self.stored = 0

    def get(self, key: str) -> Optional[_Prefix]:
        e = self.data.get(key)
        if e is None:
            self.misses += 1
            return None
        self.data.move_to_end(key)
        self.hits += 1
        return e

    def put(self, key: str, rows: HistoryRows) -> _Prefix:
        e = self.data.get(key)
        if e is None:
            e = self.data[key] = _Prefix(rows)
            self.bytes += e.nbytes
            self.stored += 1
        self.data.move_to_end(key)
        return e

    def add_kv(self, e: _Prefix, kv) -> None:
        e.kv = kv
        extra = sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in kv)
        e.nbytes += extra
        self.bytes += extra

    def trim(self, keep: set) -> None:
        while self.bytes > self.max_bytes and len(self.data) > len(keep):
            key = next(k for k in self.data if k not in keep)
            self.bytes -= self.data.pop(key).nbytes


def _rows_tensors(rows: List[HistoryRows]):
    B = len(rows)
    P = max(1, max(r.n for r in rows))
    Q = max(1, max(len(r.pointers) // 3 for r in rows))
    ints = np.zeros((B, P, N_RI), np.int64)
    floats = np.zeros((B, P, N_RF), np.float32)
    ptr = np.full((B, Q, 3), -1, np.int64)
    mask = np.zeros((B, P), bool)
    for b, r in enumerate(rows):
        if r.n:
            ints[b, :r.n] = np.frombuffer(r.ints.tobytes(), np.int64).reshape(r.n, N_RI)
            floats[b, :r.n] = np.frombuffer(r.floats.tobytes(), np.float32).reshape(r.n, N_RF)
            mask[b, :r.n] = True
        q = np.frombuffer(r.pointers.tobytes(), np.int64).reshape(-1, 3)
        ptr[b, :len(q)] = q
    t = torch.from_numpy
    return t(ints), t(floats), t(ptr), t(mask)


class TorchModelV2:
    def __init__(self, net: KeyForgeNetV2, device: str = "cuda", *, amp: bool = True,
                 prefix_bytes: int = 1536 * 2 ** 20, compile: Optional[bool] = None):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.net = net.to(self.device).eval()
        self.amp = amp and self.device.type == "cuda"
        self.prefixes = PrefixStore(prefix_bytes)
        self.evaluations = 0
        self.missing = 0
        if compile is None:
            compile = self.device.type == "cuda"
        self._encode = self.net.encode_state
        self.compiled = bool(compile)
        if compile:
            # Every padded dimension (and the batch) has at least 2 rows
            # (`min_rows`): a 0 or 1 would compile a graph of its own, and
            # each compile takes ~20 s. The limits are a margin, not a plan.
            for name in ("recompile_limit", "cache_size_limit"):
                if hasattr(torch._dynamo.config, name):
                    setattr(torch._dynamo.config, name, max(getattr(torch._dynamo.config, name), 32))
            # Without duck sizing, dimensions that happen to be equal in the
            # first batch don't share a symbol (and recompile once they
            # differ): over 150 batches of random live-like shapes, one ~20 s
            # compile plus a few of 1-2 s, against 6-10 compiles with it.
            import torch.fx.experimental._config as fx_config

            fx_config.use_duck_shape = False
            self.net.compile_inputs()

    @property
    def stream(self) -> bool:
        return self.net.history_arch == "stream"

    def __call__(self, request):
        return self.predict_many([request])[0]

    # ------------------------------------------------------- histories
    def _histories(self, reqs) -> Tuple[list, list, Dict[str, _Prefix]]:
        """Each request's history input for `collate_v2`, and (stream) its
        prefix key or None. A request whose prefix is missing gets None in
        the first list."""
        hists, keys, used = [], [], {}
        for r in reqs:
            h = r.history
            if isinstance(h, HistoryRef):
                e = self.prefixes.put(h.key, HistoryRows.from_bytes(h.prefix)) if h.prefix is not None \
                    else self.prefixes.get(h.key)
                if e is None:
                    hists.append(None)
                    keys.append(None)
                    continue
                used[h.key] = e
                suffix = HistoryRows.from_bytes(h.suffix)
                if self.stream:
                    hists.append(suffix)
                    keys.append(h.key)
                else:
                    hists.append(_concat(e.rows, suffix))
                    keys.append(None)
            elif isinstance(h, (bytes, bytearray)):
                hists.append(HistoryRows.from_bytes(h))
                keys.append(None)
            else:
                hists.append(h)
                keys.append(None)
        return hists, keys, used

    def _past(self, batch, keys: List[Optional[str]], used: Dict[str, _Prefix]) -> PrefixPast:
        """The batch's `PrefixPast`: each distinct prefix's keys and values
        (computed now if new), padded and gathered per request; a request
        with no prefix gets an empty one."""
        net = self.net
        card_ids = batch.blocks["entity"].ints[..., _CARD]
        new = [k for k in dict.fromkeys(k for k in keys if k is not None) if used[k].kv is None]
        if new:
            first = {k: keys.index(k) for k in new}
            ints, floats, ptr, mask = (t.to(self.device) for t in _rows_tensors([used[k].rows for k in new]))
            ids = card_ids[[first[k] for k in new]]
            pp = net.history_prefix(ids, ints, floats, ptr, mask)
            for j, k in enumerate(new):
                n = used[k].rows.n
                self.prefixes.add_kv(used[k], [(kk[j:j + 1, :, :n].clone(), vv[j:j + 1, :, :n].clone())
                                               for kk, vv in pp.kv])
        uniq = list(dict.fromkeys(k for k in keys if k is not None))
        B = len(keys)
        heads = net.trunk.layers[0].self_attn.heads
        dh = net.d // heads
        P = max([2 if self.compiled else 1] + [used[k].rows.n for k in uniq])
        dtype = used[uniq[0]].kv[0][0].dtype if uniq else torch.float32
        slot = {k: i + 1 for i, k in enumerate(uniq)}  # 0: the empty prefix
        index = torch.tensor([slot.get(k, 0) if k is not None else 0 for k in keys], device=self.device)
        kv = []
        for layer in range(len(net.trunk.layers)):
            K = torch.zeros(len(uniq) + 1, heads, P, dh, dtype=dtype, device=self.device)
            V = torch.zeros_like(K)
            for k in uniq:
                kk, vv = used[k].kv[layer]
                K[slot[k], :, :kk.shape[2]] = kk[0]
                V[slot[k], :, :vv.shape[2]] = vv[0]
            kv.append((K.index_select(0, index), V.index_select(0, index)))
        n = torch.tensor([used[k].rows.n if k is not None else 0 for k in keys], device=self.device)
        valid = torch.arange(P, device=self.device).unsqueeze(0) < n.unsqueeze(1)
        return PrefixPast(kv=kv, valid=valid, n=n)

    # ---------------------------------------------------------- answer
    @torch.no_grad()
    def predict_many(self, requests: Sequence) -> List[Tuple[Optional[List[float]], Optional[float]]]:
        if not requests:
            return []
        reqs = list(requests)
        hists, keys, used = self._histories(reqs)
        results: List = [None] * len(reqs)
        live = []
        for i, r in enumerate(reqs):
            if isinstance(r.history, HistoryRef) and hists[i] is None:
                results[i] = PREFIX_MISSING
                self.missing += 1
            else:
                live.append(i)
        if live:
            pad = self.compiled and len(live) == 1  # a batch of one would compile its own graph
            idx = live + live[:1] if pad else live
            answers = self._answer([reqs[i] for i in idx], [hists[i] for i in idx], [keys[i] for i in idx], used)
            for i, a in zip(live, answers):
                results[i] = a
        self.prefixes.trim(set(used))
        return results

    def _answer(self, reqs, hists, keys, used):
        self.evaluations += len(reqs)
        net = self.net
        batch = collate_v2([(r.enc, h, r.turn) for r, h in zip(reqs, hists)],
                           min_rows=2 if self.compiled else 1).to(self.device)
        by_head: Dict[str, List[int]] = {}
        for i, r in enumerate(reqs):
            by_head.setdefault(r.head, []).append(i)
        on_device = {}
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
            # (a batch of whole histories only runs the plain forward)
            past = self._past(batch, keys, used) if self.stream and any(k is not None for k in keys) else None
            out = self._encode(batch, past)
            values_t = net.value(out).float()
            for head, idxs in by_head.items():
                sel = torch.tensor(idxs, device=self.device)
                if head in (HEAD_POLICY, HEAD_TOPK, HEAD_Q):
                    fn = {HEAD_POLICY: net.policy_logits, HEAD_TOPK: net.topk_logits, HEAD_Q: net.q_values}[head]
                    scores = fn(out).float()[sel]
                    on_device[head] = torch.softmax(scores, dim=-1) if head == HEAD_POLICY else scores
                elif head == HEAD_VALUE:
                    pass
                elif head == HEAD_BELIEF:
                    zones, draw = net.belief_v2(out, batch)
                    on_device[head] = torch.cat([torch.softmax(zones.float(), -1), torch.sigmoid(draw.float()).unsqueeze(-1)],
                                                dim=-1)[sel]
                elif head == HEAD_ORACLE:
                    card_ids = batch.blocks["entity"].ints[..., _CARD].cpu().numpy()
                    rows, cols = [], []
                    for j, i in enumerate(idxs):
                        for part, ents in enumerate(reqs[i].hidden):
                            for ent in ents:
                                if ent >= 0:
                                    rows.append(j)
                                    cols.append(part * VOCAB + int(card_ids[i, ent]))
                    hidden = torch.zeros(len(idxs), 3 * VOCAB, device=self.device)
                    if rows:
                        at = (torch.tensor(rows, device=self.device), torch.tensor(cols, device=self.device))
                        hidden.index_put_(at, torch.ones(len(rows), device=self.device), accumulate=True)
                    sub = type(out)(g=out.g[sel], h=out.h[sel], e_opt=out.e_opt[sel], option_mask=out.option_mask[sel])
                    on_device[head] = net.oracle_value(sub, hidden).float()
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
                    scores = net.subset_scores(out, torch.tensor(rows, device=self.device),
                                               candidates_tensor(members, self.device),
                                               torch.tensor(ordered, device=self.device)).float()
                    padded = torch.full((len(spans), width), float("-inf"), device=self.device)
                    for r_, (_i, s, n) in enumerate(spans):
                        padded[r_, :n] = scores[s:s + n]
                    on_device[head] = (torch.softmax(padded, dim=-1), spans)
                elif head == HEAD_SEQUENTIAL:
                    K = batch.blocks["option"].mask.shape[1]
                    prefix_np = np.zeros((len(idxs), K), dtype=np.float32)
                    legal_np = np.zeros((len(idxs), K + 1), dtype=bool)
                    for j, i in enumerate(idxs):
                        r = reqs[i]
                        prefix_np[j, list(r.prefix)] = 1.0
                        legal_np[j, [K if l == -1 else l for l in r.legal]] = True
                    logits = net.sequential_logits(out, sel, torch.from_numpy(prefix_np).to(self.device),
                                                   torch.from_numpy(legal_np).to(self.device)).float()
                    on_device[head] = (torch.softmax(logits, dim=-1), K)
                else:
                    raise ValueError(f"unknown head {head!r}")
        values = values_t.cpu().tolist()
        results: List = [None] * len(reqs)
        for head, idxs in by_head.items():
            if head == HEAD_VALUE:
                for i in idxs:
                    results[i] = ([], values[i])
            elif head in (HEAD_POLICY, HEAD_TOPK, HEAD_Q):
                rows = on_device[head].cpu().numpy()
                for j, i in enumerate(idxs):
                    results[i] = (rows[j, :reqs[i].enc.n_options].tolist(), values[i])
            elif head == HEAD_BELIEF:
                rows = on_device[head].cpu().numpy()
                for j, i in enumerate(idxs):
                    results[i] = (rows[j].reshape(-1).tolist(), values[i])
            elif head == HEAD_ORACLE:
                ov = on_device[head].cpu().tolist()
                for j, i in enumerate(idxs):
                    results[i] = ([], ov[j])
            elif head == HEAD_SUBSET:
                probs_t, spans = on_device[head]
                probs = probs_t.cpu().numpy() if probs_t is not None else None
                for r_, (i, _s, n) in enumerate(spans):
                    results[i] = (probs[r_, :n].tolist() if n else [], values[i])
            elif head == HEAD_SEQUENTIAL:
                probs_t, K = on_device[head]
                probs = probs_t.cpu().numpy()
                for j, i in enumerate(idxs):
                    k = reqs[i].enc.n_options
                    results[i] = (probs[j, :k].tolist() + [float(probs[j, K])], values[i])
        return results


def serve_v2(net: KeyForgeNetV2, address, authkey: bytes, device: str = "cuda", max_batch_size: int = 512,
             window: float = 0.002):
    from bots.inference_client import InferenceServer

    model = TorchModelV2(net, device)
    server = InferenceServer(address, authkey, model, max_batch_size=max_batch_size, batch_window_seconds=window)
    print(f"v2 inference server on {address} ({model.device}, history {net.history_arch})", flush=True)
    server.serve_forever()


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run a v2 inference server (O8).")
    parser.add_argument("--checkpoint", default=None, help="a network-v2 checkpoint (ml.bc_train_v2)")
    parser.add_argument("--net-cfg", default=None,
                        help="the network config as JSON: an untrained network, for throughput measurements")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6150)
    parser.add_argument("--authkey", default="keyforge")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.checkpoint:
        from .checkpoints import load_model_v2

        net, _meta = load_model_v2(args.checkpoint)
    elif args.net_cfg:
        net = KeyForgeNetV2(json.loads(args.net_cfg))
    else:
        parser.error("--checkpoint or --net-cfg is required")
    serve_v2(net, (args.host, args.port), args.authkey.encode(), args.device)


if __name__ == "__main__":
    main()

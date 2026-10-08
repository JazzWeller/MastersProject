"""The v2 inference server's throughput, end to end (Agent Observation Plan,
O8). WSL, CUDA.

    python -m tools.bench_infer_v2 [--sizes 128/4,256/8] [--batches 64,512] [--suffix 16] [--per-prefix 8]

`TorchModelV2.predict_many` on real v2 requests (fuzz positions, both seats):
collation, the network and the copy back, per history architecture. Each
position's history is split into a prefix (cached, as a search's) and its
last `--suffix` rows (what a world added); batches group `--per-prefix`
requests per prefix, as an actor's leaves share their search's prefix.
`stream` is measured with its prefix keys and values cached (warm) and
whole (`stream_whole`), `joint` from cached prefix rows.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time

import torch

from agent.agents.requests import HEAD_VALUE, HistoryRef, RequestV2
from agent.history import HistoryFolded, HistoryRows, N_RF, N_RI
from ml.infer_v2 import TorchModelV2
from ml.model_v2 import KeyForgeNetV2
from tests.test_agent_observation_o7 import _items

ARCHS = ("none", "summary", "turn_tokens", "joint", "stream", "stream_whole")


def _split(h, suffix: int):
    rows = h.to_rows()
    k = max(0, rows.n - suffix)
    p = rows.pointers
    cut = next((t for t in range(0, len(p), 3) if p[t] >= k), len(p))
    pre = HistoryRows(k, rows.ints[:k * N_RI], rows.floats[:k * N_RF], p[:cut], 0)
    suf = HistoryRows(rows.n - k, rows.ints[k * N_RI:], rows.floats[k * N_RF:], p[cut:], k)
    return pre, suf


def _requests(items, arch: str, size: int, suffix: int, per_prefix: int):
    reqs = []
    i = 0
    while len(reqs) < size:
        enc, h, turn = items[i % len(items)]
        if arch in ("joint", "stream"):
            pre, suf = _split(h, suffix)
            key = hashlib.blake2b(pre.to_bytes(), digest_size=16).hexdigest()  # as a real key: by content
            for j in range(per_prefix):
                reqs.append(RequestV2(enc=enc, head=HEAD_VALUE, turn=turn,
                                      history=HistoryRef(key, pre.n, suf.to_bytes(), pre.to_bytes() if j == 0 else None)))
        elif arch == "stream_whole":
            reqs.extend(RequestV2(enc=enc, head=HEAD_VALUE, turn=turn, history=h.to_rows().to_bytes())
                        for _ in range(per_prefix))
        elif arch in ("turn_tokens", "summary"):
            reqs.extend(RequestV2(enc=enc, head=HEAD_VALUE, turn=turn, history=HistoryFolded.of(h, turn))
                        for _ in range(per_prefix))
        else:
            reqs.extend(RequestV2(enc=enc, head=HEAD_VALUE, turn=turn) for _ in range(per_prefix))
        i += 1
    return reqs[:size]


def bench(size: str, arch: str, batches, items, suffix: int, per_prefix: int, rounds: int = 10) -> list:
    d, layers = (int(x) for x in size.split("/"))
    net_arch = "stream" if arch == "stream_whole" else arch
    cfg = {"d_model": d, "heads": max(4, d // 64), "ff": 4 * d, "layers": layers, "dropout": 0.0,
           "history_arch": net_arch, "attention_kernel": "efficient"}
    rows = []
    for bs in batches:
        model = TorchModelV2(KeyForgeNetV2(cfg), "cuda")
        reqs = _requests(items, arch, bs, suffix, per_prefix)
        torch.cuda.reset_peak_memory_stats()
        try:
            model.predict_many(reqs)  # warm: prefixes stored, keys and values computed
            for r in reqs:
                if isinstance(r.history, HistoryRef):
                    r.history.prefix = None
            model.predict_many(reqs)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(rounds):
                model.predict_many(reqs)
            torch.cuda.synchronize()
            rate = bs * rounds / (time.perf_counter() - t0)
        except torch.cuda.OutOfMemoryError:
            rate = float("nan")
        rows.append({"size": size, "arch": arch, "batch": bs, "evals_per_s": round(rate) if rate == rate else None,
                     "ms_per_batch": round(1000 * bs / rate, 1) if rate == rate else None,
                     "peak_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)})
        del model
        torch.cuda.empty_cache()
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sizes", default="128/4,256/8")
    ap.add_argument("--batches", default="64,512")
    ap.add_argument("--archs", default=",".join(ARCHS))
    ap.add_argument("--suffix", type=int, default=16)
    ap.add_argument("--per-prefix", type=int, default=8)
    args = ap.parse_args(argv)
    torch.cuda.set_per_process_memory_fraction(0.95)
    items, _ = _items(n_games=3, every=5)
    print(f"{len(items)} positions; history rows: mean {sum(it[1].n for it in items) / len(items):.0f}; "
          f"suffix {args.suffix} rows, {args.per_prefix} requests per prefix", flush=True)
    for size in args.sizes.split(","):
        for arch in args.archs.split(","):
            for r in bench(size, arch, [int(b) for b in args.batches.split(",")], items, args.suffix, args.per_prefix):
                print(r, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

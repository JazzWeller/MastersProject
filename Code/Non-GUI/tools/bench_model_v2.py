"""Throughput of network v2, per history architecture (Agent Observation
Plan, Milestone O7). WSL, CUDA.

    python -m tools.bench_model_v2 [--sizes 128/4,256/8] [--batches 64,512]

Real v2 inputs (fuzz positions, both seats, with their histories),
replicated to each batch size. Reports inference evaluations/s (bf16
autocast, no grad) and training samples/s (a policy + value + belief step,
AdamW) at each batch size, the token count, and peak memory.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
from torch.nn import functional as F

from ml.encode_v2 import collate_v2
from ml.model_v2 import KeyForgeNetV2, param_count
from tests.test_agent_observation_o7 import _items

ARCHS = ("none", "summary", "turn_tokens", "joint", "stream")


def _batch(items, size):
    reps = (items * (size // len(items) + 1))[:size]
    return collate_v2(reps)


def _loss(net, batch):
    out = net.encode_state(batch)
    logits = net.policy_logits(out)
    target = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    zones, draw = net.belief_v2(out, batch)
    return (F.cross_entropy(logits.float(), target) + net.value(out).float().pow(2).mean()
            + zones.float().logsumexp(-1).mean() + draw.float().mean() * 0)


def bench(size: str, arch: str, batches, items, device, steps: int = 10) -> list:
    d, layers = (int(x) for x in size.split("/"))
    cfg = {"d_model": d, "heads": max(4, d // 64), "ff": 4 * d, "layers": layers, "dropout": 0.0,
           "history_arch": arch, "attention_kernel": "efficient"}
    net = KeyForgeNetV2(cfg).to(device)
    rows = []
    for bs in batches:
        batch = _batch(items, bs).to(device)
        tokens = (1 + 72 + sum(batch.blocks[n].mask.shape[1] for n in ("effect", "resolution", "cleanup", "match", "option"))
                  + (batch.hist_mask.shape[1] if arch in ("joint", "stream") else 0)
                  + (batch.turn_mask.shape[1] if arch == "turn_tokens" else 0))
        torch.cuda.reset_peak_memory_stats()
        net.eval()
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for _ in range(3):
                    net.value(net.encode_state(batch))
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(steps):
                    net.value(net.encode_state(batch))
                torch.cuda.synchronize()
                infer = bs * steps / (time.perf_counter() - t0)
        except torch.cuda.OutOfMemoryError:
            infer = float("nan")
        net.train()
        opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
        try:
            for k in range(3 + steps):
                if k == 3:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = _loss(net, batch)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            torch.cuda.synchronize()
            train = bs * steps / (time.perf_counter() - t0)
        except torch.cuda.OutOfMemoryError:
            train = float("nan")
        del opt
        torch.cuda.empty_cache()
        rows.append({"size": size, "arch": arch, "batch": bs, "tokens": tokens, "params": param_count(net),
                     "infer_per_s": round(infer) if infer == infer else None, "train_per_s": round(train) if train == train else None,
                     "peak_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)})
        del batch
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sizes", default="128/4,256/8")
    ap.add_argument("--batches", default="64,512")
    ap.add_argument("--archs", default=",".join(ARCHS))
    args = ap.parse_args(argv)
    device = torch.device("cuda")
    # Out of memory is a result, not a slow run: without a cap the driver
    # spills past the card into system memory (seen: 21 GiB "allocated").
    torch.cuda.set_per_process_memory_fraction(0.95)
    items, _ = _items(n_games=3, every=5)
    print(f"{len(items)} positions; history rows: mean {sum(it[1].n for it in items) / len(items):.0f}, "
          f"max {max(it[1].n for it in items)}")
    for size in args.sizes.split(","):
        for arch in args.archs.split(","):
            for r in bench(size, arch, [int(b) for b in args.batches.split(",")], items, device):
                print(r, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

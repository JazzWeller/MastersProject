#!/usr/bin/env python3
"""Linear probes on trunk embeddings (Agent Observation Plan, O10): what
each trained network's representation carries, shown directly. WSL, CUDA.

    python -m tools.probe_v2 --runs tier0b-a0,tier0b2-a1,tier0b2-a8 [--positions 40000]

For each run (v1 or v2, found from its checkpoint), embeddings are taken on
validation positions of its own corpus (the same games for both: one by-game
split over the same records) and a logistic readout is fitted on 70% of the
games, scored on the other 30%:

- `opp_hand`: is this hidden opponent card in their hand? (the entity's
  final embedding; hidden = O3's prior > 0 in v2, v1's "opponent unseen")
- `my_next_draw`: is this card of mine my next draw?
- `opp_next_draw` (v2): is this hidden opponent card their next draw?
- `outcome`: does the viewer win? (the global token)

Reported as log-loss against the base rate on the same rows; lower is
better.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch.nn import functional as F


def _checkpoint(run_id: str) -> str:
    from tools.eval_v2 import _checkpoint as ck

    return ck(run_id)


def _collect_v2(model, corpus, n: int, device):
    from agent import spec_v2 as S
    from ml.bc_train_v2 import _hidden_rows

    sel = corpus.indices(split=1).subsample(n)
    feats = {k: [] for k in ("opp_hand", "my_next_draw", "opp_next_draw", "outcome")}
    owner = S.ENTITY.i["owner"]
    for batch, tg in corpus.iterate(sel, 256, shuffle=False):
        batch, tg = batch.to(device), tg.to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.encode_state(batch)
        h, g = out.h.float(), out.g.float()
        game = torch.as_tensor(tg.game, device=device)
        hidden = _hidden_rows(batch)
        b, e = hidden.nonzero(as_tuple=True)
        feats["opp_hand"].append((h[b, e], (tg.zone[b, e] == 1).float(), game[b]))
        top = tg.opp_deck[:, 0]
        feats["opp_next_draw"].append((h[b, e], (top[b] == e).float(), game[b]))
        mine = batch.blocks["entity"].ints[..., owner] == 0
        mine &= tg.zone == 3
        b2, e2 = mine.nonzero(as_tuple=True)
        feats["my_next_draw"].append((h[b2, e2], (tg.my_draws[b2, 0] == e2).float(), game[b2]))
        feats["outcome"].append((g, (tg.z > 0).float(), game))
    return feats


def _collect_v1(model, corpus, n: int, device):
    from keyforge.infoset import ZONE

    idx = corpus.indices(split=1)
    idx = np.sort(np.random.default_rng(0).choice(idx, min(n, len(idx)), replace=False))
    feats = {k: [] for k in ("opp_hand", "my_next_draw", "outcome")}
    for batch, tg in corpus.iterate(idx, 256, shuffle=False):
        batch = batch.to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.encode_state(batch)
        h, g = out.h.float(), out.g.float()
        game = torch.as_tensor(tg.game, device=device)
        unseen = batch.zones == ZONE["opp_unseen"]
        b, e = unseen.nonzero(as_tuple=True)
        feats["opp_hand"].append((h[b, e], tg.opp_hand.to(device)[b, e], game[b]))
        mydeck = batch.zones == ZONE["my_deck_unordered"]
        b2, e2 = mydeck.nonzero(as_tuple=True)
        nd = tg.next_draws.to(device)
        feats["my_next_draw"].append((h[b2, e2], (nd[b2, 0] == e2).float(), game[b2]))
        feats["outcome"].append((g, (tg.z.to(device) > 0).float(), game))
    return feats


def _fit(x: torch.Tensor, y: torch.Tensor, game: torch.Tensor) -> dict:
    """Logistic readout on 70% of the games, scored on the rest."""
    test = (game % 10) >= 7
    if test.sum() < 50 or (~test).sum() < 50:
        return {"n": int(len(y))}
    mu, sd = x[~test].mean(0), x[~test].std(0).clamp(min=1e-4)
    x = (x - mu) / sd
    w = torch.zeros(x.shape[1], device=x.device, requires_grad=True)
    b = torch.zeros(1, device=x.device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=0.5, max_iter=200, line_search_fn="strong_wolfe")
    xt, yt = x[~test], y[~test]

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(xt @ w + b, yt) + 1e-4 * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        ll = float(F.binary_cross_entropy_with_logits(x[test] @ w + b, y[test]))
        p = float(yt.mean().clamp(1e-4, 1 - 1e-4))
        base = float(F.binary_cross_entropy(torch.full_like(y[test], p), y[test]))
    return {"n": int(len(y)), "positive_rate": round(float(y.mean()), 4), "logloss": round(ll, 4),
            "base_logloss": round(base, 4)}


def probe(run_id: str, n: int, device) -> dict:
    from ml.checkpoints import CheckpointMismatch, load_model, load_model_v2

    path = _checkpoint(run_id)
    try:
        model, _meta = load_model_v2(path, device=device)
        v2 = True
    except CheckpointMismatch:
        model, _meta, _opt = load_model(path, device=device)
        v2 = False
    model.eval()
    if v2:
        from ml.dataset_v2 import PackedCorpusV2

        arch = model.history_arch
        form = {"joint": "rows", "stream": "rows", "turn_tokens": "folded", "summary": "folded"}.get(arch, "none")
        corpus = PackedCorpusV2(os.path.expanduser("~/keyforge-data/bc/d973255d6dd1b036/packed_v2"), history=form)
        feats = _collect_v2(model, corpus, n, device)
    else:
        from ml.dataset import Corpus

        corpus = Corpus.from_dir(os.path.expanduser("~/keyforge-data/bc/d973255d6dd1b036/encoded"))
        feats = _collect_v1(model, corpus, n, device)
    out = {"run": run_id, "network": "v2" if v2 else "v1"}
    for k, parts in feats.items():
        x = torch.cat([p[0] for p in parts])
        y = torch.cat([p[1] for p in parts])
        g = torch.cat([p[2] for p in parts])
        out[k] = _fit(x, y, g)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--positions", type=int, default=40000)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    device = torch.device("cuda")
    results = []
    for r in args.runs.split(","):
        res = probe(r, args.positions, device)
        print(json.dumps(res), flush=True)
        results.append(res)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Targeted probes (Agent Observation Plan, O10): how each v2 rung does on
the validation positions where one new kind of information matters. WSL,
CUDA.

    python -m tools.probe_targeted --runs tier0b2-a1,tier0b2-a2,tier0b2-a7 [--positions 120000]

The positions are found in the v2 encoding itself:

- `known_in_their_hand`: a card of theirs the viewer knows is in their hand
  (bounced, returned, revealed: O2's exact knowledge).
- `key_cost_effect`: Lash of Broken Dreams' pending +3 key cost (a duration
  effect on `KeyForgeCost` of value 3; the +1 rows are other effects).
- `known_top_card`: a known deck position (a card's place in a deck), with
  the next-draw head's log-loss on their known next draw as the readout.
  Fignor-vs-Igor games never reveal one, so on Tier 0's corpus it is empty.
- `mid_resolution`: an ability part-way through resolving -- resolution
  frames 5 or more deep (every position has the turn's own 3-5 frames).
- `all`: every scored position, the reference.

(The belief head adds O3's exact prior, so on a card known to be in their
hand every rung is right by construction: not a readout.)

For each condition and run: positions, top-1 on single-choice decisions,
value log-loss and the readouts. Every run reads the same validation
positions (a fixed subsample of the packed corpus's by-game split 1). A1
(no knowledge inputs) against A2 isolates the knowledge features; A1
already has the complete state (resolution, effects). v1 reads a different
corpus and is not in this table.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F

CONDITIONS = ("all", "known_in_their_hand", "key_cost_effect", "known_top_card", "mid_resolution")


def conditions(batch) -> dict:
    """[B] bool per condition, plus the entity rows each readout scores."""
    from agent import spec_v2 as S
    from agent.features_v2 import _V

    ent = batch.blocks["entity"]
    fi, ii = S.ENTITY.f, S.ENTITY.i
    theirs = ent.ints[..., ii["owner"]] == 1
    in_hand = theirs & (ent.floats[..., fi["p_hand"]] >= 0.999) & ent.mask
    next_draw = theirs & (ent.floats[..., fi["p_next_draw"]] >= 0.999) & ent.mask
    known_pos = ((ent.ints[..., ii["deck_pos"]] > 0) & ent.mask).any(1)
    eff = batch.blocks["effect"]
    E, Ef = S.BLOCK_BY_NAME["effect"].i, S.BLOCK_BY_NAME["effect"].f
    lash = ((eff.ints[..., E["variable"]] == _V["effect_variables"].id("KeyForgeCost"))
            & (eff.floats[..., Ef["value"]] == 3) & eff.mask).any(1)
    res = batch.blocks["resolution"]
    R = S.BLOCK_BY_NAME["resolution"].i
    deep = ((res.ints[..., R["depth"]] >= 5) & res.mask).any(1)
    return {"all": torch.ones_like(deep), "known_in_their_hand": in_hand.any(1), "key_cost_effect": lash,
            "known_top_card": known_pos, "mid_resolution": deep, "_next_draw": next_draw}


class _Acc:
    def __init__(self):
        self.n = self.hit = self.n_single = self.draw_n = 0
        self.draw_ll = 0.0
        self.vp, self.vy = [], []

    def report(self) -> dict:
        from ml.bc_train import _logloss

        vp, vy = (np.concatenate(self.vp), np.concatenate(self.vy)) if self.vp else (np.zeros(0), np.zeros(0))
        return {"n": self.n, "top1": round(self.hit / self.n_single, 4) if self.n_single else None,
                "n_single": self.n_single, "value_logloss": round(_logloss((vp + 1) / 2, vy), 4) if len(vp) else None,
                "next_draw_logloss": round(self.draw_ll / self.draw_n, 4) if self.draw_n else None}


@torch.no_grad()
def score(model, corpus, sel, device, max_tokens) -> dict:
    from ml.bc_train import Prefetcher
    from ml.bc_train_v2 import pieces_of

    model.eval()
    acc = {k: _Acc() for k in CONDITIONS}
    t0 = time.perf_counter()
    for k_batch, (whole, wtg) in enumerate(Prefetcher(corpus.iterate(sel, 256, shuffle=False))):
        whole, wtg = whole.to(device), wtg.to(device)
        for lo, hi in pieces_of(whole, max_tokens):
            batch, tg = whole.slice(lo, hi), wtg.slice(lo, hi)
            out = model.encode_state(batch)
            c = conditions(batch)
            single = ~tg.forced & (tg.target >= 0)
            correct = model.policy_logits(out).argmax(-1) == tg.target
            value = model.value(out).float()
            _zones, draw = model.belief_v2(out, batch)
            draw_ll = F.binary_cross_entropy_with_logits(draw.float(), torch.ones_like(draw, dtype=torch.float32),
                                                         reduction="none")
            for k in CONDITIONS:
                m = c[k]
                if not bool(m.any()):
                    continue
                a = acc[k]
                a.n += int(m.sum())
                a.hit += int((correct & single & m).sum())
                a.n_single += int((single & m).sum())
                a.vp.append(value[m].cpu().numpy())
                a.vy.append(((tg.z[m] + 1) / 2).cpu().numpy())
                rows = c["_next_draw"] & m.unsqueeze(1)
                a.draw_ll += float(draw_ll[rows].sum())
                a.draw_n += int(rows.sum())
        if k_batch and k_batch % 100 == 0:
            print(f"  batch {k_batch} ({time.perf_counter() - t0:.0f}s)", flush=True)
    model.train()
    return {k: a.report() for k, a in acc.items()}


def main():
    from ml.checkpoints import load_model_v2
    from ml.dataset_v2 import PackedCorpusV2
    from sim import data_root
    from tools.eval_v2 import _checkpoint

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--data", default="bc/d973255d6dd1b036/packed_v2", help="the packed v2 corpus (split 1 is scored)")
    parser.add_argument("--positions", type=int, default=120000, help="validation positions scored (a fixed subsample)")
    parser.add_argument("--max-tokens", type=int, default=600000, help="the memory cap (bc.max_tokens)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    device = torch.device("cuda")
    form_of = {"joint": "rows", "stream": "rows", "turn_tokens": "folded", "summary": "folded"}
    report = {"positions": args.positions, "runs": {}}
    out = args.out or data_root.resolve(os.path.join("evals", "o10", "probe_targeted.json"))
    for run in args.runs.split(","):
        net, _meta = load_model_v2(_checkpoint(run))
        net.to(device)
        corpus = PackedCorpusV2(args.data, history=form_of.get(net.history_arch, "none"))
        sel = corpus.indices(split=1).subsample(args.positions)  # the same chunks for every run
        print(f"{run} ({net.history_arch})", flush=True)
        report["runs"][run] = score(net, corpus, sel, device, args.max_tokens)
        print(run, json.dumps(report["runs"][run]), flush=True)
        del net, corpus
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(f"written to {out}")


if __name__ == "__main__":
    main()

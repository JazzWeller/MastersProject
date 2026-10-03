"""The cheap screens (Agent Training Plan, Milestone M3): behaviour cloning
and value regression on the `sim.bc_corpus` corpus, every head trained on
one shared trunk.

- Screen 1 -- policy imitation: cross-entropy against HeuristicBot's
  choice; top-1 accuracy reported **per DecisionKind**.
- Screen 2 -- value regression: the eventual outcome; held-out log-loss and
  accuracy as a function of turn number, against a constant predictor.
- Screen 3 -- multi-select: enumerate / sequential / top-k as parallel heads
  on the same targets; exact-set match on CHOOSE_CARDS and exact
  permutation on ORDER_EFFECTS.
- Screen 4 -- ablations: re-run with config overrides (`tools/run_screens.py`).
- M6 auxiliaries ride along: the belief head (against the uniform-over-
  consistent-worlds baseline) and the oracle value head (against the
  student), both trained from the privileged labels.

`train(cfg, corpus, ...)` returns the model and a report dict; `main()`
wires it to a `Run` (config hash, journal, metrics, content-hashed
checkpoint).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from agent import spec
from agent.multiselect import decode_topk, enumerate_candidates, sequential_steps
from keyforge.enums import DecisionKind
from keyforge.infoset import PL_HAND, ZONE

from .checkpoints import CheckpointStore, save_model
from .dataset import KINDS, MULTI_KINDS, Corpus, Targets
from .encode import Batch
from .model import KeyForgeNet, TrunkOut, amp_dtype, candidates_tensor, param_count

K_CARDS = KINDS.index(DecisionKind.CHOOSE_CARDS)
K_ORDER = KINDS.index(DecisionKind.ORDER_EFFECTS)
OPP_UNSEEN = ZONE["opp_unseen"]
_THEM_HAND = spec.GLOBAL.offset("them") + PL_HAND
DEFAULT_WEIGHTS = {"policy": 1.0, "value": 1.0, "enumerate": 1.0, "sequential": 1.0, "topk": 1.0, "belief": 0.25, "oracle": 0.25}


def hidden_counts(batch: Batch, tg: Targets, vocab: int) -> torch.Tensor:
    """[B, 2*vocab]: the opponent's actual hand, then my actual next draws,
    as vocabulary count vectors -- the oracle head's privileged input."""
    B = batch.card_ids.shape[0]
    out = torch.zeros(B, 2 * vocab, device=batch.card_ids.device)
    out.scatter_add_(1, batch.card_ids, tg.opp_hand)
    draws = tg.next_draws
    valid = (draws >= 0).float()
    ids = torch.gather(batch.card_ids, 1, draws.clamp(min=0)) + vocab
    out.scatter_add_(1, ids, valid)
    return out


class MultiPrep:
    """CPU-side preparation of one batch's multi-select rows for all three
    treatments."""

    def __init__(self, tg: Targets, cap: int, device):
        kinds = tg.kind.tolist()
        forced = tg.forced.tolist()
        n_opt = tg.n_opt.tolist()
        mins, maxs = tg.min_n.tolist(), tg.max_n.tolist()
        self.rows: List[int] = []
        self.skipped = 0
        # enumerate
        c_rows, c_members, c_ordered, spans = [], [], [], []
        # sequential
        s_rows, s_prefix, s_legal, s_target = [], [], [], []
        self.keys: Dict[int, tuple] = {}
        for b, k in enumerate(kinds):
            if k not in MULTI_KINDS or forced[b]:
                continue
            ordered = k == K_ORDER
            n = n_opt[b]
            chosen = tg.chosen[b]
            key = tuple(chosen) if ordered else tuple(sorted(chosen))
            self.rows.append(b)
            self.keys[b] = key
            try:
                cands = enumerate_candidates(ordered, n, mins[b], maxs[b], cap)
            except ValueError:
                self.skipped += 1
                cands = None
            if cands is not None and key in cands:
                spans.append((b, len(c_members), len(cands), cands.index(key), cands))
                c_rows.extend([b] * len(cands))
                c_members.extend(cands)
                c_ordered.extend([ordered] * len(cands))
            for prefix, legal, stop_ok, target in sequential_steps(ordered, n, mins[b], maxs[b], chosen):
                s_rows.append(b)
                s_prefix.append(prefix)
                s_legal.append(legal + ([-1] if stop_ok else []))
                s_target.append(target)
        self.device = device
        self.spans = spans
        self.c_rows = torch.tensor(c_rows, dtype=torch.int64, device=device)
        self.c_members = candidates_tensor(c_members, device) if c_members else None
        self.c_ordered = torch.tensor(c_ordered, dtype=torch.bool, device=device)
        self.s_rows = s_rows
        self.s_prefix = s_prefix
        self.s_legal = s_legal
        self.s_target = s_target

    # ------------------------------------------------------------- losses
    # Every index structure below is built on the CPU and sent in one copy:
    # writing into a device tensor element by element launched a kernel per
    # element and left a graph node per element for backward to replay.
    def enumerate_loss(self, model: KeyForgeNet, out: TrunkOut):
        if not self.spans:
            return None, None
        scores = model.subset_scores(out, self.c_rows, self.c_members, self.c_ordered)
        counts = np.array([n for _b, _s, n, _t, _c in self.spans], dtype=np.int64)
        starts = np.array([s for _b, s, _n, _t, _c in self.spans], dtype=np.int64)
        row_of = np.repeat(np.arange(len(self.spans)), counts)
        col_of = np.arange(int(counts.sum())) - np.repeat(starts, counts)
        padded = torch.full((len(self.spans), int(counts.max())), float("-inf"), device=scores.device)
        padded[torch.from_numpy(row_of).to(scores.device), torch.from_numpy(col_of).to(scores.device)] = scores.float()
        targets = torch.tensor([t for _b, _s, _n, t, _c in self.spans], dtype=torch.int64, device=scores.device)
        return F.cross_entropy(padded, targets), padded

    def sequential_tensors(self, K: int):
        """(rows [S], prefix [S, K], legal [S, K+1]) for every step; the
        last column of `legal` is "stop"."""
        S = len(self.s_rows)
        prefix = np.zeros((S, K), dtype=np.float32)
        legal = np.zeros((S, K + 1), dtype=bool)
        for i in range(S):
            prefix[i, self.s_prefix[i]] = 1.0
            legal[i, [K if l == -1 else l for l in self.s_legal[i]]] = True
        return (
            torch.tensor(self.s_rows, dtype=torch.int64, device=self.device),
            torch.from_numpy(prefix).to(self.device),
            torch.from_numpy(legal).to(self.device),
        )

    def sequential_loss(self, model: KeyForgeNet, out: TrunkOut, n_opt: List[int]):
        if not self.s_rows:
            return None
        K = out.e_opt.shape[1]
        rows, prefix, legal = self.sequential_tensors(K)
        target = torch.tensor(
            [K if t == n_opt[b] else t for b, t in zip(self.s_rows, self.s_target)], dtype=torch.int64, device=self.device,
        )
        logits = model.sequential_logits(out, rows, prefix, legal)
        return F.cross_entropy(logits, target)

    def topk_loss(self, model: KeyForgeNet, out: TrunkOut, n_opt: List[int], kinds: List[int]):
        if not self.rows:
            return None
        logits = model.topk_logits(out)
        total, count = 0.0, sum(n_opt[b] for b in self.rows)
        # CHOOSE_CARDS rows, all at once: an independent binary target per
        # offered option, summed over the options each row offers.
        cards = [b for b in self.rows if kinds[b] != K_ORDER]
        if cards:
            K = logits.shape[1]
            mask = np.zeros((len(cards), K), dtype=bool)
            y = np.zeros((len(cards), K), dtype=np.float32)
            for r, b in enumerate(cards):
                mask[r, : n_opt[b]] = True
                y[r, list(self.keys[b])] = 1.0
            mask_t = torch.from_numpy(mask).to(logits.device)
            s = logits[torch.tensor(cards, device=logits.device)]
            # Offered options only: the rest are -inf, and -inf x 0 is NaN.
            s = torch.where(mask_t, s, torch.zeros_like(s))
            per = F.binary_cross_entropy_with_logits(s, torch.from_numpy(y).to(logits.device), reduction="none")
            total = total + (per * mask_t).sum()
        # ORDER_EFFECTS rows (0.2% of decisions): Plackett-Luce over the
        # target permutation, with each option scored independently.
        for b in self.rows:
            if kinds[b] != K_ORDER:
                continue
            key = self.keys[b]
            ss = logits[b, : n_opt[b]][torch.tensor(list(key), dtype=torch.int64, device=logits.device)]
            total = total + sum(torch.logsumexp(ss[t:], 0) - ss[t] for t in range(len(key)))
        return total / max(count, 1)


def compute_losses(model: KeyForgeNet, batch: Batch, tg: Targets, *, weights: dict, policy_head: str, cap: int, heads: Sequence[str]):
    out = model.encode_state(batch)
    losses: Dict[str, torch.Tensor] = {}
    single = ~tg.forced & (tg.target >= 0)
    if "policy" in heads and bool(single.any()):
        logits = model.fixed_logits(out, batch) if policy_head == "fixed" else model.policy_logits(out)
        losses["policy"] = F.cross_entropy(logits[single], tg.target[single])
    if "value" in heads:
        losses["value"] = F.mse_loss(model.value(out), tg.z)
    prep = MultiPrep(tg, cap, batch.card_ids.device)
    n_opt = tg.n_opt.tolist()
    if "enumerate" in heads:
        l, _ = prep.enumerate_loss(model, out)
        if l is not None:
            losses["enumerate"] = l
    if "sequential" in heads:
        l = prep.sequential_loss(model, out, n_opt)
        if l is not None:
            losses["sequential"] = l
    if "topk" in heads:
        l = prep.topk_loss(model, out, n_opt, tg.kind.tolist())
        if l is not None:
            losses["topk"] = l
    if "belief" in heads:
        unseen = batch.zones == OPP_UNSEEN
        if bool(unseen.any()):
            losses["belief"] = F.binary_cross_entropy_with_logits(model.belief_logits(out)[unseen], tg.opp_hand[unseen])
    if "oracle" in heads:
        losses["oracle"] = F.mse_loss(model.oracle_value(out, hidden_counts(batch, tg, model.vocab_size)), tg.z)
    total = sum(weights.get(k, 1.0) * v for k, v in losses.items())
    return total, losses, out, prep


class Prefetcher:
    """Assembles the next batches on a background thread while the GPU works
    (numpy releases the GIL for most of the copying)."""

    def __init__(self, it, depth: int = 3):
        self.q: "queue.Queue" = queue.Queue(maxsize=depth)
        self.t = threading.Thread(target=self._run, args=(it,), daemon=True)
        self.t.start()

    def _run(self, it):
        try:
            for item in it:
                self.q.put(item)
        finally:
            self.q.put(None)

    def __iter__(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            yield item


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list:
    """The M6 reliability diagram as data: per predicted-probability bin,
    the mean prediction and the observed frequency (a calibrated head has
    them equal)."""
    out = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1.0 else (p <= hi))
        if m.sum() >= 20:
            out.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()), "predicted": round(float(p[m].mean()), 4),
                        "observed": round(float(y[m].mean()), 4)})
    return out


def _logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))) if len(p) else float("nan")


@torch.no_grad()
def evaluate(model: KeyForgeNet, corpus: Corpus, idx: np.ndarray, *, policy_head: str, cap: int, batch_size: int = 512, device=None) -> dict:
    model.eval()
    single_hits: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    multi_hits = {t: {"CHOOSE_CARDS": [0, 0], "ORDER_EFFECTS": [0, 0]} for t in ("enumerate", "sequential", "topk")}
    v_pred, v_z, v_turn, v_src = [], [], [], []
    b_p, b_y, b_u = [], [], []
    o_pred = []
    for batch, tg in corpus.iterate(idx, batch_size, shuffle=False, device=device):
        out = model.encode_state(batch)
        kinds = tg.kind.tolist()
        n_opt = tg.n_opt.tolist()
        logits = model.fixed_logits(out, batch) if policy_head == "fixed" else model.policy_logits(out)
        pred = logits.argmax(-1).tolist()
        targets = tg.target.tolist()
        forced = tg.forced.tolist()
        for b, k in enumerate(kinds):
            if forced[b] or targets[b] < 0:
                continue
            h = single_hits[KINDS[k].name]
            h[0] += pred[b] == targets[b]
            h[1] += 1
        prep = MultiPrep(tg, cap, batch.card_ids.device)
        if prep.rows:
            # enumerate: argmax over candidates
            _l, padded = prep.enumerate_loss(model, out)
            if padded is not None:
                best = padded.argmax(-1).tolist()
                for r, (b, _s, _n, t, cands) in enumerate(prep.spans):
                    h = multi_hits["enumerate"][KINDS[kinds[b]].name]
                    h[0] += best[r] == t
                    h[1] += 1
            # top-k: independent scores
            tk = model.topk_logits(out)
            for b in prep.rows:
                n = n_opt[b]
                dec = decode_topk(tk[b, :n].tolist(), kinds[b] == K_ORDER, int(tg.min_n[b]), int(tg.max_n[b]))
                h = multi_hits["topk"][KINDS[kinds[b]].name]
                h[0] += tuple(dec) == prep.keys[b]
                h[1] += 1
            # sequential: greedy chain
            for b in prep.rows:
                ordered = kinds[b] == K_ORDER
                picked = _greedy_sequential(model, out, b, n_opt[b], int(tg.min_n[b]), int(tg.max_n[b]), ordered)
                key = tuple(picked) if ordered else tuple(sorted(picked))
                h = multi_hits["sequential"][KINDS[kinds[b]].name]
                h[0] += key == prep.keys[b]
                h[1] += 1
        v_pred.append(model.value(out).float().cpu().numpy())
        v_z.append(tg.z.cpu().numpy())
        v_turn.append(tg.turn.cpu().numpy())
        v_src.append(tg.source.cpu().numpy())
        unseen = batch.zones == OPP_UNSEEN
        if bool(unseen.any()):
            p = torch.sigmoid(model.belief_logits(out))[unseen].float().cpu().numpy()
            y = tg.opp_hand[unseen].float().cpu().numpy()
            hand = torch.round(batch.globals[:, _THEM_HAND] * spec.SCALE["hand"]).clamp(min=0)
            n_unseen = unseen.sum(-1).clamp(min=1).float()
            u = (hand / n_unseen).clamp(0, 1).unsqueeze(-1).expand_as(unseen)[unseen].float().cpu().numpy()
            b_p.append(p)
            b_y.append(y)
            b_u.append(u)
        o_pred.append(model.oracle_value(out, hidden_counts(batch, tg, model.vocab_size)).float().cpu().numpy())
    model.train()

    v_pred = np.concatenate(v_pred) if v_pred else np.zeros(0)
    v_z = np.concatenate(v_z) if v_z else np.zeros(0)
    v_turn = np.concatenate(v_turn) if v_turn else np.zeros(0)
    o_pred = np.concatenate(o_pred) if o_pred else np.zeros(0)
    y = (v_z + 1) / 2
    base_p = np.full_like(y, 0.5)
    by_turn = []
    for lo in range(0, int(v_turn.max()) + 1 if len(v_turn) else 0, 2):
        m = (v_turn >= lo) & (v_turn < lo + 2)
        if m.sum() < 20:
            continue
        decided = m & (v_z != 0)
        by_turn.append({
            "turns": f"{lo}-{lo + 1}",
            "n": int(m.sum()),
            "logloss": round(_logloss((v_pred[m] + 1) / 2, y[m]), 4),
            "constant_logloss": round(_logloss(base_p[m], y[m]), 4),
            "accuracy": round(float(np.mean(np.sign(v_pred[decided]) == v_z[decided])), 4) if decided.any() else None,
        })
    belief = None
    if b_p:
        p, yy, u = np.concatenate(b_p), np.concatenate(b_y), np.concatenate(b_u)
        belief = {"n": int(len(p)), "logloss": round(_logloss(p, yy), 4), "uniform_logloss": round(_logloss(u, yy), 4),
                  "reliability": reliability(p, yy)}
    return {
        "top1_by_kind": {k: {"accuracy": round(h[0] / h[1], 4), "n": h[1]} for k, h in sorted(single_hits.items()) if h[1]},
        "multi_select_exact": {
            t: {k: {"accuracy": round(h[0] / h[1], 4) if h[1] else None, "n": h[1]} for k, h in d.items()}
            for t, d in multi_hits.items()
        },
        "value": {
            "n": int(len(v_pred)),
            "logloss": round(_logloss((v_pred + 1) / 2, y), 4),
            "constant_logloss": round(_logloss(base_p, y), 4),
            "mse": round(float(np.mean((v_pred - v_z) ** 2)), 4) if len(v_pred) else None,
            "by_turn": by_turn,
        },
        "oracle": {
            "logloss": round(_logloss((o_pred + 1) / 2, y), 4),
            "mse": round(float(np.mean((o_pred - v_z) ** 2)), 4) if len(o_pred) else None,
        },
        "belief": belief,
    }


def _greedy_sequential(model: KeyForgeNet, out: TrunkOut, b: int, n: int, min_n: int, max_n: int, ordered: bool) -> List[int]:
    from agent.multiselect import next_sequential_legal

    K = out.e_opt.shape[1]
    prefix: List[int] = []
    rows = torch.tensor([b], device=out.g.device)
    while True:
        legal, stop_ok = next_sequential_legal(ordered, n, min_n, max_n, prefix)
        if not legal and not stop_ok:
            return prefix
        pm = torch.zeros(1, K, device=out.g.device)
        for p in prefix:
            pm[0, p] = 1.0
        lm = torch.zeros(1, K + 1, dtype=torch.bool, device=out.g.device)
        for l in legal:
            lm[0, l] = True
        if stop_ok:
            lm[0, K] = True
        j = int(model.sequential_logits(out, rows, pm, lm).argmax(-1))
        if j == K:
            return prefix
        prefix.append(j)


def train(
    cfg: dict, corpus: Corpus, *, device, heads: Sequence[str] = tuple(DEFAULT_WEIGHTS), metrics=None, journal=None,
    log_every: int = 200, eval_sources: Optional[Sequence[str]] = None, train_sources: Optional[Sequence[str]] = None,
) -> Tuple[KeyForgeNet, dict]:
    net_cfg = cfg["network"]
    bc = cfg["bc"]
    torch.manual_seed(int(cfg.get("seed", 0)))
    model = KeyForgeNet(net_cfg).to(device)
    policy_head = net_cfg.get("policy_head", "pointer")
    cap = int(net_cfg.get("enumerate_cap", 1024))
    weights = dict(DEFAULT_WEIGHTS, **(bc.get("loss_weights") or {}))
    train_idx = corpus.indices(split=0, sources=train_sources)
    val_idx = corpus.indices(split=1, sources=eval_sources)
    steps_per_epoch = math.ceil(len(train_idx) / bc["batch"])
    total_steps = max(1, steps_per_epoch * int(bc["epochs"]))
    opt = torch.optim.AdamW(model.parameters(), lr=float(bc["lr"]), weight_decay=float(bc["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=float(bc["lr_min"]))
    amp = amp_dtype(bc.get("precision", "fp32"), device)
    step = 0
    t0 = time.perf_counter()
    history = []
    for epoch in range(int(bc["epochs"])):
        running: Dict[str, float] = defaultdict(float)
        n_run = 0
        it = Prefetcher(corpus.iterate(train_idx, int(bc["batch"]), shuffle=True, seed=epoch, device=None))
        for batch, tg in it:
            batch = batch.to(device)
            for f in ("kind", "target", "forced", "z", "turn", "source", "min_n", "max_n", "n_opt", "opp_hand", "next_draws"):
                setattr(tg, f, getattr(tg, f).to(device, non_blocking=True))
            with torch.autocast(torch.device(device).type, dtype=amp or torch.float32, enabled=amp is not None):
                total, losses, _out, _prep = compute_losses(model, batch, tg, weights=weights, policy_head=policy_head, cap=cap, heads=heads)
            opt.zero_grad(set_to_none=True)
            total.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(bc["grad_clip"]))
            opt.step()
            sched.step()
            step += 1
            for k, v in losses.items():
                running[k] += float(v.detach())
            n_run += 1
            if metrics is not None:
                metrics.count("gradient_steps")
                metrics.count("positions", batch.size)
                metrics.observe("grad_norm", float(gnorm))
                for k, v in losses.items():
                    metrics.observe(f"loss_{k}", float(v.detach()))
                metrics.gauge("lr", sched.get_last_lr()[0])
                metrics.tick()
            if step % log_every == 0:
                msg = " ".join(f"{k}={running[k] / n_run:.4f}" for k in sorted(running))
                print(f"epoch {epoch} step {step}/{total_steps} {msg} ({time.perf_counter() - t0:.0f}s)", flush=True)
                history.append({"step": step, **{k: running[k] / n_run for k in running}})
                running.clear()
                n_run = 0
    elapsed = time.perf_counter() - t0
    report = evaluate(model, corpus, val_idx, policy_head=policy_head, cap=cap, device=device)
    report["training"] = {
        "steps": step, "epochs": int(bc["epochs"]), "train_positions": int(len(train_idx)),
        "val_positions": int(len(val_idx)), "seconds": round(elapsed, 1), "params": param_count(model),
        "history": history,
    }
    return model, report


def main():
    from agent.telemetry import Run

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="tier0_bc.json")
    parser.add_argument("--run", required=True, help="run id (under $KEYFORGE_DATA/runs)")
    parser.add_argument("--data", required=True, help="encoded shard directory (ml.dataset)")
    parser.add_argument("--override", default=None, help="JSON dict merged over the config (Screen 4 ablations)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    overrides = json.loads(args.override) if args.override else None
    run = Run.resume_or_create(args.run, args.config, overrides)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    corpus = Corpus.from_dir(args.data)
    run.journal("bc_train_start", game_index=0, data=args.data, positions=corpus.size, device=str(device))
    model, report = train(run.config, corpus, device=device, metrics=run.metrics("learner"))
    store = CheckpointStore(os.path.join(run.root, "checkpoints"))
    ckpt, digest = save_model(model, store, config_hash=run.config_hash, extra={"kind": "bc"})
    report["checkpoint"] = ckpt
    report["weights_digest"] = digest
    report["config_hash"] = run.config_hash
    path = run.write_artifact("reports", json.dumps(report, indent=2, sort_keys=True).encode("utf-8"), ".json")
    with open(os.path.join(run.root, "bc_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    run.journal("bc_train_done", game_index=0, checkpoint=ckpt, report=os.path.basename(path))
    print(json.dumps({k: report[k] for k in ("top1_by_kind", "multi_select_exact", "belief")}, indent=1))
    print(f"checkpoint {ckpt[:16]} (weights {digest[:16]})")


if __name__ == "__main__":
    main()

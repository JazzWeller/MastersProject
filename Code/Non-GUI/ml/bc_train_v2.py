"""Behaviour cloning with network v2: Tier 0b (Agent Observation Plan, O9).

v1's screens on v2 inputs (`ml/dataset_v2.py`, `ml/model_v2.py`): policy
imitation, value regression, the three multi-select treatments, plus

- **belief v2**: for each of the opponent's cards whose place the viewer
  doesn't know (every card with an O3 prior), cross-entropy over hand /
  archive / deck against where it really is, and whether it is their next
  draw (binary). The head adds to O3's exact prior, so its log-loss is
  reported against the prior's own;
- **the oracle** over the opponent's hand, their archive and my next draws.

`training.history_dropout` / `bc.history_dropout` (default 0) drops event
rows at random in training. `train(cfg, corpus, ...)` returns the model and
a report: v1's (top-1 per kind, multi-select exact, value log-loss by turn
and by boundary / mid-turn) plus belief v2 against the prior.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from agent import spec_v2 as S
from agent.multiselect import decode_topk
from keyforge.enums import BOUNDARY_KINDS

from .bc_train import K_ORDER, MultiPrep, Prefetcher, _greedy_sequential, _logloss, reliability
from .checkpoints import CheckpointStore, save_model
from .dataset import KINDS, MULTI_KINDS
from .dataset_v2 import CorpusV2, PackedCorpusV2, TargetsV2
from .encode_v2 import BatchV2
from .layers import set_activation_checkpointing
from .model import amp_dtype
from .model_v2 import VOCAB, KeyForgeNetV2, TrunkOutV2, checkpoint_stamp, param_count

DEFAULT_WEIGHTS = {"policy": 1.0, "value": 1.0, "enumerate": 1.0, "sequential": 1.0, "topk": 1.0, "belief": 0.25,
                   "draw": 0.25, "oracle": 0.25}
_F = S.ENTITY.f
_PRIOR = [_F["p_hand"], _F["p_archive"], _F["p_deck"], _F["p_elsewhere"]]
_OWNER = S.ENTITY.i["owner"]
_CARD = S.ENTITY.i["card"]


def _hidden_rows(batch: BatchV2) -> torch.Tensor:
    """[B, 72] bool: the opponent's cards whose place the viewer doesn't
    know -- the ones O3 gives a prior."""
    return batch.blocks["entity"].floats[..., _PRIOR].sum(-1) > 0


def oracle_counts(batch: BatchV2, tg: TargetsV2) -> torch.Tensor:
    """[B, 3 * vocab]: the opponent's actual hand, their actual archive, my
    actual next draws, as card-count vectors (the oracle's privileged input)."""
    ent = batch.blocks["entity"].ints
    ids = ent[..., _CARD].clamp(min=0)
    theirs = (ent[..., _OWNER] == 1).float()
    B = ids.shape[0]
    out = torch.zeros(B, 3 * VOCAB, device=ids.device)
    out.scatter_add_(1, ids, theirs * (tg.zone == 1).float())
    out.scatter_add_(1, ids + VOCAB, theirs * (tg.zone == 2).float())
    draws = tg.my_draws
    valid = (draws >= 0).float()
    out.scatter_add_(1, torch.gather(ids, 1, draws.clamp(min=0)) + 2 * VOCAB, valid)
    return out


def _belief_targets(batch: BatchV2, tg: TargetsV2):
    """(rows for the zone loss [B, 72] bool, class 0..2, rows for the draw
    loss, is-next-draw float)."""
    hidden = _hidden_rows(batch)
    zone_rows = hidden & (tg.zone >= 1)
    top = tg.opp_deck[:, :1]
    has_deck = (top >= 0).squeeze(1)
    is_next = torch.zeros_like(hidden, dtype=torch.float32)
    is_next.scatter_(1, top.clamp(min=0), has_deck.float().unsqueeze(1))
    draw_rows = hidden & has_deck.unsqueeze(1)
    return zone_rows, (tg.zone - 1).clamp(min=0), draw_rows, is_next


def compute_losses(model: KeyForgeNetV2, batch: BatchV2, tg: TargetsV2, *, weights: dict, cap: int,
                   heads: Sequence[str], prep: Optional[MultiPrep] = None):
    out = model.encode_state(batch)
    losses: Dict[str, torch.Tensor] = {}
    single = ~tg.forced & (tg.target >= 0)
    if "policy" in heads and bool(single.any()):
        losses["policy"] = F.cross_entropy(model.policy_logits(out)[single].float(), tg.target[single])
    if "value" in heads:
        losses["value"] = F.mse_loss(model.value(out).float(), tg.z)
    if prep is None:
        prep = MultiPrep(tg, cap, tg.kind.device)
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
    if "belief" in heads or "draw" in heads:
        zones, draw = model.belief_v2(out, batch)
        zone_rows, cls, draw_rows, is_next = _belief_targets(batch, tg)
        if "belief" in heads and bool(zone_rows.any()):
            losses["belief"] = F.cross_entropy(zones[zone_rows].float(), cls[zone_rows])
        if "draw" in heads and bool(draw_rows.any()):
            losses["draw"] = F.binary_cross_entropy_with_logits(draw[draw_rows].float(), is_next[draw_rows])
    if "oracle" in heads:
        losses["oracle"] = F.mse_loss(model.oracle_value(out, oracle_counts(batch, tg)).float(), tg.z)
    total = sum(weights.get(k, 1.0) * v for k, v in losses.items())
    return total, losses, out, prep


@torch.no_grad()
def evaluate(model: KeyForgeNetV2, corpus: CorpusV2, idx: np.ndarray, *, cap: int, batch_size: int = 256,
             device=None) -> dict:
    model.eval()
    single_hits: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    multi_hits = {t: {"CHOOSE_CARDS": [0, 0], "ORDER_EFFECTS": [0, 0]} for t in ("enumerate", "sequential", "topk")}
    v_pred, v_z, v_turn, v_kind = [], [], [], []
    bz_model, bz_prior, n_zone = 0.0, 0.0, 0
    hand_p, hand_prior, hand_y = [], [], []
    d_model, d_prior, n_draw = 0.0, 0.0, 0
    o_pred = []
    for batch, tg in Prefetcher(corpus.iterate(idx, batch_size, shuffle=False)):
        batch, tg = batch.to(device), tg.to(device)
        out = model.encode_state(batch)
        kinds = tg.kind.tolist()
        n_opt = tg.n_opt.tolist()
        pred = model.policy_logits(out).argmax(-1).tolist()
        targets = tg.target.tolist()
        forced = tg.forced.tolist()
        for b, k in enumerate(kinds):
            if forced[b] or targets[b] < 0:
                continue
            h = single_hits[KINDS[k].name]
            h[0] += pred[b] == targets[b]
            h[1] += 1
        prep = MultiPrep(tg, cap, tg.kind.device)
        if prep.rows:
            _l, padded = prep.enumerate_loss(model, out)
            if padded is not None:
                best = padded.argmax(-1).tolist()
                for r, (b, _s, _n, t, _c) in enumerate(prep.spans):
                    h = multi_hits["enumerate"][KINDS[kinds[b]].name]
                    h[0] += best[r] == t
                    h[1] += 1
            tk = model.topk_logits(out)
            for b in prep.rows:
                dec = decode_topk(tk[b, :n_opt[b]].tolist(), kinds[b] == K_ORDER, int(tg.min_n[b]), int(tg.max_n[b]))
                h = multi_hits["topk"][KINDS[kinds[b]].name]
                h[0] += tuple(dec) == prep.keys[b]
                h[1] += 1
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
        v_kind.append(np.asarray(kinds))
        zones, draw = model.belief_v2(out, batch)
        zone_rows, cls, draw_rows, is_next = _belief_targets(batch, tg)
        if bool(zone_rows.any()):
            logp = torch.log_softmax(zones.float(), -1)[zone_rows]
            prior = batch.blocks["entity"].floats[..., _PRIOR[:3]].float()[zone_rows].clamp(min=1e-6)
            prior = prior / prior.sum(-1, keepdim=True)
            c = cls[zone_rows]
            bz_model += float(-logp.gather(1, c.unsqueeze(1)).sum())
            bz_prior += float(-torch.log(prior.gather(1, c.unsqueeze(1))).sum())
            n_zone += int(c.numel())
        hidden = _hidden_rows(batch)
        if bool(hidden.any()):
            # P(in hand) over every hidden card, O3's calibration metric
            hand_p.append(torch.softmax(zones.float(), -1)[..., 0][hidden].cpu().numpy())
            hand_prior.append(batch.blocks["entity"].floats[..., _F["p_hand"]].float()[hidden].cpu().numpy())
            hand_y.append((tg.zone == 1)[hidden].float().cpu().numpy())
        if bool(draw_rows.any()):
            y = is_next[draw_rows]
            d_model += float(F.binary_cross_entropy_with_logits(draw[draw_rows].float(), y, reduction="sum"))
            pn = batch.blocks["entity"].floats[..., _F["p_next_draw"]].float()[draw_rows].clamp(1e-6, 1 - 1e-6)
            d_prior += float(F.binary_cross_entropy(pn, y, reduction="sum"))
            n_draw += int(y.numel())
        o_pred.append(model.oracle_value(out, oracle_counts(batch, tg)).float().cpu().numpy())
    model.train()
    v_pred, v_z = np.concatenate(v_pred), np.concatenate(v_z)
    v_turn, v_kind = np.concatenate(v_turn), np.concatenate(v_kind)
    o_pred = np.concatenate(o_pred)
    y = (v_z + 1) / 2
    base = np.full_like(y, 0.5)

    def value_by_turn(mask):
        rows = []
        for lo in range(0, int(v_turn.max()) + 1 if len(v_turn) else 0, 2):
            m = mask & (v_turn >= lo) & (v_turn < lo + 2)
            if m.sum() < 20:
                continue
            decided = m & (v_z != 0)
            rows.append({"turns": f"{lo}-{lo + 1}", "n": int(m.sum()), "logloss": round(_logloss((v_pred[m] + 1) / 2, y[m]), 4),
                         "constant_logloss": round(_logloss(base[m], y[m]), 4),
                         "accuracy": round(float(np.mean(np.sign(v_pred[decided]) == v_z[decided])), 4) if decided.any() else None})
        return rows

    at_boundary = np.isin(v_kind, np.asarray([KINDS.index(k) for k in BOUNDARY_KINDS]))
    by_position = {name: {"n": int(m.sum()), "logloss": round(_logloss((v_pred[m] + 1) / 2, y[m]), 4),
                          "constant_logloss": round(_logloss(base[m], y[m]), 4)}
                   for name, m in (("boundary", at_boundary), ("mid_turn", ~at_boundary))}
    belief = None
    if n_zone:
        hp, hpr, hy = np.concatenate(hand_p), np.concatenate(hand_prior), np.concatenate(hand_y)
        belief = {
            "zone_n": n_zone, "zone_logloss": round(bz_model / n_zone, 4), "zone_prior_logloss": round(bz_prior / n_zone, 4),
            "hand_n": int(len(hy)), "hand_logloss": round(_logloss(hp, hy), 4),
            "hand_prior_logloss": round(_logloss(hpr, hy), 4), "reliability": reliability(hp, hy),
            "next_draw_n": n_draw, "next_draw_logloss": round(d_model / max(1, n_draw), 4),
            "next_draw_prior_logloss": round(d_prior / max(1, n_draw), 4),
        }
    return {
        "top1_by_kind": {k: {"accuracy": round(h[0] / h[1], 4), "n": h[1]} for k, h in sorted(single_hits.items()) if h[1]},
        "multi_select_exact": {t: {k: {"accuracy": round(h[0] / h[1], 4) if h[1] else None, "n": h[1]} for k, h in d.items()}
                               for t, d in multi_hits.items()},
        "value": {"n": int(len(v_pred)), "logloss": round(_logloss((v_pred + 1) / 2, y), 4),
                  "constant_logloss": round(_logloss(base, y), 4), "by_turn": value_by_turn(np.ones(len(v_pred), bool)),
                  "by_position": by_position},
        "oracle": {"logloss": round(_logloss((o_pred + 1) / 2, y), 4)},
        "belief": belief,
    }


def train(cfg: dict, corpus: CorpusV2, *, device, heads: Sequence[str] = tuple(DEFAULT_WEIGHTS), metrics=None,
          log_every: int = 200, max_steps: Optional[int] = None, train_idx: Optional[np.ndarray] = None,
          val_idx: Optional[np.ndarray] = None, state_path: Optional[str] = None,
          save_minutes: float = 10.0) -> Tuple[KeyForgeNetV2, dict]:
    """`train_idx` / `val_idx` default to the corpus's by-game split.

    `state_path`: makes the run pausable (`ml/resume.py`) -- its state is
    saved there every `save_minutes` and on a pause request (then `Paused`
    is raised), and a run started with a state file there resumes from it."""
    net_cfg = cfg["network"]
    bc = cfg["bc"]
    torch.manual_seed(int(cfg.get("seed", 0)))
    model = KeyForgeNetV2(net_cfg).to(device)
    cap = int(net_cfg.get("enumerate_cap", 1024))
    weights = dict(DEFAULT_WEIGHTS, **(bc.get("loss_weights") or {}))
    as_idx = lambda x: x if hasattr(x, "chunks") else np.asarray(x)  # (a packed corpus's Selection stays one)
    train_idx = corpus.indices(split=0) if train_idx is None else as_idx(train_idx)
    val_idx = corpus.indices(split=1) if val_idx is None else as_idx(val_idx)
    if bc.get("eval_positions"):  # a fixed subsample of the validation split, for quicker screens
        if hasattr(val_idx, "subsample"):
            val_idx = val_idx.subsample(int(bc["eval_positions"]))
        else:
            val_idx = np.random.default_rng(0).choice(val_idx, min(len(val_idx), int(bc["eval_positions"])), replace=False)
            val_idx.sort()
    steps_per_epoch = math.ceil(len(train_idx) / bc["batch"])
    total_steps = max(1, steps_per_epoch * int(bc["epochs"]))
    if max_steps:
        total_steps = min(total_steps, max_steps)
    opt = torch.optim.AdamW(model.parameters(), lr=float(bc["lr"]), weight_decay=float(bc["weight_decay"]),
                            fused=torch.device(device).type == "cuda")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=float(bc["lr_min"]))
    amp = amp_dtype(bc.get("precision", "bf16"), device)
    set_activation_checkpointing(model.trunk, bool(bc.get("activation_checkpointing", False)))
    if bc.get("compile", True) and torch.device(device).type == "cuda":
        model.compile_inputs()
    step, t0, history, seen = 0, time.perf_counter(), [], 0
    start_epoch, skip, before = 0, 0, 0.0  # (resume: the epoch, batches into it, seconds already trained)
    pauser = digest = None
    if state_path is not None:
        from . import resume

        digest = resume.config_digest(cfg, total_steps, len(train_idx))
        prog = resume.load_state(state_path, model=model, opt=opt, sched=sched, config_hash=digest)
        if prog is not None:
            step, history, seen = prog["step"], prog["history"], prog["seen"]
            start_epoch, skip, before = prog["epoch"], prog["in_epoch"], prog["seconds"]
            print(f"resuming at step {step}/{total_steps} (epoch {start_epoch}, batch {skip})", flush=True)
        pauser = resume.Pauser(state_path, save_minutes)
    done = step >= total_steps
    t0 -= before
    for epoch in range(start_epoch, int(bc["epochs"])):
        if done:
            break
        running: Dict[str, float] = defaultdict(float)
        n_run = 0
        in_epoch = skip if epoch == start_epoch else 0
        it = corpus.iterate(train_idx, int(bc["batch"]), shuffle=True, seed=epoch, train=True,
                            **({"skip": in_epoch} if in_epoch and isinstance(corpus, PackedCorpusV2) else {}))
        if in_epoch and not isinstance(corpus, PackedCorpusV2):
            import itertools

            it = itertools.islice(it, in_epoch, None)  # (a windowed corpus still reads what it skips)
        for batch, tg in Prefetcher(it):
            batch, tg = batch.to(device), tg.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(torch.device(device).type, dtype=amp or torch.float32, enabled=amp is not None):
                total, losses, _out, _prep = compute_losses(model, batch, tg, weights=weights, cap=cap, heads=heads)
            total.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(bc["grad_clip"]))
            opt.step()
            sched.step()
            step += 1
            in_epoch += 1
            seen += batch.size
            for k, v in losses.items():
                running[k] += float(v.detach())
            n_run += 1
            if metrics is not None:
                metrics.count("gradient_steps")
                metrics.count("positions", batch.size)
                metrics.observe("grad_norm", float(gnorm))
                for k, v in losses.items():
                    metrics.observe(f"loss_{k}", float(v.detach()))
                metrics.tick()
            if step % log_every == 0:
                msg = " ".join(f"{k}={running[k] / n_run:.4f}" for k in sorted(running))
                el = time.perf_counter() - t0
                print(f"epoch {epoch} step {step}/{total_steps} {msg} ({el:.0f}s, {seen / el:.0f} samples/s)", flush=True)
                history.append({"step": step, **{k: running[k] / n_run for k in running}})
                running.clear()
                n_run = 0
            if step >= total_steps:
                done = True
                break
            asked = pauser is not None and pauser.requested()
            if asked or (pauser is not None and pauser.save_due()):
                # (the partial log window since the last line is dropped)
                resume.save_state(state_path, model=model, opt=opt, sched=sched, config_hash=digest, step=step,
                                  epoch=epoch, in_epoch=in_epoch, history=history, seen=seen,
                                  seconds=time.perf_counter() - t0)
                pauser.saved()
                if asked:
                    pauser.close()
                    print(f"paused at step {step}/{total_steps}: state saved to {state_path}", flush=True)
                    raise resume.Paused(state_path)
    elapsed = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.device(device).type == "cuda" else None
    if pauser is not None:  # trained: a pause from here on resumes at the evaluation
        resume.save_state(state_path, model=model, opt=opt, sched=sched, config_hash=digest, step=step,
                          epoch=int(bc["epochs"]), in_epoch=0, history=history, seen=seen, seconds=elapsed)
        pauser.close()
    report = evaluate(model, corpus, val_idx, cap=cap, device=device)
    report["training"] = {"steps": step, "epochs": int(bc["epochs"]), "train_positions": int(len(train_idx)),
                          "val_positions": int(len(val_idx)), "seconds": round(elapsed, 1),
                          "samples_per_second": round(seen / max(elapsed, 1e-9)), "params": param_count(model),
                          "peak_gib": round(peak, 2) if peak is not None else None, "history": history,
                          "history_arch": model.history_arch}
    return model, report


def main():
    from agent.telemetry import Run

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="tier0b_bc.json")
    parser.add_argument("--run", required=True, help="run id (under $KEYFORGE_DATA/runs)")
    parser.add_argument("--data", required=True, help="v2 shard directory (ml.dataset_v2)")
    parser.add_argument("--override", default=None, help="JSON dict merged over the config")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-minutes", type=float, default=10.0,
                        help="how often the resumable state is saved (ml/resume.py: pause with SIGTERM, Ctrl-C or a "
                             "PAUSE file in the run directory; rerun the same command to resume)")
    args = parser.parse_args()
    overrides = json.loads(args.override) if args.override else None
    run = Run.resume_or_create(args.run, args.config, overrides)
    cfg = run.config
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    arch = cfg["network"].get("history_arch", "none")
    form = {"joint": "rows", "stream": "rows", "turn_tokens": "folded", "summary": "folded"}.get(arch, "none")
    dropout = float(cfg["bc"].get("history_dropout", 0.0))
    from sim import data_root

    packed = any(n.endswith(".idx.npz") for n in os.listdir(data_root.resolve(args.data)))
    # (a packed corpus spreads each game's positions across the epoch; see ml/dataset_v2.py)
    corpus = (PackedCorpusV2(args.data, history=form, history_dropout=dropout) if packed
              else CorpusV2.from_dir(args.data, history=form, history_dropout=dropout))
    run.journal("bc_train_start", game_index=0, data=args.data, positions=corpus.size, device=str(device))
    from . import resume

    state_path = os.path.join(run.root, "train_state.pt")
    model, report = train(cfg, corpus, device=device, metrics=run.metrics("learner"), max_steps=args.max_steps,
                          state_path=state_path, save_minutes=args.save_minutes)
    store = CheckpointStore(os.path.join(run.root, "checkpoints"))
    ckpt, digest = save_model(model, store, config_hash=run.config_hash, extra={"kind": "bc_v2", **checkpoint_stamp()})
    report.update(checkpoint=ckpt, weights_digest=digest, config_hash=run.config_hash)
    run.write_artifact("reports", json.dumps(report, indent=2, sort_keys=True).encode("utf-8"), ".json")
    with open(os.path.join(run.root, "bc_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    run.journal("bc_train_done", game_index=0, checkpoint=ckpt)
    resume.finished(state_path)
    print(json.dumps({k: report[k] for k in ("top1_by_kind", "belief", "training")}, indent=1, default=str))


if __name__ == "__main__":
    main()

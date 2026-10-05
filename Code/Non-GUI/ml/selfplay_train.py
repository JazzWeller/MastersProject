"""The self-play training loop (Agent Training Plan, Milestones M7 and M8).

    python -m ml.selfplay_train --run tier2-wt-s0 --config tier2_selfplay_within_turn.json --init <bc.kfc>
    python -m ml.selfplay_train --run dmc-s0 --config tier4_dmc.json --mode dmc [--init <bc.kfc>]

**Topology** (one machine): this process is the learner and the
orchestrator. It starts one inference-server process (the only one that
touches CUDA for acting; it hot-reloads whichever checkpoint `best.json`
names) and `selfplay.workers` actor processes (`agent.selfplay`), each
playing its share of the game budget into its own append-only shard. The
learner tails the shards into a replay buffer and trains.

**Targets and loss** (search mode):

    L = CE(pi, p) + 1.0 MSE(z, v) + 0.25 BCE(belief) + 0.25 MSE(oracle)   (+ AdamW weight decay)

pi = the root visit distribution of full searches only (playout-cap
randomization); multi-select positions train the subset head on their
candidate distribution. z = the outcome from that seat. DMC mode (M8)
regresses Q(s, a) toward z for the action actually taken instead.

**Gating.** Every `gate_every_steps` gradient steps: an SPRT of the
candidate against the current best (paired seeds, both seat orders and
deck assignments, up to `gate_games` games), run by `ml.selfplay_gate` in
its own process while the learner keeps training; at most one gate is in
flight, and the next is due `gate_every_steps` after the last one started.
Promotion only on a pass; every promoted checkpoint is kept (the frozen
anchors for M9) and journalled. The first gate is against the warm start,
which is gate G4.

**Variable budgets** (M7): the run is defined by a game budget, is
resumable after a kill at any point (the learner saves weights, optimizer
state and its step to `learner_state.kfc` every few minutes and at every
gate; an in-flight gate is replayed; the buffer is rebuilt from the shards;
actors resume exactly the missing game indices, and exit if the learner
dies), and every mid-run change is recorded in the run journal with the
game index.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from agent import spec
from agent.selfplay import EXIT_SHARD_BUSY, read_frames
from agent.telemetry import Run
from keyforge.infoset import ZONE

from .accumulate import backward_accumulated, slices
from .checkpoints import CheckpointStore, load_model, save_model
from .encode import Batch, collate
from .layers import set_activation_checkpointing
from .model import KeyForgeNet, amp_dtype, candidates_tensor, compile_trunk
from .selfplay_gate import GATE_SPRT, gate_verdict  # noqa: F401 -- re-exported (tests, tools)

OPP_UNSEEN = ZONE["opp_unseen"]
_NON_GUI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MULTI_KINDS = ("CHOOSE_CARDS", "ORDER_EFFECTS")

# Mirror augmentation (M7): reversing both battlelines swaps left/right.
_FLANK_OFF = spec.INPLAY.offset("flank")
_BL_OFF = spec.INPLAY.offset("battleline_index")
_OPT_FLANK = spec.OPTION.offset("flank")
_POSITIONAL_ZONES = [ZONE[z] for z in ("my_creature", "opp_creature", "my_upgrade_attached", "opp_upgrade_attached", "my_under", "opp_under")]


def mirror(batch: Batch, which: torch.Tensor) -> None:
    """In place, for rows where `which` is set: swap the left/right flank
    one-hots, reflect the battleline index, and swap CHOOSE_FLANK's
    left/right options. Exact for Fignor/Igor (no card there names a side,
    and none of the four engine sites that force a left placement is in
    either deck); re-check per pool (M11)."""
    if not bool(which.any()):
        return
    rows = which.nonzero().squeeze(-1)
    ip = batch.inplay[rows]
    left = ip[..., _FLANK_OFF].clone()
    ip[..., _FLANK_OFF] = ip[..., _FLANK_OFF + 1]
    ip[..., _FLANK_OFF + 1] = left
    positional = torch.zeros_like(batch.zones[rows], dtype=torch.bool)
    for z in _POSITIONAL_ZONES:
        positional |= batch.zones[rows] == z
    bl = ip[..., _BL_OFF]
    ip[..., _BL_OFF] = torch.where(positional, 1.0 - bl, bl)
    batch.inplay[rows] = ip
    op = batch.options[rows]
    l2 = op[..., _OPT_FLANK].clone()
    op[..., _OPT_FLANK] = op[..., _OPT_FLANK + 1]
    op[..., _OPT_FLANK + 1] = l2
    batch.options[rows] = op


class Buffer:
    """The most recent `max_games` games' positions, sampled uniformly.
    Measured in games, not positions (M11: buffer composition), with a
    position cap as a memory backstop."""

    def __init__(self, max_positions: int, max_games: int):
        self.games: deque = deque()
        self.max_positions = max_positions
        self.max_games = max_games
        self.size = 0
        self.total_games = 0
        self.total_positions = 0

    def add(self, record: dict) -> int:
        pos = []
        for p in record["positions"]:
            p = dict(p)
            p["z"] = float(record["outcome"][p["seat"]])
            pos.append(p)
        self.games.append(pos)
        self.size += len(pos)
        self.total_games += 1
        self.total_positions += len(pos)
        while self.games and (len(self.games) > self.max_games or self.size > self.max_positions):
            self.size -= len(self.games.popleft())
        return len(pos)

    def sample(self, n: int, rng: random.Random) -> List[dict]:
        flat_sizes = [len(g) for g in self.games]
        out = []
        cum = np.cumsum(flat_sizes)
        picks = rng.sample(range(int(cum[-1])), min(n, int(cum[-1])))
        for k in picks:
            gi = int(np.searchsorted(cum, k, side="right"))
            start = int(cum[gi - 1]) if gi else 0
            out.append(self.games[gi][k - start])
        return out


class Tailer:
    """Follows every actor shard in a directory from saved byte offsets."""

    def __init__(self, directory: str, offsets: Optional[Dict[str, int]] = None):
        self.dir = directory
        self.offsets: Dict[str, int] = dict(offsets or {})

    def poll(self) -> List[dict]:
        out = []
        if not os.path.isdir(self.dir):
            return out
        for name in sorted(os.listdir(self.dir)):
            if not name.endswith(".bin"):
                continue
            path = os.path.join(self.dir, name)
            start = self.offsets.get(name, 0)
            for end, rec in read_frames(path, start):
                out.append(rec)
                self.offsets[name] = end
        return out


def _hidden_counts(positions: List[dict], vocab: int, device) -> torch.Tensor:
    """[B, 2*vocab] count vectors of the opponent's true hand and my next
    draws. Index pairs are gathered on the CPU from each position's own
    encoding and written with one accumulate -- never one tiny GPU write
    per card (that was ~13 s per step at batch 1024)."""
    rows, cols = [], []
    for b, p in enumerate(positions):
        ids = p["enc"].card_ids
        for e in p["priv"]["opp_hand"]:
            rows.append(b)
            cols.append(ids[e])
        for e in p["priv"]["next_draws"]:
            if e >= 0:
                rows.append(b)
                cols.append(vocab + ids[e])
    out = torch.zeros(len(positions), 2 * vocab, device=device)
    if rows:
        idx = (torch.tensor(rows, device=device), torch.tensor(cols, device=device))
        out.index_put_(idx, torch.ones(len(rows), device=device), accumulate=True)
    return out


def loss_counts(positions: List[dict], weights: dict, mode: str) -> Dict[str, int]:
    """Each loss's denominator in `losses_for` (the rows its mean is
    over), without a forward pass: what `micro_batch` weights pieces by.
    Kept in step with `losses_for` by the accumulation test."""
    n: Dict[str, int] = defaultdict(int)
    B = len(positions)
    n["rows"] = n["value"] = n["oracle"] = B
    if weights.get("distill", 0.0) > 0:
        n["distill"] = B
    for p in positions:
        n["belief"] += sum(1 for z in p["enc"].zones if z == OPP_UNSEEN)
        if p["target"] is None or p["value_only"]:
            continue
        if mode == "dmc":
            n["q"] += len(p["target"])
        elif p["kind"] in MULTI_KINDS:
            if p["candidates"] and not any(c is None for c in p["candidates"]):
                n["multi"] += 1
        else:
            n["policy"] += 1
    return dict(n)


def backward_step(model: KeyForgeNet, positions: List[dict], device, weights: dict, mode: str, augment: bool, rng: random.Random,
                  micro: Optional[int], amp) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """Forward and backward for one training batch, in pieces of at most
    `micro` positions (ml/accumulate.py); returns the batch's (losses,
    stats). Pieces are taken in order, so mirror augmentation draws the
    same coin per position either way."""
    autocast = lambda: torch.autocast(torch.device(device).type, dtype=amp or torch.float32, enabled=amp is not None)
    spans = slices(len(positions), micro)
    if len(spans) == 1:
        with autocast():
            total, L, st = losses_for(model, positions, device, weights, mode, augment, rng)
        total.backward()
        return L, st
    return backward_accumulated(
        [positions[a:b] for a, b in spans],
        lambda piece: loss_counts(piece, weights, mode),
        lambda piece: losses_for(model, piece, device, weights, mode, augment, rng),
        weights, autocast, stat_counts={"policy_entropy": "policy", "value_calibration": "value"},
    )


def losses_for(model: KeyForgeNet, positions: List[dict], device, weights: dict, mode: str, augment: bool, rng: random.Random):
    batch = collate([p["enc"] for p in positions], device)
    if augment:
        which = torch.tensor([rng.random() < 0.5 for _ in positions], device=device)
        mirror(batch, which)
    out = model.encode_state(batch)
    z = torch.tensor([p["z"] for p in positions], device=device)
    L: Dict[str, torch.Tensor] = {}
    stats: Dict[str, float] = {}
    if mode == "dmc":
        q = model.q_values(out)
        rows, cols, tgt = [], [], []
        for b, p in enumerate(positions):
            if p["target"] is None or p["value_only"]:
                continue
            for a in p["target"]:
                rows.append(b)
                cols.append(a)
                tgt.append(p["z"])
        if rows:
            L["q"] = F.mse_loss(q[torch.tensor(rows, device=device), torch.tensor(cols, device=device)], torch.tensor(tgt, device=device))
    else:
        single_rows = []
        multi = []
        K = batch.options.shape[1]
        for b, p in enumerate(positions):
            if p["target"] is None or p["value_only"]:
                continue
            if p["kind"] in MULTI_KINDS:
                if not p["candidates"] or any(c is None for c in p["candidates"]):
                    continue  # a fixed-policy fallback (shards from before the actor stopped recording these)
                multi.append(b)
            else:
                single_rows.append(b)
        if single_rows:
            pi_np = np.zeros((len(single_rows), K), dtype=np.float32)
            for r, b in enumerate(single_rows):
                t = positions[b]["target"]
                pi_np[r, : len(t)] = t
            logits = model.policy_logits(out)[torch.tensor(single_rows, device=device)]
            pi = torch.from_numpy(pi_np).to(device)
            logp = torch.log_softmax(logits, dim=-1).masked_fill(pi == 0, 0.0)
            L["policy"] = -(pi * logp).sum(-1).mean()
            probs = torch.softmax(logits.detach().float(), -1)
            stats["policy_entropy"] = float(-(probs * torch.log(probs.clamp_min(1e-9))).sum(-1).mean())
        if multi:
            c_rows, members, ordered, spans = [], [], [], []
            for b in multi:
                p = positions[b]
                cands = [tuple(c) for c in p["candidates"]]
                spans.append((len(members), len(cands), p["target"]))
                c_rows.extend([b] * len(cands))
                members.extend(cands)
                ordered.extend([p["kind"] == "ORDER_EFFECTS"] * len(cands))
            scores = model.subset_scores(out, torch.tensor(c_rows, device=device), candidates_tensor(members, device), torch.tensor(ordered, device=device))
            cmax = max(n for _s, n, _t in spans)
            padded = torch.full((len(spans), cmax), float("-inf"), device=device)
            tgt = np.zeros((len(spans), cmax), dtype=np.float32)
            src_rows, src_cols, src_idx = [], [], []
            for r, (start, n, t) in enumerate(spans):
                tgt[r, :n] = t
                src_rows.extend([r] * n)
                src_cols.extend(range(n))
                src_idx.extend(range(start, start + n))
            padded[torch.tensor(src_rows, device=device), torch.tensor(src_cols, device=device)] = scores[torch.tensor(src_idx, device=device)].float()
            logp = torch.log_softmax(padded, dim=-1)
            tg = torch.from_numpy(tgt).to(device)
            L["multi"] = -(tg * logp.masked_fill(tg == 0, 0.0)).sum(-1).mean()
    v = model.value(out)
    L["value"] = F.mse_loss(v, z)
    stats["value_calibration"] = float((v.detach() - z).abs().mean())
    unseen = batch.zones == OPP_UNSEEN
    if bool(unseen.any()):
        hand_np = np.zeros(tuple(batch.zones.shape), dtype=np.float32)
        for b, p in enumerate(positions):
            if p["priv"]["opp_hand"]:
                hand_np[b, p["priv"]["opp_hand"]] = 1.0
        in_hand = torch.from_numpy(hand_np).to(device)
        L["belief"] = F.binary_cross_entropy_with_logits(model.belief_logits(out)[unseen], in_hand[unseen])
    oracle = model.oracle_value(out, _hidden_counts(positions, model.vocab_size, device))
    L["oracle"] = F.mse_loss(oracle, z)
    if weights.get("distill", 0.0) > 0:
        # Suphx-style oracle guidance, the distillation route (M6): the
        # student value regresses onto the oracle's (a far less noisy
        # target than the outcome), never the other way round.
        L["distill"] = F.mse_loss(v, oracle.detach())
    total = sum(weights.get(k, 1.0) * x for k, x in L.items())
    return total, L, stats


def _start_server(ckpt_path: str, port: int, best_json: str, log_path: str) -> subprocess.Popen:
    with open(log_path, "a") as log:  # the child keeps its own copy of the handle
        return subprocess.Popen(
            [sys.executable, "-m", "ml.infer_server", ckpt_path, "--port", str(port), "--watch", best_json],
            cwd=_NON_GUI, stdout=log, stderr=subprocess.STDOUT,
        )


def _start_actor(run: Run, worker: int, games: int, port: int, mode: str) -> subprocess.Popen:
    with open(os.path.join(run.root, f"actor_{worker:03d}.log"), "a") as log:
        return subprocess.Popen(
            [sys.executable, "-m", "agent.selfplay", "--run", run.run_id, "--worker", str(worker), "--games", str(games),
             "--port", str(port), "--mode", mode],
            cwd=_NON_GUI, stdout=log, stderr=subprocess.STDOUT,
        )


def _start_gate(run: Run, pending: dict, args) -> Tuple[subprocess.Popen, str]:
    out = os.path.join(run.artifact_dir("gates"), f"gate_{pending['number']:04d}.json")
    with open(os.path.join(run.root, "gate.log"), "a") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "ml.selfplay_gate", "--run", run.run_id, "--candidate", pending["candidate"],
             "--best", pending["best"], "--number", str(pending["number"]), "--sims", str(args.gate_sims),
             "--mode", args.mode, "--device", args.device, "--out", out],
            cwd=_NON_GUI, stdout=log, stderr=subprocess.STDOUT,
        )
    return proc, out


STATE_FILE = "learner_state.kfc"  # one overwritten file -- not a store entry per save
SAVE_EVERY_SECONDS = 300.0
RESTART_LIMIT = (5, 600.0)  # at most 5 restarts of one child per 10 minutes; after that it stays down (journalled)


def _may_restart(history: deque, now: float) -> bool:
    while history and now - history[0] > RESTART_LIMIT[1]:
        history.popleft()
    if len(history) >= RESTART_LIMIT[0]:
        return False
    history.append(now)
    return True


def cosine_schedule(opt, total_steps: int, lr_min: float, step: int, lr: float):
    """The learner's cosine decay, positioned at `step`. After a resume the
    optimizer's saved lr is already decayed, and CosineAnnealingLR is
    recursive (each step scales the *current* lr) -- replaying it from the
    saved lr would decay twice. So every group restarts from its initial lr
    and the schedule is replayed from step 0."""
    for g in opt.param_groups:
        g["lr"] = g.get("initial_lr", lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=lr_min)
    for _ in range(step):
        sched.step()
    return sched


def _write_json(path: str, obj: dict) -> None:
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(path + ".tmp", path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--config", default="tier2_selfplay_within_turn.json")
    parser.add_argument("--init", default=None, help="warm-start checkpoint (the M3 BC network)")
    parser.add_argument("--mode", choices=("search", "dmc"), default="search")
    parser.add_argument("--games", type=int, default=None, help="override the config's game budget (journalled)")
    parser.add_argument("--port", type=int, default=6150)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gate-sims", type=int, default=50)
    parser.add_argument("--no-actors", action="store_true", help="learn from existing shards only")
    args = parser.parse_args()

    run = Run.resume_or_create(args.run, args.config)
    cfg = run.config
    sp = cfg["selfplay"]
    if args.mode == "search" and cfg["search"]["resample"] != "all":
        # A resumed run keeps the config it started with, so this also stops
        # a run created before the default was fixed from carrying on.
        raise SystemExit(f"run {args.run!r} searches with resample={cfg['search']['resample']!r}, which leaves true hidden "
                         "cards in every world (plan status, departure 4) -- its data can't train an honest agent. "
                         "Start a new run id with resample 'all'.")
    budget = args.games or int(sp["games"])
    if args.games:
        run.journal("budget_override", game_index=0, games=budget)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    store = CheckpointStore(os.path.join(run.root, "checkpoints"))
    best_json = os.path.join(run.root, "best.json")
    state_json = os.path.join(run.root, "learner.json")
    rng = random.Random(int(cfg["seed"]))
    torch.manual_seed(int(cfg["seed"]))

    # ------------------------------------------------ model, optimizer, resume
    net_over = {"dropout": cfg["network"].get("dropout", 0.0)}
    state = json.load(open(state_json)) if os.path.exists(state_json) else None
    step = 0
    if state is not None:
        # "state_file": the learner's own overwritten state (current format);
        # "checkpoint": an older learner.json that pointed into the store.
        src = os.path.join(run.root, state["state_file"]) if "state_file" in state else store.path_of(state["checkpoint"])
        model, meta, opt_state = load_model(src, net_overrides=net_over)
        # The step saved *with the weights* wins over learner.json's (a kill
        # between the two writes must not replay or skip steps).
        step = int((meta.get("extra") or {}).get("step", state["step"]))
        run.journal("learner_resumed", game_index=state["games_consumed"], step=step)
    elif args.init:
        model, _meta, opt_state = load_model(args.init, net_overrides=net_over)
        opt_state = None
    else:
        model, opt_state = KeyForgeNet(dict(cfg["network"], dropout=0.0)), None
    model.to(device)
    # Fused: one kernel for the whole update (see ml/bc_train.py).
    opt = torch.optim.AdamW(model.parameters(), lr=float(sp["lr"]), weight_decay=float(sp["weight_decay"]),
                            fused=torch.device(device).type == "cuda")
    if opt_state is not None and opt_state.get("state"):
        opt.load_state_dict({"state": opt_state["state"], "param_groups": opt_state["param_groups"]})
    amp = amp_dtype(sp.get("precision", "fp32"), device)
    set_activation_checkpointing(model.trunk, bool(sp.get("activation_checkpointing", False)))
    compile_trunk(model, sp.get("compile", False), device)
    positions_per_game = 200.0  # ~140 non-forced decisions + half as many value-only records
    total_steps = max(1, int(budget * positions_per_game / sp["positions_per_step"]))
    sched = cosine_schedule(opt, total_steps, float(sp["lr_min"]), step, float(sp["lr"]))

    if not os.path.exists(best_json):
        h, digest = save_model(model, store, config_hash=run.config_hash, extra={"kind": "warm_start", "step": 0})
        _write_json(best_json, {"checkpoint": h, "step": 0, "games": 0, "promotions": 0})
        run.journal("promoted", game_index=0, checkpoint=h, reason="warm start")
    best = json.load(open(best_json))

    gate_count = int(state.get("gates", 0)) if state else 0
    if state and "gates" not in state:
        gate_count = sum(1 for j in run.journal_entries() if j["event"] in ("gate", "gate_failed"))
    last_gate = int(state.get("last_gate", step)) if state else step
    pending: Optional[dict] = state.get("gate_pending") if state else None

    def save_state(games: int) -> None:
        path = os.path.join(run.root, STATE_FILE)
        save_model(model, path + ".tmp", config_hash=run.config_hash, optimizer=opt, extra={"kind": "learner_state", "step": step})
        os.replace(path + ".tmp", path)
        _write_json(state_json, {"state_file": STATE_FILE, "step": step, "games_consumed": games, "gates": gate_count,
                                 "last_gate": last_gate, "gate_pending": pending})

    # ------------------------------------------------ processes
    procs: Dict[int, subprocess.Popen] = {}
    restarts: Dict[object, deque] = {}
    server = None
    gate_proc: Optional[subprocess.Popen] = None
    gate_out: Optional[str] = None
    workers = int(sp["workers"])
    per_worker = math.ceil(budget / workers)

    def stop_children() -> None:
        for p in list(procs.values()) + [server, gate_proc]:
            if p is not None and p.poll() is None:
                p.terminate()

    def on_term(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, on_term)

    # ------------------------------------------------ learning loop
    tailer = Tailer(run.artifact_dir("shards"))
    buffer = Buffer(int(sp["buffer_positions"]), max(200, int(sp["buffer_positions"] // 150)))
    for rec in tailer.poll():  # rebuild the buffer (resume) or pick up anything already there
        buffer.add(rec)
    metrics = run.metrics("learner")
    weights = dict(sp["loss_weights"])
    if args.mode == "dmc":
        weights["q"] = 1.0
    augment = bool(sp.get("mirror_augmentation", True))
    if augment:
        from agent.augment import mirror_safe

        if not mirror_safe(cfg["pool"]["decks"]):
            augment = False
            run.journal("mirror_augmentation_off", game_index=0, reason="a card in the pool breaks the left/right symmetry")
    if pending is not None:
        # A gate was in flight when the learner stopped: read its verdict
        # if it finished, otherwise play it again (same seeds).
        out = os.path.join(run.artifact_dir("gates"), f"gate_{pending['number']:04d}.json")
        if not os.path.exists(out):
            run.journal("gate_resumed", game_index=buffer.total_games, number=pending["number"], candidate=pending["candidate"])
            gate_proc, gate_out = _start_gate(run, pending, args)
        else:
            gate_out = out
    t_last = time.time()
    t_saved = time.time()
    try:
        if not args.no_actors:
            server = _start_server(store.path_of(best["checkpoint"]), args.port, best_json, os.path.join(run.root, "server.log"))
            time.sleep(8)
            for w in range(workers):
                procs[w] = _start_actor(run, w, per_worker, args.port, args.mode)
        while True:
            new = tailer.poll()
            for rec in new:
                n = buffer.add(rec)
                metrics.count("positions_produced", n)
                metrics.count("games_seen")
                meta = rec["meta"]
                if meta.get("resigned_by") is not None:
                    metrics.count("resignations")
                if meta.get("would_resign") is not None:
                    # Exempt game: would the resignation have been wrong?
                    metrics.count("resign_tests")
                    if rec["outcome"][meta["would_resign"]] >= 0:
                        metrics.count("false_resignations")
            games_done = buffer.total_games
            # Train at most one step per `positions_per_step` new positions.
            allowed = int(buffer.total_positions / sp["positions_per_step"]) - step
            if buffer.size >= int(sp["batch"]) and allowed > 0:
                model.train()
                positions = buffer.sample(int(sp["batch"]), rng)
                opt.zero_grad(set_to_none=True)
                L, st = backward_step(model, positions, device, weights, args.mode, augment, rng, sp.get("micro_batch"), amp)
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float(sp["grad_clip"]))
                opt.step()
                sched.step()
                step += 1
                metrics.count("gradient_steps")
                metrics.count("positions_consumed", len(positions))
                metrics.observe("grad_norm", float(gn))
                for k, v in L.items():
                    metrics.observe(f"loss_{k}", float(v.detach()))
                for k, v in st.items():
                    metrics.observe(k, v)
                metrics.gauge("lr", sched.get_last_lr()[0])
            else:
                time.sleep(0.5)
            # Gauges every pass, not only on training passes: an idle
            # interval's record must still say where the run is.
            metrics.gauge("buffer_positions", buffer.size)
            metrics.gauge("games", games_done)
            metrics.gauge("step", step)
            metrics.gauge("gate_running", gate_proc is not None)
            metrics.tick()

            # ---- gating, in its own process: the learner never waits on it.
            if gate_proc is None and gate_out is None and step - last_gate >= int(sp["gate_every_steps"]):
                last_gate = step
                h, _d = save_model(model, store, config_hash=run.config_hash, extra={"kind": "candidate", "step": step})
                gate_count += 1
                pending = {"candidate": h, "best": best["checkpoint"], "number": gate_count, "step": step, "games": games_done}
                gate_proc, gate_out = _start_gate(run, pending, args)
                save_state(games_done)
                t_saved = time.time()
            if gate_out is not None and (gate_proc is None or gate_proc.poll() is not None):
                result = json.load(open(gate_out)) if os.path.exists(gate_out) else None
                if result is None:
                    run.journal("gate_failed", game_index=games_done, number=pending["number"], candidate=pending["candidate"],
                                exit_code=gate_proc.returncode if gate_proc is not None else None)
                else:
                    metrics.gauge("gate_score", result["score"])
                    run.journal("gate", game_index=games_done, step=pending["step"], candidate=pending["candidate"],
                                best=pending["best"], verdict=result["verdict"], score=round(result["score"], 4),
                                games=result["games"], number=pending["number"], seconds=result.get("seconds"))
                    if result["verdict"] == "H1":
                        best = {"checkpoint": pending["candidate"], "step": pending["step"], "games": games_done,
                                "promotions": best.get("promotions", 0) + 1}
                        _write_json(best_json, best)
                        run.journal("promoted", game_index=games_done, checkpoint=pending["candidate"], step=pending["step"])
                        if best["promotions"] == 1:
                            run.journal("gate_G4", game_index=games_done, verdict="PASS", step=pending["step"])
                gate_proc, gate_out, pending = None, None, None
                save_state(games_done)
                t_saved = time.time()
            if time.time() - t_saved > SAVE_EVERY_SECONDS:
                # A kill costs at most this much learning (the plan: "at most
                # the in-flight games" for data; the learner's share is here).
                save_state(games_done)
                t_saved = time.time()

            # ---- supervision: restart a crashed child (journalled, rate-limited);
            # finish when every actor is done and the buffer is consumed.
            alive = 0
            now = time.time()
            for w, p in list(procs.items()):
                code = p.poll()
                if code is None:
                    alive += 1
                elif code == 0:
                    continue
                elif code == EXIT_SHARD_BUSY:
                    run.journal("actor_refused", game_index=games_done, worker=w, reason="shard held by another live process")
                    del procs[w]
                elif _may_restart(restarts.setdefault(w, deque()), now):
                    run.journal("actor_restarted", game_index=games_done, worker=w, exit_code=code)
                    procs[w] = _start_actor(run, w, per_worker, args.port, args.mode)
                    alive += 1
                else:
                    run.journal("actor_abandoned", game_index=games_done, worker=w, exit_code=code,
                                reason=f"{RESTART_LIMIT[0]} restarts in {RESTART_LIMIT[1]:.0f}s")
                    del procs[w]
            if server is not None and server.poll() is not None and alive:
                if _may_restart(restarts.setdefault("server", deque()), now):
                    run.journal("server_restarted", game_index=games_done, exit_code=server.returncode)
                    server = _start_server(store.path_of(best["checkpoint"]), args.port, best_json, os.path.join(run.root, "server.log"))
                else:
                    run.journal("server_abandoned", game_index=games_done, exit_code=server.returncode)
                    raise SystemExit("the inference server keeps dying -- see server.log")
            starved = buffer.size < int(sp["batch"])  # nothing more will ever arrive to fill a batch
            if (args.no_actors or alive == 0) and (allowed <= 0 or starved) and not new and gate_out is None:
                break
            if time.time() - t_last > 60:
                t_last = time.time()
                print(f"[learner] step {step} games {games_done} buffer {buffer.size} alive {alive}"
                      f"{' gate ' + str(pending['number']) + ' running' if pending else ''}", flush=True)
    except (KeyboardInterrupt, SystemExit):
        save_state(buffer.total_games)
        raise
    finally:
        stop_children()

    h, _d = save_model(model, store, config_hash=run.config_hash, optimizer=opt, extra={"kind": "learner", "step": step, "final": True})
    save_state(buffer.total_games)
    run.journal("run_finished", game_index=buffer.total_games, step=step, final=h)
    metrics.flush()


if __name__ == "__main__":
    main()

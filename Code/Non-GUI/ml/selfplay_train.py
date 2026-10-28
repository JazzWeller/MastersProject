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

**Network v2** (`observation.version` 2): the actors search with v2 leaves
and record compact positions; the learner replays each game to encode them
(`ml/selfplay_v2.py`, in a pool of `selfplay.encode_workers` processes) and
trains `KeyForgeNetV2` with the soft visit targets plus BC v2's belief,
next-draw and oracle terms. The server is `ml.infer_v2`, the gate searches
with v2 leaves. Mirror augmentation is v1 only.

**Reuse limits and replay** (after Tier 2 round 2): `selfplay.policy_reuse`
and `value_reuse` cap how often one position trains its policy target and its
outcome targets (`UseCounts`; the value head memorized whole games when
every position's z trained ~6 times). `--replay-from RUN` trains a new
learner on another run's games in the order they arrived, at the pace its
own credit allows, without actors or gates, and ends with the losses on
every held-out position: learner settings compared on the same games.

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
from agent.config import held_out, positions_per_step
from agent.telemetry import Run
from keyforge.infoset import ZONE

from .accumulate import backward_accumulated, slices
from .checkpoints import CheckpointStore, load_model, save_model
from .encode import Batch, collate
from .layers import set_activation_checkpointing
from .resume import PAUSED
from .model import KeyForgeNet, amp_dtype, candidates_tensor, compile_trunk, gpu_generation_ok
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


class UseCounts:
    """`selfplay.policy_reuse` / `value_reuse`: how many times one position
    may train its policy target and its outcome targets (None: no limit).
    Kept for a buffer's positions in buffer order, in one array (games leave
    oldest first, so dropping one is a slice): whether each has a policy
    target, and how often each group has trained. `pick` samples uniformly
    among positions with a group left and spends what it takes."""

    def __init__(self, policy: Optional[int] = None, value: Optional[int] = None):
        self.policy = None if policy is None else int(policy)
        self.value = None if value is None else int(value)
        self.u = np.zeros((3, 0), dtype=np.int32)  # rows: has a policy target, policy uses, value uses
        self.lengths: deque = deque()

    @property
    def active(self) -> bool:
        return self.policy is not None or self.value is not None

    def add(self, has_policy) -> None:
        g = np.zeros((3, len(has_policy)), dtype=np.int32)
        g[0] = has_policy
        self.u = np.concatenate([self.u, g], axis=1)
        self.lengths.append(len(has_policy))

    def drop_oldest(self) -> None:
        self.u = self.u[:, self.lengths.popleft():]

    def _open(self) -> Tuple[np.ndarray, np.ndarray]:
        pol = self.u[0] > 0 if self.policy is None else (self.u[0] > 0) & (self.u[1] < self.policy)
        val = np.ones(self.u.shape[1], dtype=bool) if self.value is None else self.u[2] < self.value
        return pol, val

    def trainable(self) -> int:
        """Positions with a group left to train."""
        pol, val = self._open()
        return int(np.count_nonzero(pol | val))

    def pick(self, n: int, rng: random.Random) -> List[Tuple[int, int, bool, bool]]:
        """Up to `n` positions as (game, position in it, policy on, value
        on), in buffer order; the groups they train are counted as spent."""
        pol, val = self._open()
        elig = np.flatnonzero(pol | val)
        flat = elig[sorted(rng.sample(range(len(elig)), min(n, len(elig))))]
        self.u[1, flat] += pol[flat]
        self.u[2, flat] += val[flat]
        starts = np.concatenate([[0], np.cumsum(self.lengths)])
        gi = np.searchsorted(starts, flat, side="right") - 1
        return [(int(g), int(k - starts[g]), bool(pol[k]), bool(val[k])) for g, k in zip(gi, flat)]


class Buffer:
    """The most recent `max_games` games' positions, sampled uniformly.
    Measured in games, not positions (M11: buffer composition), with a
    position cap as a memory backstop. With reuse limits (`UseCounts`),
    only positions with a target group left are sampled, and a spent
    group's targets are dropped from the sampled copy."""

    def __init__(self, max_positions: int, max_games: int, limits: Optional[UseCounts] = None):
        self.games: deque = deque()
        self.max_positions = max_positions
        self.max_games = max_games
        self.limits = limits if limits is not None and limits.active else None
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
        if self.limits:
            self.limits.add([p["target"] is not None and not p["value_only"] for p in pos])
        self.size += len(pos)
        self.total_games += 1
        self.total_positions += len(pos)
        while self.games and (len(self.games) > self.max_games or self.size > self.max_positions):
            self.size -= len(self.games.popleft())
            if self.limits:
                self.limits.drop_oldest()
        return len(pos)

    def trainable(self) -> int:
        return self.limits.trainable() if self.limits else self.size

    def batches(self, n: int):
        """Every position in buffer order, `n` at a time (full validation)."""
        flat = [p for g in self.games for p in g]
        for a in range(0, len(flat), n):
            yield flat[a:a + n]

    def sample(self, n: int, rng: random.Random) -> List[dict]:
        if self.limits:
            out = []
            for gi, j, pol, val in self.limits.pick(n, rng):
                p = self.games[gi][j]
                if not pol and p["target"] is not None:
                    p = dict(p, target=None)
                if not val:
                    p = dict(p, value_on=False)
                out.append(p)
            return out
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


def count_frames(path: str) -> int:
    """Complete frames in a shard, from the length prefixes alone."""
    from agent.selfplay import FRAME

    size, n, at = os.path.getsize(path), 0, 0
    with open(path, "rb") as f:
        while True:
            f.seek(at)
            head = f.read(4)
            if len(head) < 4:
                return n
            at += 4 + FRAME.unpack(head)[0]
            if at > size:
                return n
            n += 1


class ReplayReader:
    """Another run's shards, as the games arrived, a few at a time
    (`--replay-from`): the learner trains on them as it would have while
    they came in, without actors. A shard is one actor's games in the order
    they finished and the actors play at one rate, so frame k of a shard of
    N arrived about k/N of the way through the run; the shards are merged on
    that. Only the frames complete when the reader opens are read (a source
    run still playing is a snapshot)."""

    def __init__(self, directory: str):
        import heapq

        self._heapq = heapq
        self._heap = []
        self.total = 0
        for name in sorted(os.listdir(directory)):
            if not name.endswith(".bin"):
                continue
            path = os.path.join(directory, name)
            n = count_frames(path)
            if n:
                self.total += n
                frames = (rec for _end, rec in read_frames(path))
                heapq.heappush(self._heap, (0.5 / n, name, 0, n, frames))
        self.read = 0

    @property
    def done(self) -> bool:
        return not self._heap

    def poll(self, limit: int = 64) -> List[dict]:
        out = []
        while self._heap and len(out) < limit:
            _at, name, k, n, frames = self._heapq.heappop(self._heap)
            out.append(next(frames))
            if k + 1 < n:
                self._heapq.heappush(self._heap, ((k + 1.5) / n, name, k + 1, n, frames))
        self.read += len(out)
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
    n["rows"] = len(positions)
    n["value"] = n["oracle"] = sum(1 for p in positions if p.get("value_on", True))
    if weights.get("distill", 0.0) > 0:
        n["distill"] = n["value"]
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
    on = value_rows([p.get("value_on", True) for p in positions], device)
    if on is None or on.numel():
        L["value"] = F.mse_loss(_rows(v, on), _rows(z, on))
        stats["value_calibration"] = float((_rows(v, on).detach() - _rows(z, on)).abs().mean())
    unseen = batch.zones == OPP_UNSEEN
    if bool(unseen.any()):
        hand_np = np.zeros(tuple(batch.zones.shape), dtype=np.float32)
        for b, p in enumerate(positions):
            if p["priv"]["opp_hand"]:
                hand_np[b, p["priv"]["opp_hand"]] = 1.0
        in_hand = torch.from_numpy(hand_np).to(device)
        L["belief"] = F.binary_cross_entropy_with_logits(model.belief_logits(out)[unseen], in_hand[unseen])
    if on is None or on.numel():
        oracle = _rows(model.oracle_value(out, _hidden_counts(positions, model.vocab_size, device)), on)
        L["oracle"] = F.mse_loss(oracle, _rows(z, on))
        if weights.get("distill", 0.0) > 0:
            # Suphx-style oracle guidance, the distillation route (M6): the
            # student value regresses onto the oracle's (a far less noisy
            # target than the outcome), never the other way round.
            L["distill"] = F.mse_loss(_rows(v, on), oracle.detach())
    total = sum(weights.get(k, 1.0) * x for k, x in L.items())
    return total, L, stats


def value_rows(flags, device) -> Optional[torch.Tensor]:
    """The row indices whose outcome targets train (`UseLimits`), or None
    when every row's do."""
    flags = [bool(f) for f in flags]
    if all(flags):
        return None
    return torch.tensor([b for b, f in enumerate(flags) if f], dtype=torch.long, device=device)


def _rows(x: torch.Tensor, on: Optional[torch.Tensor]) -> torch.Tensor:
    return x if on is None else x[on]


def _start_server(ckpt_path: str, port: int, best_json: str, log_path: str, v2: bool = False) -> subprocess.Popen:
    argv = ([sys.executable, "-m", "ml.infer_v2", "--checkpoint", ckpt_path] if v2
            else [sys.executable, "-m", "ml.infer_server", ckpt_path])
    with open(log_path, "a") as log:  # the child keeps its own copy of the handle
        return subprocess.Popen(argv + ["--port", str(port), "--watch", best_json],
                                cwd=_NON_GUI, stdout=log, stderr=subprocess.STDOUT)


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


def _history_form(cfg: dict) -> str:
    """What a v2 batch carries for the run's network (`dataset_v2.assemble`)."""
    from agent.search.leaf_v2 import history_form

    return history_form(cfg["network"].get("history_arch", "none"))


def _write_json(path: str, obj: dict) -> None:
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(path + ".tmp", path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--config", default="tier2_selfplay_within_turn.json")
    parser.add_argument("--init", default=None, help="warm-start checkpoint (the M3 BC network)")
    parser.add_argument("--init-run", default=None, help="warm-start from a BC run's checkpoint, by run id (v2: ml.bc_train_v2)")
    parser.add_argument("--override", default=None,
                        help="JSON merged over the config when the run is created (e.g. a cluster's worker count); "
                             "a resumed run keeps the config it started with")
    parser.add_argument("--mode", choices=("search", "dmc"), default="search")
    parser.add_argument("--games", type=int, default=None, help="override the config's game budget (journalled)")
    parser.add_argument("--port", type=int, default=6150)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gate-sims", type=int, default=50)
    parser.add_argument("--no-actors", action="store_true", help="learn from existing shards only")
    parser.add_argument("--replay-from", default=None, metavar="RUN",
                        help="learn from another run's games, in arrival order and at the pace its config's credit "
                             "allows, with no actors and no gates: learner settings compared offline on the same games")
    args = parser.parse_args()
    if args.replay_from:
        args.no_actors = True
    if args.init_run and not args.init:
        from tools.eval_v2 import _checkpoint

        args.init = _checkpoint(args.init_run)

    run = Run.resume_or_create(args.run, args.config, json.loads(args.override) if args.override else None)
    cfg = run.config
    sp = cfg["selfplay"]
    if args.mode == "search" and cfg["search"]["resample"] != "all":
        # A resumed run keeps the config it started with, so this also stops
        # a run created before the default was fixed from carrying on.
        raise SystemExit(f"run {args.run!r} searches with resample={cfg['search']['resample']!r}, which leaves true hidden "
                         "cards in every world (plan status, departure 4) -- its data can't train an honest agent. "
                         "Start a new run id with resample 'all'.")
    reader = None
    if args.replay_from:
        from agent.telemetry import runs_root

        reader = ReplayReader(os.path.join(runs_root(), args.replay_from, "shards"))
        if not reader.total:
            raise SystemExit(f"run {args.replay_from!r} has no games to replay")
        run.journal("replay", game_index=0, source=args.replay_from, games=reader.total)
    budget = args.games or (reader.total if reader else int(sp["games"]))
    if args.games:
        run.journal("budget_override", game_index=0, games=budget)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    store = CheckpointStore(os.path.join(run.root, "checkpoints"))
    best_json = os.path.join(run.root, "best.json")
    state_json = os.path.join(run.root, "learner.json")
    rng = random.Random(int(cfg["seed"]))
    torch.manual_seed(int(cfg["seed"]))

    # ------------------------------------------------ model, optimizer, resume
    v2 = int(cfg.get("observation", {}).get("version", 1)) == 2
    if v2 and args.mode == "dmc":
        raise SystemExit("DMC self-play is v1 only")
    if v2:
        from .checkpoints import load_model_v2_state
        from .model_v2 import KeyForgeNetV2, checkpoint_stamp

        stamp = {**checkpoint_stamp()}  # load_model_v2 refuses a checkpoint without it
        load = lambda src, over: load_model_v2_state(src, net_overrides=over)
        fresh = lambda: KeyForgeNetV2(dict(cfg["network"], dropout=0.0))
    else:
        stamp = {}
        load = lambda src, over: load_model(src, net_overrides=over)
        fresh = lambda: KeyForgeNet(dict(cfg["network"], dropout=0.0))
    net_over = {"dropout": cfg["network"].get("dropout", 0.0)}
    state = json.load(open(state_json)) if os.path.exists(state_json) else None
    step = 0
    if state is not None:
        # "state_file": the learner's own overwritten state (current format);
        # "checkpoint": an older learner.json that pointed into the store.
        src = os.path.join(run.root, state["state_file"]) if "state_file" in state else store.path_of(state["checkpoint"])
        model, meta, opt_state = load(src, net_over)
        # The step saved *with the weights* wins over learner.json's (a kill
        # between the two writes must not replay or skip steps).
        step = int((meta.get("extra") or {}).get("step", state["step"]))
        run.journal("learner_resumed", game_index=state["games_consumed"], step=step)
    elif args.init:
        model, _meta, opt_state = load(args.init, net_over)
        opt_state = None
    else:
        model, opt_state = fresh(), None
    model.to(device)
    # Fused: one kernel for the whole update (see ml/bc_train.py).
    opt = torch.optim.AdamW(model.parameters(), lr=float(sp["lr"]), weight_decay=float(sp["weight_decay"]),
                            fused=torch.device(device).type == "cuda")
    if opt_state is not None and opt_state.get("state"):
        opt.load_state_dict({"state": opt_state["state"], "param_groups": opt_state["param_groups"]})
    amp = amp_dtype(sp.get("precision", "fp32"), device)
    set_activation_checkpointing(model.trunk, bool(sp.get("activation_checkpointing", False)))
    if v2:
        if sp.get("compile", False) and gpu_generation_ok(device):
            model.compile_inputs()
    else:
        compile_trunk(model, sp.get("compile", False), device)
    positions_per_game = 200.0  # ~140 non-forced decisions + half as many value-only records
    pps = positions_per_step(sp)
    total_steps = max(1, int(budget * positions_per_game / pps))
    sched = cosine_schedule(opt, total_steps, float(sp["lr_min"]), step, float(sp["lr"]))

    if not os.path.exists(best_json):
        h, digest = save_model(model, store, config_hash=run.config_hash, extra={"kind": "warm_start", "step": 0, **stamp})
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
        save_model(model, path + ".tmp", config_hash=run.config_hash, optimizer=opt, extra={"kind": "learner_state", "step": step, **stamp})
        os.replace(path + ".tmp", path)
        _write_json(state_json, {"state_file": STATE_FILE, "step": step, "games_consumed": games, "gates": gate_count,
                                 "last_gate": last_gate, "gate_pending": pending, "credit": credit})

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
        # State is saved on the way out, so this is a pause: the same command
        # resumes (run.py requeues a Slurm job on this code).
        raise SystemExit(PAUSED)

    signal.signal(signal.SIGTERM, on_term)

    # ------------------------------------------------ learning loop
    tailer = reader or Tailer(run.artifact_dir("shards"))
    limits = UseCounts(sp.get("policy_reuse"), sp.get("value_reuse"))
    # A replay that resumes reads its source again from the start: the games
    # it had consumed refill the buffer without earning credit (their use
    # counts start over).
    catch_up = int(state["games_consumed"]) if state and reader else 0
    inbox: deque = deque()  # arrived games not yet in a buffer (a replay takes them as its credit runs out)
    lookahead = 4 * max(1, int(sp.get("encode_workers", 2))) + 16
    max_games = max(200, int(sp["buffer_positions"] // 150))
    metrics = run.metrics("learner")
    encoder = None
    val_fraction = float(sp.get("validation_fraction", 0.0))
    val_games = int(sp.get("validation_games", 400))
    # The training credit (steps the learner may take): +1 per `pps` new
    # positions, -1 per step, and never more than one buffer's worth -- a
    # learner that falls behind lets older positions go rather than train
    # on a frozen buffer once the actors stop (Tier 2 round 1: 685k steps on
    # the last buffer, the value head memorized it).
    credit = float(state.get("credit", 0.0)) if state else 0.0
    fresh_run = state is None

    def take(n: int) -> None:
        nonlocal credit
        cap = max(1.0, buffer.size / pps)
        credit += n / pps
        if credit > cap:
            metrics.count("positions_unlearned", int((credit - cap) * pps))
            credit = cap

    def is_val(meta) -> bool:
        return held_out(int(meta["seed"]), val_fraction)

    if v2:
        from .selfplay_v2 import DEFAULT_WEIGHTS as V2_WEIGHTS
        from .selfplay_v2 import BufferV2, EncodingPool, backward_step_v2, losses_v2
        from .selfplay_v2 import loss_counts as loss_counts_v2

        buffer = BufferV2(int(sp["buffer_positions"]), max_games, _history_form(cfg), limits)
        val = BufferV2(val_games * 250, val_games, _history_form(cfg))
        encoder = EncodingPool(int(sp.get("encode_workers", 2)))
        weights = dict(V2_WEIGHTS, **sp["loss_weights"])
        if not reader:
            encoder.submit(tailer.poll())  # rebuild the buffer (resume) or pick up anything already there
        for arrays, info in encoder.wait_all():
            if is_val(info["meta"]):
                val.add(arrays)
            elif fresh_run:
                take(buffer.add(arrays))
            else:
                buffer.add(arrays)
    else:
        buffer = Buffer(int(sp["buffer_positions"]), max_games, limits)
        val = Buffer(val_games * 250, val_games)
        for rec in ([] if reader else tailer.poll()):  # rebuild the buffer (resume) or pick up anything already there
            if is_val(rec["meta"]):
                val.add(rec)
            elif fresh_run:
                take(buffer.add(rec))
            else:
                buffer.add(rec)
        weights = dict(sp["loss_weights"])

    @torch.no_grad()
    def validate() -> None:
        """Value (and policy) loss on held-out games, the same losses the
        learner trains -- the memorization check."""
        if val.size < 64:
            return
        vrng = random.Random(step)
        model.eval()
        try:
            with torch.autocast(torch.device(device).type, dtype=amp or torch.float32, enabled=amp is not None):
                if v2:
                    b, t, p, c = val.sample(256, vrng)
                    _tot, VL, _st = losses_v2(model, b.to(device), t.to(device), p, c, weights, ["policy", "multi", "value"])
                else:
                    _tot, VL, _st = losses_for(model, val.sample(256, vrng), device, {"value": 1.0, "policy": 1.0},
                                               args.mode, False, vrng)
        finally:
            model.train()
        for k in ("value", "policy", "multi"):
            if k in VL:
                metrics.observe(f"val_loss_{k}", float(VL[k]))

    @torch.no_grad()
    def validate_all() -> Dict[str, float]:
        """Every held-out position's losses at the end of the run, each a
        mean over its own rows (as training reports it)."""
        sums: Dict[str, float] = defaultdict(float)
        counts: Dict[str, int] = defaultdict(int)
        heads = ["policy", "multi", "value"]
        model.eval()
        try:
            with torch.autocast(torch.device(device).type, dtype=amp or torch.float32, enabled=amp is not None):
                for chunk in val.batches(512):
                    if v2:
                        b, t, p, c = chunk
                        n = loss_counts_v2(b, t, p, c, heads)
                        _tot, VL, _st = losses_v2(model, b.to(device), t.to(device), p, c, weights, heads)
                    else:
                        n = loss_counts(chunk, {}, args.mode)
                        _tot, VL, _st = losses_for(model, chunk, device, {}, args.mode, False, random.Random(0))
                    for k in heads:
                        if k in VL:
                            sums[k] += float(VL[k]) * n[k]
                            counts[k] += n[k]
        finally:
            model.train()
        return {k: round(sums[k] / counts[k], 5) for k in sums if counts[k]}
    if args.mode == "dmc":
        weights["q"] = 1.0
    augment = bool(sp.get("mirror_augmentation", True)) and not v2
    if augment:
        from agent.augment import mirror_safe

        if cfg["pool"].get("deck_source", "presets") != "presets":
            # Random decks (M11): checking every game's cards isn't worth it; v2 never mirrors anyway.
            augment = False
            run.journal("mirror_augmentation_off", game_index=0, reason="decks vary by game")
        elif not mirror_safe(cfg["pool"]["decks"]):
            augment = False
            run.journal("mirror_augmentation_off", game_index=0, reason="a card in the pool breaks the left/right symmetry")
    if pending is not None:
        # A gate was in flight when the learner stopped: read its verdict
        # if it finished, otherwise play it again (same seeds).
        out = os.path.join(run.artifact_dir("gates"), f"gate_{pending['number']:04d}.json")
        if not os.path.exists(out):
            run.journal("gate_resumed", game_index=buffer.total_games + val.total_games, number=pending["number"], candidate=pending["candidate"])
            gate_proc, gate_out = _start_gate(run, pending, args)
        else:
            gate_out = out
    gate_every = float("inf") if reader else int(sp["gate_every_steps"])  # a replay is compared on held-out loss
    t_last = time.time()
    t_saved = time.time()
    try:
        if not args.no_actors:
            server = _start_server(store.path_of(best["checkpoint"]), args.port, best_json, os.path.join(run.root, "server.log"), v2)
            time.sleep(8)
            for w in range(workers):
                procs[w] = _start_actor(run, w, per_worker, args.port, args.mode)
        while True:
            if reader:  # keep the encoders a little ahead of what the credit will take
                new = reader.poll(max(0, lookahead - len(inbox) - (len(encoder.pending) if v2 else 0)))
            else:
                new = tailer.poll()
            inbox.extend(encoder.ready() if v2 else [(r, r) for r in new])
            if v2:
                encoder.submit(new)
            arrived = []
            while inbox and (not reader or credit < 1 or buffer.total_games + val.total_games < catch_up
                             or buffer.trainable() < int(sp["batch"])):
                catching = buffer.total_games + val.total_games < catch_up
                item, rec = inbox.popleft()
                if is_val(rec["meta"]):
                    n = val.add(item)
                    metrics.count("validation_games")
                else:
                    n = buffer.add(item)
                    if not catching:
                        take(n)
                arrived.append((n, rec))
            # Games still to come: read this pass, waiting, encoding, or unread in a replay's source.
            more = bool(new) or bool(inbox) or bool(v2 and encoder.pending) or bool(reader and not reader.done)
            for n, rec in arrived:
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
            games_done = buffer.total_games + val.total_games
            if credit >= 1 and buffer.trainable() >= int(sp["batch"]):
                credit -= 1
                model.train()
                opt.zero_grad(set_to_none=True)
                if v2:
                    batch, tg, pis, cands = buffer.sample(int(sp["batch"]), rng)
                    autocast = lambda: torch.autocast(torch.device(device).type, dtype=amp or torch.float32,
                                                      enabled=amp is not None)
                    L, st = backward_step_v2(model, (batch.to(device), tg.to(device), pis, cands), weights=weights,
                                             max_tokens=sp.get("max_tokens"), autocast=autocast)
                    positions = pis
                else:
                    positions = buffer.sample(int(sp["batch"]), rng)
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
                if step % int(sp.get("validation_every_steps", 250)) == 0:
                    validate()
            else:
                time.sleep(0.02 if reader else 0.5)
            # Gauges every pass, not only on training passes: an idle
            # interval's record must still say where the run is.
            metrics.gauge("buffer_positions", buffer.size)
            metrics.gauge("games", games_done)
            metrics.gauge("step", step)
            metrics.gauge("credit", round(credit, 2))
            metrics.gauge("validation_positions", val.size)
            metrics.gauge("gate_running", gate_proc is not None)
            metrics.tick()

            # ---- gating, in its own process: the learner never waits on it.
            if gate_proc is None and gate_out is None and step - last_gate >= gate_every:
                last_gate = step
                h, _d = save_model(model, store, config_hash=run.config_hash, extra={"kind": "candidate", "step": step, **stamp})
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
                    server = _start_server(store.path_of(best["checkpoint"]), args.port, best_json, os.path.join(run.root, "server.log"), v2)
                else:
                    run.journal("server_abandoned", game_index=games_done, exit_code=server.returncode)
                    raise SystemExit("the inference server keeps dying -- see server.log")
            starved = buffer.trainable() < int(sp["batch"])  # nothing more will ever arrive to fill a batch
            if (args.no_actors or alive == 0) and (credit < 1 or starved) and not more and gate_out is None:
                break
            if time.time() - t_last > 60:
                t_last = time.time()
                print(f"[learner] step {step} games {games_done} buffer {buffer.size} alive {alive}"
                      f"{' gate ' + str(pending['number']) + ' running' if pending else ''}", flush=True)
    except (KeyboardInterrupt, SystemExit):
        save_state(buffer.total_games + val.total_games)
        raise
    finally:
        stop_children()
        if encoder is not None:
            encoder.close()

    h, _d = save_model(model, store, config_hash=run.config_hash, optimizer=opt, extra={"kind": "learner", "step": step, "final": True, **stamp})
    save_state(buffer.total_games + val.total_games)
    run.journal("run_finished", game_index=buffer.total_games + val.total_games, step=step, final=h)
    held = validate_all()  # the comparison a replay is run for; a self-play run's last word on memorization
    run.journal("final_validation", game_index=buffer.total_games + val.total_games, step=step,
                games=val.total_games, positions=val.size, **{f"val_loss_{k}": x for k, x in held.items()})
    print(f"[learner] held-out losses over {val.size} positions: {held}", flush=True)
    metrics.flush()


if __name__ == "__main__":
    main()

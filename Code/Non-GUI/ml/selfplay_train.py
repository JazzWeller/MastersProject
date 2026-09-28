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
deck assignments, up to `gate_games` games). Promotion only on a pass;
every promoted checkpoint is kept (the frozen anchors for M9) and
journalled. The first gate is against the warm start, which is gate G4.

**Variable budgets** (M7): the run is defined by a game budget, is
resumable after a kill at any point (the learner checkpoints optimizer
state and its step; the buffer is rebuilt from the shards; actors resume
exactly the missing game indices), and every mid-run change is recorded in
the run journal with the game index.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from agent import spec
from agent.selfplay import read_frames
from agent.telemetry import Run
from keyforge.infoset import ZONE

from .arena import Player, play_paired
from .checkpoints import CheckpointStore, load_model, save_model
from .encode import Batch, collate
from .infer_server import TorchModel
from .model import KeyForgeNet, candidates_tensor

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


def _hidden_counts(batch: Batch, positions: List[dict], vocab: int, device) -> torch.Tensor:
    B = len(positions)
    out = torch.zeros(B, 2 * vocab, device=device)
    ids = batch.card_ids
    for b, p in enumerate(positions):
        for e in p["priv"]["opp_hand"]:
            out[b, ids[b, e]] += 1.0
        for e in p["priv"]["next_draws"]:
            if e >= 0:
                out[b, vocab + ids[b, e]] += 1.0
    return out


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
        single_rows, single_t = [], []
        multi = []
        for b, p in enumerate(positions):
            if p["target"] is None or p["value_only"]:
                continue
            if p["kind"] in MULTI_KINDS:
                multi.append(b)
            else:
                t = torch.zeros(batch.options.shape[1])
                t[: len(p["target"])] = torch.tensor(p["target"])
                single_rows.append(b)
                single_t.append(t)
        if single_rows:
            logits = model.policy_logits(out)[torch.tensor(single_rows, device=device)]
            pi = torch.stack(single_t).to(device)
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
            total = 0.0
            for start, n, t in spans:
                total = total - (torch.tensor(t, device=device) * torch.log_softmax(scores[start : start + n], 0)).sum()
            L["multi"] = total / len(spans)
    v = model.value(out)
    L["value"] = F.mse_loss(v, z)
    stats["value_calibration"] = float((v.detach() - z).abs().mean())
    unseen = batch.zones == OPP_UNSEEN
    if bool(unseen.any()):
        in_hand = torch.zeros_like(batch.zones, dtype=torch.float32)
        for b, p in enumerate(positions):
            for e in p["priv"]["opp_hand"]:
                in_hand[b, e] = 1.0
        L["belief"] = F.binary_cross_entropy_with_logits(model.belief_logits(out)[unseen], in_hand[unseen])
    oracle = model.oracle_value(out, _hidden_counts(batch, positions, model.vocab_size, device))
    L["oracle"] = F.mse_loss(oracle, z)
    if weights.get("distill", 0.0) > 0:
        # Suphx-style oracle guidance, the distillation route (M6): the
        # student value regresses onto the oracle's (a far less noisy
        # target than the outcome), never the other way round.
        L["distill"] = F.mse_loss(v, oracle.detach())
    total = sum(weights.get(k, 1.0) * x for k, x in L.items())
    return total, L, stats


def _start_server(ckpt_path: str, port: int, best_json: str, log_path: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "ml.infer_server", ckpt_path, "--port", str(port), "--watch", best_json],
        cwd=_NON_GUI, stdout=open(log_path, "a"), stderr=subprocess.STDOUT,
    )


def _start_actor(run: Run, worker: int, games: int, port: int, mode: str) -> subprocess.Popen:
    log = open(os.path.join(run.root, f"actor_{worker:03d}.log"), "a")
    return subprocess.Popen(
        [sys.executable, "-m", "agent.selfplay", "--run", run.run_id, "--worker", str(worker), "--games", str(games),
         "--port", str(port), "--mode", mode],
        cwd=_NON_GUI, stdout=log, stderr=subprocess.STDOUT,
    )


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
    if state is not None:
        model, _meta, opt_state = load_model(store.path_of(state["checkpoint"]), net_overrides=net_over)
        run.journal("learner_resumed", game_index=state["games_consumed"], step=state["step"])
    elif args.init:
        model, _meta, opt_state = load_model(args.init, net_overrides=net_over)
        opt_state = None
    else:
        model, opt_state = KeyForgeNet(dict(cfg["network"], dropout=0.0)), None
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(sp["lr"]), weight_decay=float(sp["weight_decay"]))
    if opt_state is not None and opt_state.get("state"):
        opt.load_state_dict({"state": opt_state["state"], "param_groups": opt_state["param_groups"]})
    step = state["step"] if state else 0
    positions_per_game = 200.0  # ~140 non-forced decisions + half as many value-only records
    total_steps = max(1, int(budget * positions_per_game / sp["positions_per_step"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=float(sp["lr_min"]))
    for _ in range(step):
        sched.step()

    if not os.path.exists(best_json):
        h, digest = save_model(model, store, config_hash=run.config_hash, extra={"kind": "warm_start", "step": 0})
        with open(best_json, "w") as f:
            json.dump({"checkpoint": h, "step": 0, "games": 0, "promotions": 0}, f)
        run.journal("promoted", game_index=0, checkpoint=h, reason="warm start")
    best = json.load(open(best_json))

    # ------------------------------------------------ processes
    procs: Dict[int, subprocess.Popen] = {}
    server = None
    workers = int(sp["workers"])
    per_worker = math.ceil(budget / workers)
    if not args.no_actors:
        server = _start_server(store.path_of(best["checkpoint"]), args.port, best_json, os.path.join(run.root, "server.log"))
        time.sleep(8)
        for w in range(workers):
            procs[w] = _start_actor(run, w, per_worker, args.port, args.mode)

    # ------------------------------------------------ learning loop
    tailer = Tailer(run.artifact_dir("shards"))
    buffer = Buffer(int(sp["buffer_positions"]), max(200, int(sp["buffer_positions"] // 150)))
    for rec in tailer.poll():  # rebuild the buffer (resume) or pick up anything already there
        buffer.add(rec)
    metrics = run.metrics("learner")
    weights = dict(sp["loss_weights"])
    if args.mode == "dmc":
        weights["q"] = 1.0
    consumed = step * int(sp["batch"])
    last_gate = step
    augment = bool(sp.get("mirror_augmentation", True))
    if augment:
        from agent.augment import mirror_safe

        if not mirror_safe(cfg["pool"]["decks"]):
            augment = False
            run.journal("mirror_augmentation_off", game_index=0, reason="a card in the pool breaks the left/right symmetry")
    t_last = time.time()
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
            total, L, st = losses_for(model, positions, device, weights, args.mode, augment, rng)
            opt.zero_grad(set_to_none=True)
            total.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float(sp["grad_clip"]))
            opt.step()
            sched.step()
            step += 1
            consumed += len(positions)
            metrics.count("gradient_steps")
            metrics.count("positions_consumed", len(positions))
            metrics.observe("grad_norm", float(gn))
            for k, v in L.items():
                metrics.observe(f"loss_{k}", float(v.detach()))
            for k, v in st.items():
                metrics.observe(k, v)
            metrics.gauge("lr", sched.get_last_lr()[0])
            metrics.gauge("buffer_positions", buffer.size)
            metrics.gauge("games", games_done)
            metrics.gauge("step", step)
        else:
            time.sleep(0.5)
        metrics.tick()

        if step - last_gate >= int(sp["gate_every_steps"]):
            last_gate = step
            h, _d = save_model(model, store, config_hash=run.config_hash, optimizer=opt, extra={"kind": "learner", "step": step})
            with open(state_json, "w") as f:
                json.dump({"checkpoint": h, "step": step, "games_consumed": games_done, "offsets": tailer.offsets}, f)
            verdict, rep = gate(model, store, best, cfg, args, device)
            metrics.gauge("gate_score", rep.score)
            run.journal("gate", game_index=games_done, step=step, candidate=h, best=best["checkpoint"], verdict=verdict,
                        score=round(rep.score, 4), games=rep.games)
            if verdict == "H1":
                best = {"checkpoint": h, "step": step, "games": games_done, "promotions": best.get("promotions", 0) + 1}
                with open(best_json + ".tmp", "w") as f:
                    json.dump(best, f)
                os.replace(best_json + ".tmp", best_json)
                run.journal("promoted", game_index=games_done, checkpoint=h, step=step)
                if best["promotions"] == 1:
                    run.journal("gate_G4", game_index=games_done, verdict="PASS", step=step)

        # Supervise actors: restart a crashed one (journalled); finish when
        # every actor is done and the buffer has been fully consumed.
        alive = 0
        for w, p in list(procs.items()):
            code = p.poll()
            if code is None:
                alive += 1
            elif code != 0:
                run.journal("actor_restarted", game_index=games_done, worker=w, exit_code=code)
                procs[w] = _start_actor(run, w, per_worker, args.port, args.mode)
                alive += 1
        if (args.no_actors or alive == 0) and allowed <= 0 and not new:
            break
        if time.time() - t_last > 60:
            t_last = time.time()
            print(f"[learner] step {step} games {games_done} buffer {buffer.size} alive {alive}", flush=True)

    h, _d = save_model(model, store, config_hash=run.config_hash, optimizer=opt, extra={"kind": "learner", "step": step, "final": True})
    with open(state_json, "w") as f:
        json.dump({"checkpoint": h, "step": step, "games_consumed": buffer.total_games, "offsets": tailer.offsets}, f)
    run.journal("run_finished", game_index=buffer.total_games, step=step, final=h)
    metrics.flush()
    if server is not None:
        server.terminate()


def gate(model: KeyForgeNet, store: CheckpointStore, best: dict, cfg: dict, args, device) -> Tuple[Optional[str], object]:
    """SPRT of the current network against the best so far, both searching
    at `--gate-sims` (search mode) or both search-free (DMC)."""
    sp, se = cfg["selfplay"], cfg["search"]
    best_net, _m, _o = load_model(store.path_of(best["checkpoint"]))
    cand = TorchModel(model.eval(), str(device))
    incumbent = TorchModel(best_net, str(device))
    kind = "net" if args.mode == "dmc" else "search"
    mode = "q" if args.mode == "dmc" else "policy"
    a = Player("candidate", kind, cand, regime=se["regime"], leaf=se["leaf"], simulations=args.gate_sims, resample=se["resample"], mode=mode)
    b = Player("best", kind, incumbent, regime=se["regime"], leaf=se["leaf"], simulations=args.gate_sims, resample=se["resample"], mode=mode)
    first = 9_000_000 + best.get("promotions", 0) * 10_000 + int(time.time()) % 10_000
    rep = play_paired(a, b, range(first, first + max(1, int(sp["gate_games"]) // 4)), cfg["pool"]["decks"][0], cfg["pool"]["decks"][1],
                      max_turns=200, concurrency=64, sprt=GATE_SPRT, min_games=40)
    model.train()
    return gate_verdict(rep), rep


GATE_SPRT = (0.0, 35.0)  # elo: "no better" vs "~55% against the best"


def gate_verdict(rep) -> Optional[str]:
    """The SPRT's verdict if it decided within the game budget; otherwise a
    fixed-N test at the budget -- promote only if the score clears 0.5 by
    two standard errors. Returns "H1" (promote), "H0" or None."""
    v = rep.sprt(*GATE_SPRT)
    if v is not None:
        return v
    if rep.games and rep.score - 0.5 > 2 * rep.standard_error:
        return "H1"
    return None


if __name__ == "__main__":
    main()

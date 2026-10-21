"""Self-play with network v2 (Agent Observation Plan, after O10): the
learner's side.

**Records stay compact.** A v2 actor (`agent.selfplay`,
`observation.version` 2) records, per searched decision, only what the search
produced -- the visit distribution (or the multi-select candidates and their
distribution), whether the search was full, the seat, the decision's index in
the game's choice record -- plus the game's config and choice record. A v2
encoding with its history and privileged labels is ~12 KB a position; 50,000
games of them would be ~120 GB of shards.

**The learner encodes by replaying.** `encode_selfplay_game` replays a record
and encodes each recorded position with exactly the code that built the BC
corpus (`ml.dataset_v2`: `encode_v2`, the history, the privileged labels), so
self-play and Tier 0b positions are the same inputs. Encoding runs in a small
process pool (`selfplay.encode_workers`), off the training loop.

**Targets.** `pi`: the root visit distribution over the options (single
choice), or over `cands` (multi-select: the candidates the search scored,
each an index form). A position with no policy target (a cheap search, a
value-only record, a multi-select past the enumeration cap) trains value,
belief and the oracle only.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from keyforge.enums import DecisionKind

from .dataset import KIND_INDEX, MULTI_KINDS, SOURCE_CODES
from .dataset_v2 import (_append_encoding, _append_folded, _append_labels, _append_privileged, _arr, _Cols,
                         assemble, cols_arrays)
from .encode_v2 import BatchV2

MULTI_KIND_IDS = set(MULTI_KINDS)
ORDER_ID = KIND_INDEX[DecisionKind.ORDER_EFFECTS]
SOURCE_SELFPLAY = SOURCE_CODES["selfplay"]
# The learner's loss weights (`selfplay.loss_weights` overrides): v1's, plus
# BC v2's next-draw term; "multi" is the multi-select policy.
DEFAULT_WEIGHTS = {"policy": 1.0, "multi": 1.0, "value": 1.0, "belief": 0.25, "draw": 0.25, "oracle": 0.25,
                   "distill": 0.0}


# ------------------------------------------------------------------ encoding
def encode_selfplay_game(record: dict) -> dict:
    """One self-play record -> its positions as a shard's arrays (what
    `assemble` reads), plus `pi_off`/`pi` (each position's target
    distribution, empty if none) and `cands` (each position's multi-select
    candidates, or None)."""
    from agent.history import N_RF, N_RI, history_for
    from keyforge.game import Game
    from keyforge.replay import config_from_dict

    game = Game(config_from_dict(record["config"]))
    c = _Cols()
    outcome = record["outcome"]
    by_decision: Dict[int, List[dict]] = {}
    for p in record["positions"]:
        by_decision.setdefault(int(p["decision"]), []).append(p)
    uid = int(record["meta"]["seed"]) & ((1 << 62) - 1)
    stream_of = {1: 0, 2: 1}
    pis: List[np.ndarray] = []
    cands: List[Optional[list]] = []
    choices = record["choice_record"]
    for n in range(len(choices) + 1):
        for p in by_decision.get(n, ()):
            d = game.pending_decision
            pid = int(p["seat"])
            value_only = bool(p["value_only"])
            if not value_only and d.player != pid:
                raise ValueError(f"record {record['meta']}: decision {n} is player {d.player}'s, recorded for {pid}")
            h = history_for(game, pid)
            _append_encoding(c, encode_v2_of(game, pid))
            _append_folded(c, h, game.turn_number)
            _append_privileged(c, game, pid)
            c.chosen.append(np.zeros(0, dtype=np.int8))
            has_target = p["target"] is not None and not value_only
            _append_labels(c, kind=KIND_INDEX[d.kind], n_opt=0 if value_only else len(d.options),
                           min_n=min(d.min_n, 255), max_n=min(d.max_n, 255), target=-1, forced=not has_target,
                           z=outcome[pid], turn=game.turn_number, seat=pid, game=uid, source=SOURCE_SELFPLAY,
                           decision_index=n, value_only=value_only, hist_stream=stream_of[pid], hist_n=h.n)
            pis.append(np.asarray(p["target"] if has_target else [], dtype=np.float32))
            cands.append([list(x) for x in p["candidates"]] if has_target and p.get("candidates") else None)
        if n < len(choices):
            game.submit_index(choices[n])
    for pid in (1, 2):
        h = history_for(game, pid)
        c.streams.append((_arr(h.ints, np.int64).reshape(h.n, N_RI).astype(np.int32),
                          _arr(h.floats, np.float32).reshape(h.n, N_RF),
                          _arr(h.pointers, np.int64).reshape(-1, 3).astype(np.int32)))
    if not c.lab["kind"]:
        return {"n": 0}
    a = cols_arrays(c)
    off = np.zeros(len(pis) + 1, dtype=np.int64)
    np.cumsum([len(x) for x in pis], out=off[1:])
    a["pi_off"] = off
    a["pi"] = np.concatenate(pis) if off[-1] else np.zeros(0, np.float32)
    a["cands"] = cands
    a["n"] = len(cands)
    return a


def encode_v2_of(game, pid):
    from agent.features_v2 import encode_v2

    return encode_v2(game, pid)


def encode_task(record: dict) -> dict:
    """Pool entry point: (game index in the run, encoded arrays)."""
    return encode_selfplay_game(record)


# ------------------------------------------------------------------- buffer
class BufferV2:
    """The most recent `max_games` games' encoded positions, sampled
    uniformly (v1's `Buffer`, over encoded games), with v1's reuse limits
    (`UseCounts`): a spent policy group empties the row's target, a spent
    outcome group clears it in `TargetsV2.value_on`."""

    def __init__(self, max_positions: int, max_games: int, history: str, limits=None):
        self.games: deque = deque()
        self.max_positions = max_positions
        self.max_games = max_games
        self.history = history
        self.limits = limits if limits is not None and limits.active else None
        self.size = 0
        self.total_games = 0
        self.total_positions = 0

    def add(self, arrays: dict) -> int:
        n = int(arrays["n"])
        self.total_games += 1
        self.total_positions += n
        if n:
            self.games.append(arrays)
            if self.limits:
                self.limits.add(np.diff(arrays["pi_off"]) > 0)
            self.size += n
        while self.games and (len(self.games) > self.max_games or self.size > self.max_positions):
            self.size -= int(self.games.popleft()["n"])
            if self.limits:
                self.limits.drop_oldest()
        return n

    def trainable(self) -> int:
        return self.limits.trainable() if self.limits else self.size

    def sample(self, n: int, rng: random.Random):
        """A batch of `n` positions (fewer if the buffer holds fewer):
        (BatchV2, TargetsV2, pi per row, candidates per row)."""
        if self.limits:
            by_game: Dict[int, List[Tuple[int, bool, bool]]] = {}
            for gi, j, pol, val in self.limits.pick(n, rng):
                by_game.setdefault(gi, []).append((j, pol, val))
        else:
            sizes = [int(g["n"]) for g in self.games]
            cum = np.cumsum(sizes)
            picks = sorted(rng.sample(range(int(cum[-1])), min(n, int(cum[-1]))))
            by_game = {}
            for k in picks:
                gi = int(np.searchsorted(cum, k, side="right"))
                by_game.setdefault(gi, []).append((k - (int(cum[gi - 1]) if gi else 0), True, True))
        return self._rows([(self.games[gi], rows) for gi, rows in by_game.items()],
                          with_value_on=self.limits is not None)

    def batches(self, n: int):
        """Every position in buffer order, `n` at a time (full validation)."""
        flat = [(g, j) for g in self.games for j in range(int(g["n"]))]
        for a in range(0, len(flat), n):
            by_game: Dict[int, Tuple[dict, list]] = {}
            for g, j in flat[a:a + n]:
                by_game.setdefault(id(g), (g, []))[1].append((j, True, True))
            yield self._rows(list(by_game.values()), with_value_on=False)

    def _rows(self, games, with_value_on: bool):
        """`games`: (a game's arrays, [(position, policy on, value on)])."""
        groups, pis, cands, value_on, row = [], [], [], [], 0
        empty = np.zeros(0, np.float32)
        for g, rows in games:
            local = np.asarray([j for j, _p, _v in rows], dtype=np.int64)
            groups.append((np.arange(row, row + len(local)), g, local))
            for j, pol, val in rows:
                pis.append(g["pi"][g["pi_off"][j]:g["pi_off"][j + 1]] if pol else empty)
                cands.append(g["cands"][j] if pol else None)
                value_on.append(val)
            row += len(local)
        batch, tg = assemble(groups, row, self.history)
        if with_value_on and not all(value_on):
            tg.value_on = torch.tensor(value_on, dtype=torch.bool)
        return batch, tg, pis, cands


# ------------------------------------------------------------------- losses
def target_rows(tg, pis: Sequence[np.ndarray], cands: Sequence[Optional[list]]):
    """(single-choice rows, multi-select rows) with a policy target."""
    kinds = tg.kind.tolist()
    single, multi = [], []
    for b, (pi, cand) in enumerate(zip(pis, cands)):
        if not len(pi):
            continue
        if kinds[b] in MULTI_KIND_IDS:
            if cand:
                multi.append(b)
        else:
            single.append(b)
    return single, multi


def loss_counts(batch: BatchV2, tg, pis, cands, heads: Sequence[str]) -> Dict[str, int]:
    """Each loss's denominator in `losses_v2` (for training in pieces)."""
    from .bc_train_v2 import _belief_targets

    single, multi = target_rows(tg, pis, cands)
    zone_rows, _cls, draw_rows, _next = _belief_targets(batch, tg)
    outcome = batch.size if tg.value_on is None else int(tg.value_on.sum())
    n = {"policy": len(single), "multi": len(multi), "value": outcome, "belief": int(zone_rows.sum()),
         "draw": int(draw_rows.sum()), "oracle": outcome, "distill": outcome if "oracle" in heads else 0}
    return {k: v for k, v in n.items() if k in heads}


def losses_v2(model, batch: BatchV2, tg, pis, cands, weights: dict, heads: Sequence[str]):
    """The self-play loss on v2 inputs: soft cross-entropy to the search's
    visit distribution (options, or multi-select candidates), value MSE to
    the outcome, and BC v2's belief / next-draw / oracle terms."""
    from .bc_train_v2 import _belief_targets, oracle_counts
    from .model import candidates_tensor

    dev = tg.z.device
    out = model.encode_state(batch)
    L: Dict[str, torch.Tensor] = {}
    stats: Dict[str, float] = {}
    single, multi = target_rows(tg, pis, cands)
    if "policy" in heads and single:
        logits = model.policy_logits(out)[torch.tensor(single, device=dev)].float()
        K = logits.shape[1]
        pi_np = np.zeros((len(single), K), dtype=np.float32)
        for r, b in enumerate(single):
            pi_np[r, :len(pis[b])] = pis[b]
        pi = torch.from_numpy(pi_np).to(dev)
        logp = torch.log_softmax(logits, dim=-1).masked_fill(pi == 0, 0.0)
        L["policy"] = -(pi * logp).sum(-1).mean()
        probs = torch.softmax(logits.detach(), -1)
        stats["policy_entropy"] = float(-(probs * torch.log(probs.clamp_min(1e-9))).sum(-1).mean())
    if "multi" in heads and multi:
        kinds = tg.kind.tolist()
        rows, members, ordered, spans = [], [], [], []
        for b in multi:
            spans.append((len(members), len(cands[b]), pis[b]))
            rows.extend([b] * len(cands[b]))
            members.extend(tuple(x) for x in cands[b])
            ordered.extend([kinds[b] == ORDER_ID] * len(cands[b]))
        scores = model.subset_scores(out, torch.tensor(rows, device=dev), candidates_tensor(members, dev),
                                     torch.tensor(ordered, device=dev)).float()
        width = max(n for _s, n, _t in spans)
        padded = torch.full((len(spans), width), float("-inf"), device=dev)
        tgt = np.zeros((len(spans), width), dtype=np.float32)
        r_idx, c_idx, s_idx = [], [], []
        for r, (start, n, t) in enumerate(spans):
            tgt[r, :n] = t
            r_idx.extend([r] * n)
            c_idx.extend(range(n))
            s_idx.extend(range(start, start + n))
        padded[torch.tensor(r_idx, device=dev), torch.tensor(c_idx, device=dev)] = scores[torch.tensor(s_idx, device=dev)]
        t = torch.from_numpy(tgt).to(dev)
        L["multi"] = -(t * torch.log_softmax(padded, -1).masked_fill(t == 0, 0.0)).sum(-1).mean()
    on = None if tg.value_on is None else tg.value_on.nonzero().squeeze(-1)
    rows = (lambda x: x) if on is None else (lambda x: x[on])
    outcome = on is None or bool(on.numel())
    v = rows(model.value(out).float())
    z = rows(tg.z)
    if "value" in heads and outcome:
        L["value"] = F.mse_loss(v, z)
        stats["value_calibration"] = float((v.detach() - z).abs().mean())
    if "belief" in heads or "draw" in heads:
        zones, draw = model.belief_v2(out, batch)
        zone_rows, cls, draw_rows, is_next = _belief_targets(batch, tg)
        if "belief" in heads and bool(zone_rows.any()):
            L["belief"] = F.cross_entropy(zones[zone_rows].float(), cls[zone_rows])
        if "draw" in heads and bool(draw_rows.any()):
            L["draw"] = F.binary_cross_entropy_with_logits(draw[draw_rows].float(), is_next[draw_rows])
    if "oracle" in heads and outcome:
        oracle = rows(model.oracle_value(out, oracle_counts(batch, tg)).float())
        L["oracle"] = F.mse_loss(oracle, z)
        if "distill" in heads:
            L["distill"] = F.mse_loss(v, oracle.detach())
    total = sum(weights.get(k, 1.0) * x for k, x in L.items())
    return total, L, stats


HEADS = ("policy", "multi", "value", "belief", "draw", "oracle", "distill")


def active_heads(weights: dict) -> List[str]:
    """The losses with a positive weight (distillation needs the oracle)."""
    on = [h for h in HEADS if weights.get(h, 0.0) > 0]
    return [h for h in on if h != "distill" or "oracle" in on]


def backward_step_v2(model, sample, *, weights: dict, max_tokens: Optional[int], autocast):
    """Forward and backward for one sampled batch, in pieces of at most
    `max_tokens` tokens with the whole batch's gradient (as BC v2)."""
    from .accumulate import backward_accumulated
    from .bc_train_v2 import pieces_of

    batch, tg, pis, cands = sample
    heads = active_heads(weights)
    spans = pieces_of(batch, max_tokens)
    if len(spans) == 1:
        with autocast():
            total, L, st = losses_v2(model, batch, tg, pis, cands, weights, heads)
        total.backward()
        return L, st
    pieces = [(batch.slice(a, b), tg.slice(a, b), pis[a:b], cands[a:b]) for a, b in spans]
    return backward_accumulated(
        pieces, lambda p: loss_counts(*p, heads),
        lambda p: losses_v2(model, *p, weights, heads), weights, autocast,
        stat_counts={"policy_entropy": "policy", "value_calibration": "value"})


# -------------------------------------------------------------- encoding pool
class EncodingPool:
    """Encodes arriving self-play records in `workers` processes (0: in this
    process), handing them back in arrival order. `pending` is truthy while
    any record is still being encoded."""

    def __init__(self, workers: int):
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        self.ex = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) if workers > 0 else None
        self.pending: deque = deque()

    def submit(self, records) -> None:
        for r in records:
            info = {"meta": r["meta"], "outcome": r["outcome"]}
            self.pending.append((self.ex.submit(encode_task, r) if self.ex else encode_task(r), info))

    def ready(self):
        """The encoded records at the front of the queue that are done."""
        out = []
        while self.pending:
            job, info = self.pending[0]
            if self.ex is not None and not job.done():
                break
            self.pending.popleft()
            out.append((job.result() if self.ex is not None else job, info))
        return out

    def wait_all(self):
        out = []
        while self.pending:
            job, info = self.pending.popleft()
            out.append((job.result() if self.ex is not None else job, info))
        return out

    def close(self) -> None:
        if self.ex is not None:
            self.ex.shutdown(wait=False, cancel_futures=True)

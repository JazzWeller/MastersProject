"""Encoded training shards (Agent Training Plan, Milestone M3; reused by M6).

`encode_records(record_path, out_dir)` replays every game in one
`sim.bc_corpus` record shard and writes, per decision:

- the compact encoding (`agent.features.encode` of the decider's InfoSet),
  as ragged numpy arrays (in-play rows and option rows stored only where
  they exist), float16 for the float blocks;
- the teacher label (single index, or the chosen list for CHOOSE_CARDS /
  the permutation for ORDER_EFFECTS), kind, `min_n`/`max_n`, and whether
  the decision was forced (one legal submission);
- the value target (the game outcome from the decider's seat) and the turn;
- the **privileged** labels (M6, never an input): which entities are in the
  opponent's hand right now, and the decider's next five draws.

Replay records stay the source of truth; these shards are a cache,
regenerated whenever `FEATURE_VERSION` changes (each shard is stamped).

`Corpus(dirs)` memory-maps shards and assembles `ml.encode.Batch` batches
plus targets; `split_of(game_uid)` is the by-game 90/5/5 split (never by
decision -- consecutive decisions within a game are heavily correlated).
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from agent import spec
from agent.features import encode
from bots.option_space import choice_space_size
from keyforge.enums import DecisionKind
from keyforge.game import Game
from keyforge.infoset import build_infoset
from keyforge.replay import config_from_dict

from .encode import Batch

KINDS = list(spec.DECISION_KINDS)
KIND_INDEX = {k: i for i, k in enumerate(KINDS)}
MULTI_KINDS = (KIND_INDEX[DecisionKind.CHOOSE_CARDS], KIND_INDEX[DecisionKind.ORDER_EFFECTS])
SOURCE_CODES = {"heuristic": 0, "epsilon": 1, "random": 2, "selfplay": 3}
NEXT_DRAWS = 5
G, O, P, N = spec.GLOBAL.width, spec.OPTION.width, spec.INPLAY.width, spec.N_ENTITIES

_ARRAYS = (
    "card_ids", "zones", "flags", "globals", "inplay_off", "inplay_idx", "inplay", "opt_off", "options", "pointers",
    "kind", "n_opt", "min_n", "max_n", "target", "chosen_off", "chosen", "forced", "z", "turn", "seat", "game",
    "source", "decision_index", "priv_opp_hand", "priv_next_draws",
)


def _game_uid(rec: dict) -> int:
    h = hashlib.sha256(f"{rec['source']}|{rec['seed']}|{rec.get('index')}".encode("utf-8")).digest()
    return int.from_bytes(h[:6], "big")


def split_of(game_uid: int, fractions=(0.9, 0.05, 0.05)) -> int:
    """0 = train, 1 = validation, 2 = test -- by game, deterministically."""
    u = (game_uid % 1_000_003) / 1_000_003
    if u < fractions[0]:
        return 0
    if u < fractions[0] + fractions[1]:
        return 1
    return 2


def _is_forced(d) -> bool:
    if d.kind in (DecisionKind.CHOOSE_CARDS, DecisionKind.ORDER_EFFECTS):
        return choice_space_size(d) <= 1
    return len(d.options) <= 1


def encode_game(rec: dict, cols: Dict[str, list]) -> int:
    """Appends every decision of one record to `cols`. Returns the count."""
    config = config_from_dict(rec["config"])
    game = Game(config)
    labels = rec.get("labels")
    record = rec["record"]
    uid = _game_uid(rec)
    src = SOURCE_CODES[rec["source"]]
    outcome = rec["outcome"]
    n_done = 0
    for n, enc_choice in enumerate(record):
        d = game.pending_decision
        pid = d.player
        info = build_infoset(game, pid)
        e = encode(info)
        label = labels[n] if labels is not None else enc_choice
        kind = KIND_INDEX[d.kind]
        k = len(d.options)
        cols["card_ids"].append(np.frombuffer(e.card_ids.tobytes(), dtype=np.int16))
        cols["zones"].append(np.frombuffer(e.zones, dtype=np.uint8))
        cols["flags"].append(np.frombuffer(e.flags, dtype=np.uint8))
        cols["globals"].append(np.frombuffer(e.globals.tobytes(), dtype=np.float32).astype(np.float16))
        cols["inplay_idx"].append(np.frombuffer(e.inplay_index, dtype=np.uint8))
        cols["inplay"].append(np.frombuffer(e.inplay.tobytes(), dtype=np.float32).reshape(-1, P).astype(np.float16))
        cols["options"].append(np.frombuffer(e.options.tobytes(), dtype=np.float32).reshape(k, O).astype(np.float16))
        cols["pointers"].append(np.frombuffer(e.pointers.tobytes(), dtype=np.int8))
        cols["kind"].append(kind)
        cols["n_opt"].append(k)
        cols["min_n"].append(min(d.min_n, 255))
        cols["max_n"].append(min(d.max_n, 255))
        if isinstance(label, list):
            cols["target"].append(-1)
            cols["chosen"].append(np.asarray(label, dtype=np.int8))
        else:
            cols["target"].append(int(label))
            cols["chosen"].append(np.zeros(0, dtype=np.int8))
        cols["forced"].append(_is_forced(d))
        cols["z"].append(outcome[str(pid)])
        cols["turn"].append(game.turn_number)
        cols["seat"].append(pid)
        cols["game"].append(uid)
        cols["source"].append(src)
        cols["decision_index"].append(n)
        opp = game.players[3 - pid]
        hand = np.zeros(N, dtype=np.uint8)
        for c in opp.hand.cards():
            i = info.index_of_iid.get(c.instance_id)
            if i is not None:
                hand[i] = 1
        cols["priv_opp_hand"].append(hand)
        draws = np.full(NEXT_DRAWS, -1, dtype=np.int8)
        for j, c in enumerate(game.players[pid].deck.cards()[:NEXT_DRAWS]):
            draws[j] = info.index_of_iid.get(c.instance_id, -1)
        cols["priv_next_draws"].append(draws)
        game.submit_index(enc_choice)
        n_done += 1
    return n_done


def _ragged(parts: List[np.ndarray], dtype, width: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    counts = np.array([len(p) for p in parts], dtype=np.int64)
    off = np.zeros(len(parts) + 1, dtype=np.int64)
    np.cumsum(counts, out=off[1:])
    if width is None:
        flat = np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype=dtype)
    else:
        flat = np.concatenate(parts).astype(dtype) if parts and off[-1] else np.zeros((0, width), dtype=dtype)
    return off, flat


def write_shard(out_dir: str, cols: Dict[str, list], meta: dict) -> None:
    tmp = out_dir + ".tmp"
    os.makedirs(tmp, exist_ok=True)
    arrays = {
        "card_ids": np.stack(cols["card_ids"]).astype(np.int16),
        "zones": np.stack(cols["zones"]).astype(np.uint8),
        "flags": np.stack(cols["flags"]).astype(np.uint8),
        "globals": np.stack(cols["globals"]).astype(np.float16),
        "kind": np.asarray(cols["kind"], dtype=np.uint8),
        "n_opt": np.asarray(cols["n_opt"], dtype=np.uint8),
        "min_n": np.asarray(cols["min_n"], dtype=np.uint8),
        "max_n": np.asarray(cols["max_n"], dtype=np.uint8),
        "target": np.asarray(cols["target"], dtype=np.int16),
        "forced": np.asarray(cols["forced"], dtype=np.bool_),
        "z": np.asarray(cols["z"], dtype=np.int8),
        "turn": np.asarray(cols["turn"], dtype=np.int16),
        "seat": np.asarray(cols["seat"], dtype=np.uint8),
        "game": np.asarray(cols["game"], dtype=np.int64),
        "source": np.asarray(cols["source"], dtype=np.uint8),
        "decision_index": np.asarray(cols["decision_index"], dtype=np.int16),
        "priv_opp_hand": np.stack(cols["priv_opp_hand"]).astype(np.uint8),
        "priv_next_draws": np.stack(cols["priv_next_draws"]).astype(np.int8),
    }
    arrays["inplay_off"], arrays["inplay_idx"] = _ragged(cols["inplay_idx"], np.uint8)
    _, arrays["inplay"] = _ragged(cols["inplay"], np.float16, P)
    arrays["opt_off"], arrays["options"] = _ragged(cols["options"], np.float16, O)
    _, arrays["pointers"] = _ragged(cols["pointers"], np.int8)
    arrays["chosen_off"], arrays["chosen"] = _ragged(cols["chosen"], np.int8)
    for name, arr in arrays.items():
        np.save(os.path.join(tmp, name + ".npy"), arr)
    with open(os.path.join(tmp, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(dict(meta, records=int(len(arrays["kind"])), stamp=spec.stamp()), f, sort_keys=True)
    if os.path.exists(out_dir):
        import shutil

        shutil.rmtree(out_dir)
    os.replace(tmp, out_dir)


def encode_records(record_path: str, out_dir: str) -> Tuple[str, int]:
    from sim.bc_corpus import read_records

    if os.path.exists(os.path.join(out_dir, "meta.json")):
        with open(os.path.join(out_dir, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        try:
            spec.check_stamp(meta["stamp"], out_dir, allow_minor=False)
            return out_dir, 0
        except spec.FeatureVersionMismatch:
            pass  # stale encoding: redo it
    cols: Dict[str, list] = {name: [] for name in _ARRAYS}
    games = 0
    for rec in read_records(record_path):
        encode_game(rec, cols)
        games += 1
    write_shard(out_dir, cols, {"records_file": os.path.basename(record_path), "games": games})
    return out_dir, len(cols["kind"])


def _encode_task(args):
    return encode_records(*args)


def encode_corpus(record_dir: str, out_dir: str, workers: int = 5) -> List[str]:
    """Encodes every `*.jsonl` record shard in `record_dir` into a shard
    directory under `out_dir` (skipping ones already encoded at this
    feature version)."""
    from sim import data_root

    record_dir = data_root.resolve(record_dir)
    out_dir = data_root.resolve(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    tasks = []
    for name in sorted(os.listdir(record_dir)):
        if name.endswith(".jsonl"):
            tasks.append((os.path.join(record_dir, name), os.path.join(out_dir, name[: -len(".jsonl")])))
    if workers <= 1:
        results = [encode_records(*t) for t in tasks]
    else:
        with mp.get_context("spawn").Pool(workers) as p:
            results = p.map(_encode_task, tasks, chunksize=1)
    return [r[0] for r in results]


# ------------------------------------------------------------------ loading
class Shard:
    def __init__(self, path: str, mmap: bool = True):
        self.path = path
        with open(os.path.join(path, "meta.json"), "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        spec.check_stamp(self.meta["stamp"], path)
        mode = "r" if mmap else None
        self.a = {name: np.load(os.path.join(path, name + ".npy"), mmap_mode=mode) for name in _ARRAYS}
        self.size = int(self.meta["records"])


@dataclass
class Targets:
    kind: torch.Tensor  # [B] long
    target: torch.Tensor  # [B] long, -1 for multi-select
    forced: torch.Tensor  # [B] bool
    z: torch.Tensor  # [B] float
    turn: torch.Tensor  # [B] long
    source: torch.Tensor  # [B] long
    min_n: torch.Tensor
    max_n: torch.Tensor
    n_opt: torch.Tensor
    chosen: List[List[int]]  # per row; [] for single-choice
    opp_hand: torch.Tensor  # [B, 72] float (privileged)
    next_draws: torch.Tensor  # [B, 5] long, -1 padded (privileged)
    game: np.ndarray


class Corpus:
    """Every record of a set of shard directories, addressed by a global
    index. `indices(split=..., sources=...)` selects; `batch(idx)` loads."""

    def __init__(self, shard_dirs: Sequence[str], mmap: bool = True):
        self.shards = [Shard(p, mmap) for p in shard_dirs]
        sizes = np.array([s.size for s in self.shards], dtype=np.int64)
        self.offsets = np.zeros(len(self.shards) + 1, dtype=np.int64)
        np.cumsum(sizes, out=self.offsets[1:])
        self.size = int(self.offsets[-1])
        self._game = np.concatenate([s.a["game"] for s in self.shards]) if self.shards else np.zeros(0, np.int64)
        self._source = np.concatenate([s.a["source"] for s in self.shards]) if self.shards else np.zeros(0, np.uint8)
        self._kind = np.concatenate([s.a["kind"] for s in self.shards]) if self.shards else np.zeros(0, np.uint8)
        self._forced = np.concatenate([s.a["forced"] for s in self.shards]) if self.shards else np.zeros(0, bool)

    @classmethod
    def from_dir(cls, root: str, mmap: bool = True) -> "Corpus":
        from sim import data_root

        root = data_root.resolve(root)
        dirs = sorted(os.path.join(root, d) for d in os.listdir(root) if os.path.exists(os.path.join(root, d, "meta.json")))
        return cls(dirs, mmap)

    def indices(self, split: Optional[int] = None, sources: Optional[Sequence[str]] = None, fractions=(0.9, 0.05, 0.05)) -> np.ndarray:
        mask = np.ones(self.size, dtype=bool)
        if split is not None:
            u = (self._game % 1_000_003) / 1_000_003
            if split == 0:
                mask &= u < fractions[0]
            elif split == 1:
                mask &= (u >= fractions[0]) & (u < fractions[0] + fractions[1])
            else:
                mask &= u >= fractions[0] + fractions[1]
        if sources is not None:
            codes = [SOURCE_CODES[s] for s in sources]
            mask &= np.isin(self._source, codes)
        return np.nonzero(mask)[0]

    def kinds(self) -> np.ndarray:
        return self._kind

    def forced(self) -> np.ndarray:
        return self._forced

    def batch(self, idx: np.ndarray, device=None) -> Tuple[Batch, Targets]:
        idx = np.asarray(idx, dtype=np.int64)
        B = len(idx)
        shard_of = np.searchsorted(self.offsets, idx, side="right") - 1
        card_ids = np.empty((B, N), np.int64)
        zones = np.empty((B, N), np.int64)
        flags = np.empty((B, N), np.uint8)
        globals_ = np.empty((B, G), np.float32)
        inplay = np.zeros((B, N, P), np.float32)
        n_opt = np.empty(B, np.int64)
        opt_rows, ptr_rows = [], []
        kind = np.empty(B, np.int64)
        target = np.empty(B, np.int64)
        forced = np.empty(B, bool)
        z = np.empty(B, np.float32)
        turn = np.empty(B, np.int64)
        source = np.empty(B, np.int64)
        min_n = np.empty(B, np.int64)
        max_n = np.empty(B, np.int64)
        opp_hand = np.empty((B, N), np.float32)
        next_draws = np.empty((B, 5), np.int64)
        game = np.empty(B, np.int64)
        chosen: List[List[int]] = []
        for b in range(B):
            s = self.shards[shard_of[b]]
            a = s.a
            i = int(idx[b] - self.offsets[shard_of[b]])
            card_ids[b] = a["card_ids"][i]
            zones[b] = a["zones"][i]
            flags[b] = a["flags"][i]
            globals_[b] = a["globals"][i]
            lo, hi = a["inplay_off"][i], a["inplay_off"][i + 1]
            if hi > lo:
                inplay[b, a["inplay_idx"][lo:hi]] = a["inplay"][lo:hi]
            lo, hi = a["opt_off"][i], a["opt_off"][i + 1]
            n_opt[b] = hi - lo
            opt_rows.append(a["options"][lo:hi])
            ptr_rows.append(a["pointers"][lo:hi])
            kind[b] = a["kind"][i]
            target[b] = a["target"][i]
            forced[b] = a["forced"][i]
            z[b] = a["z"][i]
            turn[b] = a["turn"][i]
            source[b] = a["source"][i]
            min_n[b] = a["min_n"][i]
            max_n[b] = a["max_n"][i]
            opp_hand[b] = a["priv_opp_hand"][i]
            next_draws[b] = a["priv_next_draws"][i]
            game[b] = a["game"][i]
            lo, hi = a["chosen_off"][i], a["chosen_off"][i + 1]
            chosen.append([int(x) for x in a["chosen"][lo:hi]])
        K = max(int(n_opt.max()) if B else 1, 1)
        options = np.zeros((B, K, O), np.float32)
        pointers = np.full((B, K), -1, np.int64)
        mask = np.zeros((B, K), bool)
        for b in range(B):
            k = n_opt[b]
            if k:
                options[b, :k] = opt_rows[b]
                pointers[b, :k] = ptr_rows[b]
                mask[b, :k] = True
        t = torch.from_numpy
        batch = Batch(
            card_ids=t(card_ids), zones=t(zones), flags=t(flags), inplay=t(inplay), selected=torch.zeros(B, N),
            globals=t(globals_), options=t(options), pointers=t(pointers), option_mask=t(mask),
        )
        targets = Targets(
            kind=t(kind), target=t(target), forced=t(forced), z=t(z), turn=t(turn), source=t(source),
            min_n=t(min_n), max_n=t(max_n), n_opt=t(n_opt), chosen=chosen, opp_hand=t(opp_hand),
            next_draws=t(next_draws), game=game,
        )
        if device is not None:
            batch = batch.to(device)
            for f in ("kind", "target", "forced", "z", "turn", "source", "min_n", "max_n", "n_opt", "opp_hand", "next_draws"):
                setattr(targets, f, getattr(targets, f).to(device, non_blocking=True))
        return batch, targets

    def iterate(self, idx: np.ndarray, batch_size: int, *, shuffle: bool, seed: int = 0, device=None) -> Iterator[Tuple[Batch, Targets]]:
        idx = np.array(idx)
        if shuffle:
            np.random.default_rng(seed).shuffle(idx)
        for start in range(0, len(idx), batch_size):
            yield self.batch(np.sort(idx[start : start + batch_size]), device)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Encode a sim.bc_corpus record directory into training shards.")
    parser.add_argument("records")
    parser.add_argument("out")
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    t0 = time.perf_counter()
    paths = encode_corpus(args.records, args.out, args.workers)
    corpus = Corpus(paths)
    print(f"{len(paths)} shards, {corpus.size} decisions in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()

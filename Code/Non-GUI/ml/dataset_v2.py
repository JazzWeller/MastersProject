"""v2 training shards (Agent Observation Plan, Milestone O9), `DATASET_VERSION`
4. v1's `ml/dataset.py` is untouched; replay records stay the source of truth
for both, so the same games give A0 (v1) and every v2 rung.

`encode_records_v2(record_path, out_dir)` replays every game of one
`sim.bc_corpus` record shard and writes, per decision (and, as v1 does, a
value-only record from the other seat for half of them):

- the v2 encoding (`agent.features_v2.encode_v2`): each token block's rows,
  ragged, ints in the narrowest type their columns allow (the entity block's
  zone masks and trigger bits, which can exceed 16 bits, in a side array) and
  floats as float16;
- the event history by reference: each game's projected stream is stored
  once per seat (O6's rows and pointers) and a record keeps a cursor into
  it, so storage grows with events per game, not events x positions;
- the folded history (turn tokens and summaries, `HistoryFolded`), for the
  `turn_tokens` / `summary` architectures;
- v1's labels: the teacher's choice, the kind and its bounds, forced, the
  outcome from the record's seat, the turn;
- the **privileged** labels (never an input): every entity's true zone
  class (hand / archive / deck / elsewhere) for belief v2, the opponent's
  deck order (their next draw is its top), and the viewer's next five
  draws, for the oracle.

`CorpusV2(dirs)` memory-maps shards and assembles `ml.encode_v2.BatchV2`
batches plus `TargetsV2`; `history` picks what a batch carries ("rows",
"folded", "none") and `history_dropout` drops event rows at random in
training (`bc.history_dropout`, default 0). Its split is v1's: by game.
"""

from __future__ import annotations

import json
import mmap
import multiprocessing as mp
import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from agent import spec_v2 as S
from agent.features_v2 import encode_v2
from agent.history import N_RF, N_RI, SUMMARY_ENTITY, SUMMARY_GLOBAL, TURN_SCALARS, HistoryFolded, history_for
from keyforge.game import Game
from keyforge.replay import config_from_dict

from .dataset import KIND_INDEX, NEXT_DRAWS, SOURCE_CODES, VALUE_ONLY_FRACTION, _game_uid, _is_forced, _keep_value_only
from .encode_v2 import BatchV2, TokenBlock

DATASET_VERSION = 4
N = S.N_ENTITIES
OPP_DECK = 40  # the opponent's deck order, top first (-1 padded); decks hold at most 36
ZONE_CLASS = {"hand": 1, "archive": 2, "deck": 3}  # 0: anywhere else
# Entity int columns that can pass int16: the zone masks (35 bits) and the
# trigger bit set (16 bits). Every other entity int column is checked to fit.
WIDE = ("mask", "opp_mask", "extra_triggers")
_WIDE_IDX = [S.ENTITY.i[c] for c in WIDE]
_NARROW_IDX = [j for j in range(S.ENTITY.n_int) if j not in _WIDE_IDX]
RAGGED = [b for b in S.BLOCKS if b.name not in ("global", "entity")]
_INDEX = ("game", "source", "kind", "forced", "hist_n")
_LABELS = ("kind", "n_opt", "min_n", "max_n", "target", "forced", "z", "turn", "seat", "game", "source",
           "decision_index", "value_only", "hist_stream", "hist_n")


# ------------------------------------------------------------------ encoding
class _Cols:
    def __init__(self):
        self.lab: Dict[str, list] = {k: [] for k in _LABELS}
        self.chosen: List[np.ndarray] = []
        self.glob_i: List[np.ndarray] = []
        self.glob_f: List[np.ndarray] = []
        self.ent_i: List[np.ndarray] = []
        self.ent_w: List[np.ndarray] = []
        self.ent_f: List[np.ndarray] = []
        self.blocks: Dict[str, Tuple[list, list]] = {b.name: ([], []) for b in RAGGED}
        self.f_ti: List[np.ndarray] = []
        self.f_tf: List[np.ndarray] = []
        self.f_ts: List[np.ndarray] = []
        self.f_se: List[np.ndarray] = []
        self.f_sg: List[np.ndarray] = []
        self.priv_zone: List[np.ndarray] = []
        self.priv_opp_deck: List[np.ndarray] = []
        self.priv_my_draws: List[np.ndarray] = []
        self.streams: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []  # per (game, seat)


def _arr(a, dtype):
    return np.frombuffer(a.tobytes(), dtype=dtype)


def _append_encoding(c: _Cols, e) -> None:
    g = e.blocks["global"]
    c.glob_i.append(_arr(g.ints, np.int64).astype(np.int32))
    c.glob_f.append(_arr(g.floats, np.float32).astype(np.float16))
    ent = e.blocks["entity"]
    ints = _arr(ent.ints, np.int64).reshape(ent.n, S.ENTITY.n_int)
    narrow = ints[:, _NARROW_IDX]
    if narrow.size and (narrow.min() < -(1 << 15) or narrow.max() >= (1 << 15)):
        bad = [S.ENTITY.int_names[_NARROW_IDX[j]] for j in np.nonzero((narrow < -(1 << 15)) | (narrow >= (1 << 15)))[1]]
        raise ValueError(f"entity columns {sorted(set(bad))} pass int16: add them to WIDE")
    c.ent_i.append(narrow.astype(np.int16))
    c.ent_w.append(ints[:, _WIDE_IDX].copy())
    c.ent_f.append(_arr(ent.floats, np.float32).reshape(ent.n, S.ENTITY.n_float).astype(np.float16))
    for b in RAGGED:
        blk = e.blocks[b.name]
        ints, floats = c.blocks[b.name]
        ints.append(_arr(blk.ints, np.int64).reshape(blk.n, b.n_int).astype(np.int32))
        floats.append(_arr(blk.floats, np.float32).reshape(blk.n, b.n_float).astype(np.float16))


def _append_folded(c: _Cols, h, turn: int) -> None:
    f = HistoryFolded.of(h, turn)
    c.f_ti.append(_arr(f.turn_ints, np.int64).reshape(f.u, 3).astype(np.int32))
    c.f_tf.append(_arr(f.turn_floats, np.float32).reshape(f.u, len(TURN_SCALARS)).astype(np.float16))
    c.f_ts.append(_arr(f.turn_sets, np.int64).reshape(-1, 3).astype(np.int16))
    c.f_se.append(_arr(f.summary_entity, np.float32).reshape(N, len(SUMMARY_ENTITY)).astype(np.float16))
    c.f_sg.append(_arr(f.summary_global, np.float32).astype(np.float16))


def _append_privileged(c: _Cols, game, viewer: int) -> None:
    me, them = game.players[viewer], game.players[3 - viewer]
    order = [x.instance_id for x in me.all_cards] + [x.instance_id for x in them.all_cards]
    index = {iid: i for i, iid in enumerate(order)}
    zone = np.zeros(N, dtype=np.int8)
    for p in (me, them):
        for name, cards in (("hand", p.hand.cards()), ("archive", p.archive.cards()), ("deck", p.deck.cards())):
            for x in cards:
                i = index.get(x.instance_id)
                if i is not None:
                    zone[i] = ZONE_CLASS[name]
    c.priv_zone.append(zone)
    deck = np.full(OPP_DECK, -1, dtype=np.int8)
    for j, x in enumerate(them.deck.cards()[:OPP_DECK]):
        deck[j] = index.get(x.instance_id, -1)
    c.priv_opp_deck.append(deck)
    draws = np.full(NEXT_DRAWS, -1, dtype=np.int8)
    for j, x in enumerate(me.deck.cards()[:NEXT_DRAWS]):
        draws[j] = index.get(x.instance_id, -1)
    c.priv_my_draws.append(draws)


def _append_labels(c: _Cols, **v) -> None:
    for k in _LABELS:
        c.lab[k].append(v[k])


def encode_game_v2(rec: dict, c: _Cols, *, both_seats: bool = True,
                   value_only_fraction: float = VALUE_ONLY_FRACTION) -> int:
    """Appends every decision of one record (and its value-only records) to
    `c`, and each seat's history stream once. Returns the record count."""
    game = Game(config_from_dict(rec["config"]))
    labels = rec.get("labels")
    uid = _game_uid(rec)
    src = SOURCE_CODES[rec["source"]]
    outcome = rec["outcome"]
    stream_of = {1: len(c.streams), 2: len(c.streams) + 1}
    n_done = 0

    def one(pid: int, value_only: bool, d, n: int, label):
        h = history_for(game, pid)
        _append_encoding(c, encode_v2(game, pid))
        _append_folded(c, h, game.turn_number)
        _append_privileged(c, game, pid)
        if value_only:
            target, chosen = -1, np.zeros(0, dtype=np.int8)
        elif isinstance(label, list):
            target, chosen = -1, np.asarray(label, dtype=np.int8)
        else:
            target, chosen = int(label), np.zeros(0, dtype=np.int8)
        c.chosen.append(chosen)
        _append_labels(c, kind=KIND_INDEX[d.kind], n_opt=0 if value_only else len(d.options), min_n=min(d.min_n, 255),
                       max_n=min(d.max_n, 255), target=target, forced=True if value_only else _is_forced(d),
                       z=outcome[str(pid)], turn=game.turn_number, seat=pid, game=uid, source=src, decision_index=n,
                       value_only=value_only, hist_stream=stream_of[pid], hist_n=h.n)

    for n, enc_choice in enumerate(rec["record"]):
        d = game.pending_decision
        pid = d.player
        one(pid, False, d, n, labels[n] if labels is not None else enc_choice)
        n_done += 1
        if both_seats and _keep_value_only(uid, n, value_only_fraction):
            one(3 - pid, True, d, n, None)
            n_done += 1
        game.submit_index(enc_choice)
    for pid in (1, 2):  # the whole game's stream; records point into it
        h = history_for(game, pid)
        c.streams.append((_arr(h.ints, np.int64).reshape(h.n, N_RI).astype(np.int32),
                          _arr(h.floats, np.float32).reshape(h.n, N_RF),
                          _arr(h.pointers, np.int64).reshape(-1, 3).astype(np.int32)))
    return n_done


def _ragged(parts: List[np.ndarray], dtype, width: int) -> Tuple[np.ndarray, np.ndarray]:
    counts = np.array([len(p) for p in parts], dtype=np.int64)
    off = np.zeros(len(parts) + 1, dtype=np.int64)
    np.cumsum(counts, out=off[1:])
    flat = np.concatenate(parts).astype(dtype) if parts and off[-1] else np.zeros((0, width), dtype=dtype)
    return off, flat.reshape(-1, width) if width else flat


def write_shard_v2(out_dir: str, c: _Cols, meta: dict) -> None:
    tmp = out_dir + ".tmp"
    os.makedirs(tmp, exist_ok=True)
    lab = c.lab
    a = {
        "kind": np.asarray(lab["kind"], np.uint8), "n_opt": np.asarray(lab["n_opt"], np.int16),
        "min_n": np.asarray(lab["min_n"], np.uint8), "max_n": np.asarray(lab["max_n"], np.uint8),
        "target": np.asarray(lab["target"], np.int16), "forced": np.asarray(lab["forced"], np.bool_),
        "z": np.asarray(lab["z"], np.int8), "turn": np.asarray(lab["turn"], np.int16),
        "seat": np.asarray(lab["seat"], np.uint8), "game": np.asarray(lab["game"], np.int64),
        "source": np.asarray(lab["source"], np.uint8), "decision_index": np.asarray(lab["decision_index"], np.int16),
        "value_only": np.asarray(lab["value_only"], np.bool_), "hist_stream": np.asarray(lab["hist_stream"], np.int32),
        "hist_n": np.asarray(lab["hist_n"], np.int32),
        "glob_i": np.stack(c.glob_i), "glob_f": np.stack(c.glob_f),
        "ent_i": np.stack(c.ent_i), "ent_w": np.stack(c.ent_w), "ent_f": np.stack(c.ent_f),
        "f_se": np.stack(c.f_se), "f_sg": np.stack(c.f_sg),
        "priv_zone": np.stack(c.priv_zone), "priv_opp_deck": np.stack(c.priv_opp_deck),
        "priv_my_draws": np.stack(c.priv_my_draws),
    }
    a["chosen_off"], a["chosen"] = _ragged(c.chosen, np.int8, 0)
    for b in RAGGED:
        ints, floats = c.blocks[b.name]
        a[f"{b.name}_off"], a[f"{b.name}_i"] = _ragged(ints, np.int32, b.n_int)
        _, a[f"{b.name}_f"] = _ragged(floats, np.float16, b.n_float)
    a["f_off"], a["f_ti"] = _ragged(c.f_ti, np.int32, 3)
    _, a["f_tf"] = _ragged(c.f_tf, np.float16, len(TURN_SCALARS))
    a["f_ts_off"], a["f_ts"] = _ragged(c.f_ts, np.int16, 3)
    a["s_off"], a["s_ints"] = _ragged([s[0] for s in c.streams], np.int32, N_RI)
    _, a["s_floats"] = _ragged([s[1] for s in c.streams], np.float32, N_RF)
    a["s_ptr_off"], a["s_ptr"] = _ragged([s[2] for s in c.streams], np.int32, 3)
    for name, arr in a.items():
        np.save(os.path.join(tmp, name + ".npy"), arr)
    with open(os.path.join(tmp, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(dict(meta, records=int(len(a["kind"])), streams=len(c.streams), stamp=S.stamp(),
                       dataset_version=DATASET_VERSION), f, sort_keys=True)
    if os.path.exists(out_dir):
        import shutil

        shutil.rmtree(out_dir)
    os.replace(tmp, out_dir)


def _current(out_dir: str) -> bool:
    try:
        with open(os.path.join(out_dir, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
    except OSError:
        return False
    return meta.get("dataset_version") == DATASET_VERSION and meta.get("stamp") == S.stamp()


def encode_records_v2(record_path: str, out_dir: str) -> Tuple[str, int]:
    from sim.bc_corpus import read_records

    if _current(out_dir):
        return out_dir, 0
    c = _Cols()
    games = 0
    for rec in read_records(record_path):
        encode_game_v2(rec, c)
        games += 1
    write_shard_v2(out_dir, c, {"records_file": os.path.basename(record_path), "games": games})
    return out_dir, len(c.lab["kind"])


def _encode_task(args):
    return encode_records_v2(*args)


def encode_corpus_v2(record_dir: str, out_dir: str, workers: int = 10) -> List[str]:
    """Every `*.jsonl` record shard in `record_dir` -> a v2 shard directory
    under `out_dir` (skipping ones already current)."""
    from sim import data_root

    record_dir = data_root.resolve(record_dir)
    out_dir = data_root.resolve(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    tasks = [(os.path.join(record_dir, n), os.path.join(out_dir, n[: -len(".jsonl")]))
             for n in sorted(os.listdir(record_dir)) if n.endswith(".jsonl")]
    if workers <= 1:
        return [encode_records_v2(*t)[0] for t in tasks]
    with mp.get_context("spawn").Pool(workers) as p:
        return [r[0] for r in p.imap_unordered(_encode_task, tasks, chunksize=1)]


# ------------------------------------------------------------------- loading
class ShardV2:
    def __init__(self, path: str, mmap: bool = True):
        self.path = path
        with open(os.path.join(path, "meta.json"), "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        if self.meta.get("dataset_version") != DATASET_VERSION:
            raise ValueError(f"{path}: dataset version {self.meta.get('dataset_version')}, need {DATASET_VERSION}")
        if self.meta.get("stamp") != S.stamp():
            raise ValueError(f"{path}: encoded with a different v2 layout or vocabularies -- re-encode it")
        self.mmap = mmap
        self.size = int(self.meta["records"])
        # the small per-record index the corpus selects on, always loaded;
        # the rest (`a`) only while the shard is in use: a memory map per
        # array of 120 shards would be ~5,400 open files
        self.index = {n: np.load(os.path.join(path, n + ".npy")) for n in _INDEX}
        self._a = None

    @property
    def a(self) -> Dict[str, np.ndarray]:
        if self._a is None:
            self._load(self.mmap)
        return self._a

    def _load(self, mmap: bool) -> None:
        mode = "r" if mmap else None
        self._a = {n[:-4]: np.load(os.path.join(self.path, n), mmap_mode=mode).view(np.ndarray)
                   for n in os.listdir(self.path) if n.endswith(".npy")}

    def materialize(self) -> None:
        """Reads the whole shard into memory, sequentially: a shuffled window
        then draws its batches from RAM. Random 12 KB reads through the
        memory map were page faults on a cold cache (seconds per batch)."""
        self._load(False)

    def release(self) -> None:
        self._a = None


@dataclass
class TargetsV2:
    kind: torch.Tensor
    target: torch.Tensor
    forced: torch.Tensor
    z: torch.Tensor
    turn: torch.Tensor
    source: torch.Tensor
    min_n: torch.Tensor
    max_n: torch.Tensor
    n_opt: torch.Tensor
    value_only: torch.Tensor
    chosen: List[List[int]]
    zone: torch.Tensor  # [B, 72] long: 0 elsewhere, 1 hand, 2 archive, 3 deck (privileged)
    opp_deck: torch.Tensor  # [B, 40] long, top first, -1 padded (privileged)
    my_draws: torch.Tensor  # [B, 5] long, -1 padded (privileged)
    game: np.ndarray

    _TENSORS = ("kind", "target", "forced", "z", "turn", "source", "min_n", "max_n", "n_opt", "value_only", "zone",
                "opp_deck", "my_draws")

    def to(self, device) -> "TargetsV2":
        for f in self._TENSORS:
            setattr(self, f, getattr(self, f).to(device, non_blocking=True))
        return self

    def slice(self, a: int, b: int) -> "TargetsV2":
        return TargetsV2(**{k: getattr(self, k)[a:b] for k in self.__dataclass_fields__})


def _gather_ragged(off: np.ndarray, flat: np.ndarray, local: np.ndarray):
    """Rows of each selected record: (flat rows, record position of each,
    slot within its record, counts)."""
    lo, hi = off[local], off[local + 1]
    cnt = (hi - lo).astype(np.int64)
    total = int(cnt.sum())
    if not total:
        return flat[:0], np.zeros(0, np.int64), np.zeros(0, np.int64), cnt
    starts = np.cumsum(cnt) - cnt
    rows = np.repeat(lo - starts, cnt) + np.arange(total)
    who = np.repeat(np.arange(len(local)), cnt)
    slot = np.arange(total) - np.repeat(starts, cnt)
    return flat[rows], who, slot, cnt


class CorpusV2:
    """Every record of a set of v2 shard directories, by global index.
    `indices(split=...)` selects (v1's by-game split); `batch(idx)` loads."""

    def __init__(self, shard_dirs: Sequence[str], mmap: bool = True, history: str = "rows",
                 history_dropout: float = 0.0):
        self.shards = [ShardV2(p, mmap) for p in shard_dirs]
        sizes = np.array([s.size for s in self.shards], dtype=np.int64)
        self.offsets = np.zeros(len(self.shards) + 1, dtype=np.int64)
        np.cumsum(sizes, out=self.offsets[1:])
        self.size = int(self.offsets[-1])
        self.history = history
        self.history_dropout = float(history_dropout)
        cat = lambda name, dt: np.concatenate([s.index[name] for s in self.shards]) if self.shards else np.zeros(0, dt)
        self._game = cat("game", np.int64)
        self._source = cat("source", np.uint8)
        self._kind = cat("kind", np.uint8)
        self._forced = cat("forced", np.bool_)
        self._hist_n = cat("hist_n", np.int32)

    @classmethod
    def from_dir(cls, root: str, **kw) -> "CorpusV2":
        from sim import data_root

        root = data_root.resolve(root)
        dirs = sorted(os.path.join(root, d) for d in os.listdir(root) if os.path.exists(os.path.join(root, d, "meta.json")))
        return cls(dirs, **kw)

    def indices(self, split: Optional[int] = None, sources: Optional[Sequence[str]] = None,
                fractions=(0.9, 0.05, 0.05)) -> np.ndarray:
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
            mask &= np.isin(self._source, [SOURCE_CODES[s] for s in sources])
        return np.nonzero(mask)[0]

    def history_lengths(self, idx: np.ndarray) -> np.ndarray:
        return self._hist_n[idx]

    def batch(self, idx: np.ndarray, *, train: bool = False, rng: Optional[np.random.Generator] = None
              ) -> Tuple[BatchV2, TargetsV2]:
        idx = np.asarray(idx, dtype=np.int64)
        shard_of = np.searchsorted(self.offsets, idx, side="right") - 1
        groups = []
        for s_id in np.unique(shard_of):
            rows = np.nonzero(shard_of == s_id)[0]
            groups.append((rows, self.shards[s_id].a, idx[rows] - self.offsets[s_id]))
        return assemble(groups, len(idx), self.history, self.history_dropout if train else 0.0, rng)

    def _history_rows(self, hist, B: int, train: bool, rng):
        return _history_rows(hist, B, self.history_dropout if train else 0.0, rng)


    def iterate(self, idx: np.ndarray, batch_size: int, *, shuffle: bool, seed: int = 0, train: bool = False,
                window_shards: int = 6) -> Iterator[Tuple[BatchV2, TargetsV2]]:
        """v1's windowed shuffle, each window read into memory first (a v2
        shard is ~670 MB: 6 at a time, ~4 GB); within a window, batches are
        drawn from records of similar history length (bucketed, then the
        buckets shuffled), so a batch needs little padding. Unshuffled
        iteration goes shard by shard, each read into memory in turn."""
        idx = np.array(idx)
        rng = np.random.default_rng(seed)
        shard_of = np.searchsorted(self.offsets, idx, side="right") - 1
        if not shuffle:
            for s_id in np.unique(shard_of):
                sel = idx[shard_of == s_id]
                self.shards[s_id].materialize()
                try:
                    for start in range(0, len(sel), batch_size):
                        yield self.batch(sel[start:start + batch_size])
                finally:
                    self.shards[s_id].release()
            return
        order = rng.permutation(len(self.shards))
        windows = [order[w:w + window_shards] for w in range(0, len(order), window_shards)]

        def read(window):
            for s_id in window:
                self.shards[s_id].materialize()

        # Read the next window while this one's batches are drawn (a cold
        # window takes ~45 s from disk; two windows are ~8 GB in memory).
        ahead = threading.Thread(target=read, args=(windows[0],), daemon=True) if windows else None
        if ahead is not None:
            ahead.start()
        try:
            for wi, window in enumerate(windows):
                ahead.join()
                if wi + 1 < len(windows):
                    ahead = threading.Thread(target=read, args=(windows[wi + 1],), daemon=True)
                    ahead.start()
                sel = idx[np.isin(shard_of, window)]
                try:
                    rng.shuffle(sel)
                    if self.history == "rows":
                        sel = sel[np.argsort(self._hist_n[sel], kind="stable")]
                    batches = [sel[k:k + batch_size] for k in range(0, len(sel), batch_size)]
                    for j in rng.permutation(len(batches)):
                        yield self.batch(np.sort(batches[j]), train=train, rng=rng)
                finally:
                    for s_id in window:
                        self.shards[s_id].release()
        finally:  # stopped early: the window being read ahead is freed too
            if ahead is not None:
                ahead.join()
            for s in self.shards:
                s.release()


def assemble(groups, B: int, history: str, dropout: float = 0.0, rng=None) -> Tuple[BatchV2, TargetsV2]:
    """A batch from `groups`: (batch rows, a shard's arrays, those records'
    indices in it). `history` is what the batch carries ("rows", "folded",
    "none"); `dropout` drops event rows (training only)."""
    fixed_names = ("kind", "target", "forced", "z", "turn", "source", "min_n", "max_n", "n_opt", "value_only", "game",
                   "glob_i", "glob_f", "ent_i", "ent_w", "ent_f", "priv_zone", "priv_opp_deck", "priv_my_draws",
                   "f_se", "f_sg", "hist_stream", "hist_n")
    fixed = {n: [] for n in fixed_names}
    ragged = {b.name: [] for b in RAGGED}
    turns, hist, chosen = [], [], [[] for _ in range(B)]
    for rows, a, local in groups:
        for n in fixed_names:
            fixed[n].append((rows, a[n][local]))
        for b in RAGGED:
            ints, who, slot, cnt = _gather_ragged(a[f"{b.name}_off"], a[f"{b.name}_i"], local)
            floats, _w, _s, _c = _gather_ragged(a[f"{b.name}_off"], a[f"{b.name}_f"], local)
            ragged[b.name].append((rows, ints, floats, who, slot, cnt))
        if history == "folded":
            ti, who, slot, cnt = _gather_ragged(a["f_off"], a["f_ti"], local)
            tf, _w, _s, _c = _gather_ragged(a["f_off"], a["f_tf"], local)
            ts, tw, _ts, _tc = _gather_ragged(a["f_ts_off"], a["f_ts"], local)
            turns.append((rows, ti, tf, who, slot, cnt, ts, tw))
        if history == "rows":
            hist.append((rows, a, a["hist_stream"][local], a["hist_n"][local]))
        lo, hi = a["chosen_off"][local], a["chosen_off"][local + 1]
        for j in np.nonzero(hi > lo)[0]:
            chosen[rows[j]] = a["chosen"][lo[j]:hi[j]].tolist()
    col = {}
    for n, parts in fixed.items():
        first = parts[0][1]
        out = np.empty((B,) + first.shape[1:], dtype=first.dtype)
        for rows, vals in parts:
            out[rows] = vals
        col[n] = out
    t = torch.from_numpy
    blocks = {}
    blocks["global"] = TokenBlock(t(col["glob_i"].astype(np.int64)[:, None, :]),
                                  t(col["glob_f"].astype(np.float32)[:, None, :]), torch.ones(B, 1, dtype=torch.bool))
    ent = np.zeros((B, N, S.ENTITY.n_int), np.int64)
    ent[:, :, _NARROW_IDX] = col["ent_i"]
    ent[:, :, _WIDE_IDX] = col["ent_w"]
    blocks["entity"] = TokenBlock(t(ent), t(col["ent_f"].astype(np.float32)), torch.ones(B, N, dtype=torch.bool))
    for b in RAGGED:
        R = max(1, max((int(c.max()) if len(c) else 0) for *_x, c in ragged[b.name]))
        ints = np.zeros((B, R, max(b.n_int, 1)), np.int64)
        floats = np.zeros((B, R, max(b.n_float, 1)), np.float32)
        mask = np.zeros((B, R), bool)
        for rows, vi, vf, who, slot, _cnt in ragged[b.name]:
            if len(who):
                ints[rows[who], slot, :b.n_int] = vi
                floats[rows[who], slot, :b.n_float] = vf
                mask[rows[who], slot] = True
        blocks[b.name] = TokenBlock(t(ints), t(floats), t(mask))
    hist_ints, hist_floats, hist_mask, hist_ptr = _history_rows(hist, B, dropout, rng)
    U = max([1] + [int(p[5].max()) for p in turns if len(p[5])])
    turn_ints = np.zeros((B, U, 3), np.int64)
    turn_floats = np.zeros((B, U, len(TURN_SCALARS)), np.float32)
    turn_sets = np.zeros((B, U, 3, N), np.float32)
    turn_mask = np.zeros((B, U), bool)
    for rows, ti, tf, who, slot, _cnt, ts, tw in turns:
        if len(who):
            turn_ints[rows[who], slot] = ti
            turn_floats[rows[who], slot] = tf
            turn_mask[rows[who], slot] = True
        if len(tw):
            turn_sets[rows[tw], ts[:, 0], ts[:, 1], ts[:, 2]] = 1.0
    summary_entity = col["f_se"].astype(np.float32) if history == "folded" else np.zeros((B, N, len(SUMMARY_ENTITY)), np.float32)
    summary_global = col["f_sg"].astype(np.float32) if history == "folded" else np.zeros((B, SUMMARY_GLOBAL), np.float32)
    batch = BatchV2(blocks=blocks, hist_ints=t(hist_ints), hist_floats=t(hist_floats), hist_mask=t(hist_mask),
                    hist_ptr=t(hist_ptr), turn_ints=t(turn_ints), turn_floats=t(turn_floats), turn_sets=t(turn_sets),
                    turn_mask=t(turn_mask), summary_entity=t(summary_entity), summary_global=t(summary_global),
                    turn_now=t(col["turn"].astype(np.int64)))
    tg = TargetsV2(
        kind=t(col["kind"].astype(np.int64)), target=t(col["target"].astype(np.int64)), forced=t(col["forced"]),
        z=t(col["z"].astype(np.float32)), turn=t(col["turn"].astype(np.int64)), source=t(col["source"].astype(np.int64)),
        min_n=t(col["min_n"].astype(np.int64)), max_n=t(col["max_n"].astype(np.int64)),
        n_opt=t(col["n_opt"].astype(np.int64)), value_only=t(col["value_only"]), chosen=chosen,
        zone=t(col["priv_zone"].astype(np.int64)), opp_deck=t(col["priv_opp_deck"].astype(np.int64)),
        my_draws=t(col["priv_my_draws"].astype(np.int64)), game=col["game"])
    return batch, tg

def _history_rows(hist, B: int, dropout: float, rng):
    """Each record's rows: its seat's stream up to its cursor. In training,
    `dropout` drops each row with that probability (pointer rows renumbered
    to the rows kept)."""
    per = [None] * B
    for rows, a, streams, ns in hist:
        for r, s, n in zip(rows, streams, ns):
            lo = int(a["s_off"][s])
            ints = a["s_ints"][lo:lo + n]
            floats = a["s_floats"][lo:lo + n]
            plo, phi = int(a["s_ptr_off"][s]), int(a["s_ptr_off"][s + 1])
            ptr = a["s_ptr"][plo:phi]
            ptr = ptr[ptr[:, 0] < n]
            if dropout > 0 and n:
                keep = (rng or np.random.default_rng()).random(n) >= dropout
                new_row = np.cumsum(keep) - 1
                ints, floats = ints[keep], floats[keep]
                ptr = ptr[keep[ptr[:, 0]]]
                ptr = np.column_stack([new_row[ptr[:, 0]], ptr[:, 1], ptr[:, 2]])
            per[r] = (ints, floats, ptr)
    H = max([1] + [len(p[0]) for p in per if p is not None])
    Q = max([1] + [len(p[2]) for p in per if p is not None])
    hi = np.zeros((B, H, N_RI), np.int64)
    hf = np.zeros((B, H, N_RF), np.float32)
    hm = np.zeros((B, H), bool)
    hp = np.full((B, Q, 3), -1, np.int64)
    for r, p in enumerate(per):
        if p is None:
            continue
        ints, floats, ptr = p
        k = len(ints)
        hi[r, :k], hf[r, :k], hm[r, :k] = ints, floats, True
        hp[r, :len(ptr)] = ptr
    return hi, hf, hm, hp


# --------------------------------------------------------------- packing
# A shuffle that spreads each game's positions (O9, Tier 0b): a shuffle
# window of whole shards puts a game's ~220 positions a few steps apart, and
# inputs that identify a game (turn tokens, summaries, the event history)
# then let the network memorize each game's outcome while the window lasts
# (training value MSE 0.16-0.18 against 0.59 log-loss held out). A packed
# corpus stores each shard as compressed chunks of `PACK_CHUNK` consecutive
# records (18x smaller: zeros dominate), held in memory, and training draws
# from segments of chunks taken at random across the whole corpus.

PACK_CHUNK = 16
_PER_RECORD = ("kind", "n_opt", "min_n", "max_n", "target", "forced", "z", "turn", "seat", "game", "source",
               "decision_index", "value_only", "hist_stream", "hist_n", "glob_i", "glob_f", "ent_i", "ent_w", "ent_f",
               "f_se", "f_sg", "priv_zone", "priv_opp_deck", "priv_my_draws")
_RAGGED_KEYS = [("chosen_off", ("chosen",)), ("f_off", ("f_ti", "f_tf")), ("f_ts_off", ("f_ts",))] + \
    [(f"{b.name}_off", (f"{b.name}_i", f"{b.name}_f")) for b in RAGGED]
SPLIT_FRACTIONS = (0.9, 0.05, 0.05)


def split_code(game: np.ndarray) -> np.ndarray:
    """0 / 1 / 2 (train / validation / test) by game, as `CorpusV2.indices`."""
    u = (np.asarray(game, dtype=np.int64) % 1_000_003) / 1_000_003
    return np.where(u < SPLIT_FRACTIONS[0], 0, np.where(u < SPLIT_FRACTIONS[0] + SPLIT_FRACTIONS[1], 1, 2))


def _chunk(a: dict, lo: int, hi: int) -> Tuple[dict, dict]:
    """Records [lo, hi) of a shard: (core arrays, the history streams they
    need, each cut at the furthest cursor any of them has)."""
    core = {k: np.ascontiguousarray(a[k][lo:hi]) for k in _PER_RECORD}
    for off_key, flats in _RAGGED_KEYS:
        o = a[off_key][lo:hi + 1]
        core[off_key] = (o - o[0]).astype(np.int64)
        for f in flats:
            core[f] = np.ascontiguousarray(a[f][o[0]:o[-1]])
    used, local = np.unique(core["hist_stream"], return_inverse=True)
    core["hist_stream"] = local.astype(np.int32)
    ints, floats, ptrs, s_off, p_off = [], [], [], [0], [0]
    for j, u in enumerate(used):
        n = int(core["hist_n"][local == j].max())
        lo_r = int(a["s_off"][u])
        ints.append(a["s_ints"][lo_r:lo_r + n])
        floats.append(a["s_floats"][lo_r:lo_r + n])
        p = a["s_ptr"][int(a["s_ptr_off"][u]):int(a["s_ptr_off"][u + 1])]
        p = p[p[:, 0] < n]
        ptrs.append(p)
        s_off.append(s_off[-1] + n)
        p_off.append(p_off[-1] + len(p))
    streams = {"s_off": np.asarray(s_off, np.int64), "s_ints": np.concatenate(ints).astype(np.int32),
               "s_floats": np.concatenate(floats).astype(np.float32), "s_ptr_off": np.asarray(p_off, np.int64),
               "s_ptr": np.concatenate(ptrs).astype(np.int32).reshape(-1, 3)}
    return core, streams


def pack_shard(shard_dir: str, out_stem: str, chunk: int = PACK_CHUNK, level: int = 1) -> str:
    """`<out_stem>.core` / `.str` (compressed chunks, back to back) and
    `.idx.npz` (each chunk's offsets, lengths, record count and records per
    split)."""
    import pickle
    import zlib

    sh = ShardV2(shard_dir)
    sh.materialize()
    a = sh.a
    rows = []
    with open(out_stem + ".core.tmp", "wb") as fc, open(out_stem + ".str.tmp", "wb") as fs:
        for lo in range(0, sh.size, chunk):
            hi = min(lo + chunk, sh.size)
            core, streams = _chunk(a, lo, hi)
            cb = zlib.compress(pickle.dumps(core, protocol=5), level)
            sb = zlib.compress(pickle.dumps(streams, protocol=5), level)
            counts = np.bincount(split_code(core["game"]), minlength=3)
            rows.append((fc.tell(), len(cb), fs.tell(), len(sb), hi - lo, *counts))
            fc.write(cb)
            fs.write(sb)
    r = np.asarray(rows, dtype=np.int64)
    np.savez(out_stem + ".idx.tmp.npz", core_off=r[:, 0], core_len=r[:, 1], str_off=r[:, 2], str_len=r[:, 3],
             n=r[:, 4], split_counts=r[:, 5:8], stamp=json.dumps(S.stamp()), dataset_version=DATASET_VERSION)
    os.replace(out_stem + ".core.tmp", out_stem + ".core")
    os.replace(out_stem + ".str.tmp", out_stem + ".str")
    os.replace(out_stem + ".idx.tmp.npz", out_stem + ".idx.npz")
    return out_stem


def _pack_task(args):
    return pack_shard(*args)


def pack_corpus_v2(shard_root: str, out_dir: str, workers: int = 8, chunk: int = PACK_CHUNK) -> List[str]:
    from sim import data_root

    shard_root, out_dir = data_root.resolve(shard_root), data_root.resolve(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    tasks = [(os.path.join(shard_root, d), os.path.join(out_dir, d), chunk) for d in sorted(os.listdir(shard_root))
             if os.path.exists(os.path.join(shard_root, d, "meta.json"))
             and not os.path.exists(os.path.join(out_dir, d + ".idx.npz"))]
    if workers <= 1:
        return [pack_shard(*t) for t in tasks]
    with mp.get_context("spawn").Pool(workers) as p:
        return list(p.imap_unordered(_pack_task, tasks, chunksize=1))


class Selection:
    """Chunks of a packed corpus, and the split their records are filtered
    to (None = every record). `len()` is the record count."""

    def __init__(self, chunks: np.ndarray, split: Optional[int], n: int):
        self.chunks, self.split, self.n = chunks, split, n

    def __len__(self) -> int:
        return self.n

    def subsample(self, records: int, seed: int = 0) -> "Selection":
        """About `records` records: a fixed random subset of the chunks."""
        if records >= self.n:
            return self
        k = max(1, int(len(self.chunks) * records / self.n))
        pick = np.sort(np.random.default_rng(seed).choice(self.chunks, k, replace=False))
        return Selection(pick, self.split, int(records))


class PackedCorpusV2:
    """A packed v2 corpus (`pack_corpus_v2`): its compressed chunks are
    memory-mapped (~5 GiB for Tier 0's games, held in the page cache once
    read, where the kernel can evict them rather than the process running
    out of memory); the event-history streams (another ~5 GiB) are read chunk
    by chunk, on the background thread, only when `history` is "rows" -- the
    architectures that read them train at hundreds of samples a second, so a
    segment's reads keep ahead.

    `iterate` fills a segment of `segment_chunks` chunks (shuffled: drawn at
    random across the whole corpus), decompresses and merges them, and draws
    the segment's batches from it while the next segment is prepared on a
    background thread. A game's positions then arrive at most one chunk at a
    time per segment."""

    def __init__(self, pack_dir: str, history: str = "rows", history_dropout: float = 0.0, segment_chunks: int = 4096):
        from sim import data_root

        pack_dir = data_root.resolve(pack_dir)
        self.history = history
        self.history_dropout = float(history_dropout)
        self.segment_chunks = segment_chunks
        stems = sorted(os.path.join(pack_dir, n[:-len(".idx.npz")]) for n in os.listdir(pack_dir) if n.endswith(".idx.npz"))
        cols = {k: [] for k in ("core_off", "core_len", "str_off", "str_len", "n", "split_counts")}
        file_of = []
        self._core, self._str = [], []
        for f, stem in enumerate(stems):
            idx = np.load(stem + ".idx.npz")
            if int(idx["dataset_version"]) != DATASET_VERSION or json.loads(str(idx["stamp"])) != S.stamp():
                raise ValueError(f"{stem}: packed from a different v2 layout -- re-pack it")
            for k in cols:
                cols[k].append(idx[k])
            file_of.append(np.full(len(idx["n"]), f, np.int32))
            with open(stem + ".core", "rb") as fh:
                while fh.read(1 << 24):  # read through once: random chunk reads then hit the page cache
                    pass
                self._core.append(mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ))
            if history == "rows":
                self._str.append(os.open(stem + ".str", os.O_RDONLY))
        self.c = {k: np.concatenate(v) for k, v in cols.items()}
        self.c["file"] = np.concatenate(file_of)
        self.size = int(self.c["n"].sum())

    def indices(self, split: Optional[int] = None, sources=None) -> Selection:
        if sources is not None:
            raise ValueError("a packed corpus selects by split only")
        if split is None:
            return Selection(np.arange(len(self.c["n"])), None, self.size)
        counts = self.c["split_counts"][:, split]
        return Selection(np.nonzero(counts)[0], split, int(counts.sum()))

    def _decode(self, i: int):
        import pickle
        import zlib

        f = int(self.c["file"][i])
        o, n = int(self.c["core_off"][i]), int(self.c["core_len"][i])
        core = pickle.loads(zlib.decompress(self._core[f][o:o + n]))
        streams = None
        if self.history == "rows":
            o, n = int(self.c["str_off"][i]), int(self.c["str_len"][i])
            streams = pickle.loads(zlib.decompress(os.pread(self._str[f], n, o)))
        return core, streams

    def _segment(self, chunks: np.ndarray, split: Optional[int], group: int = 512) -> dict:
        """The chunks merged into one shard-like dict (records of `split`
        only), in groups, so the decoded pieces and the merged copy are never
        all in memory at once."""
        parts = [self._merge(chunks[k:k + group], split) for k in range(0, len(chunks), group)]
        parts = [p for p in parts if p]
        if len(parts) <= 1:
            return parts[0] if parts else {}
        return self._join(parts)

    def _merge(self, chunks: np.ndarray, split: Optional[int]) -> dict:
        cores, strs = [], []
        for i in chunks:
            core, streams = self._decode(int(i))
            if split is not None:
                keep = split_code(core["game"]) == split
                if not keep.all():
                    core = _take(core, np.nonzero(keep)[0])
            if len(core["kind"]):
                cores.append(core)
                strs.append(streams)
        if not cores:
            return {}
        if self.history == "rows":
            for c, st in zip(cores, strs):
                c.update(st)
        return self._join(cores)

    def _join(self, cores: List[dict]) -> dict:
        """Shard-like dicts (chunks, or merged groups) -> one."""
        seg = {k: np.concatenate([c[k] for c in cores]) for k in _PER_RECORD}
        for off_key, flats in _RAGGED_KEYS:
            seg[off_key] = _cat_offsets([c[off_key] for c in cores])
            for f in flats:
                seg[f] = np.concatenate([c[f] for c in cores])
        if self.history == "rows":
            base = np.cumsum([0] + [len(c["s_off"]) - 1 for c in cores[:-1]])
            seg["hist_stream"] = np.concatenate([c["hist_stream"] + b for c, b in zip(cores, base)]).astype(np.int32)
            seg["s_off"] = _cat_offsets([c["s_off"] for c in cores])
            seg["s_ptr_off"] = _cat_offsets([c["s_ptr_off"] for c in cores])
            for k in ("s_ints", "s_floats", "s_ptr"):
                seg[k] = np.concatenate([c[k] for c in cores])
        return seg

    def _segment_size(self, chunks: np.ndarray, split: Optional[int]) -> int:
        return int(self.c["n"][chunks].sum() if split is None else self.c["split_counts"][chunks, split].sum())

    def iterate(self, sel: Selection, batch_size: int, *, shuffle: bool, seed: int = 0, train: bool = False,
                skip: int = 0) -> Iterator[Tuple[BatchV2, TargetsV2]]:
        """`skip`: leave out the first `skip` batches (a resumed epoch, see
        `ml/resume.py`) -- the rest are the ones a full pass would yield
        (with history dropout off: dropout draws from the same generator).
        Segments that are skipped whole are never read."""
        rng = np.random.default_rng(seed)
        chunks = rng.permutation(sel.chunks) if shuffle else np.asarray(sel.chunks)
        groups = [chunks[k:k + self.segment_chunks] for k in range(0, len(chunks), self.segment_chunks)]
        dropout = self.history_dropout if train else 0.0
        box = {}
        first = 0
        while first < len(groups) and skip:  # whole segments: replay their draws, read nothing
            n = self._segment_size(groups[first], sel.split)
            nb = -(-n // batch_size)
            if nb > skip:
                break
            if shuffle and n:
                rng.permutation(n)
                rng.permutation(nb)
            skip -= nb
            first += 1

        def prepare(j):
            box[j] = self._segment(groups[j], sel.split)

        ahead = threading.Thread(target=prepare, args=(first,), daemon=True) if first < len(groups) else None
        if ahead is not None:
            ahead.start()
        for j in range(first, len(groups)):
            ahead.join()
            seg = box.pop(j)
            if j + 1 < len(groups):
                ahead = threading.Thread(target=prepare, args=(j + 1,), daemon=True)
                ahead.start()
            if not seg:
                continue
            n = len(seg["kind"])
            order = rng.permutation(n) if shuffle else np.arange(n)
            if self.history == "rows" and shuffle:
                order = order[np.argsort(seg["hist_n"][order], kind="stable")]
            batches = [order[k:k + batch_size] for k in range(0, n, batch_size)]
            for b in (rng.permutation(len(batches)) if shuffle else range(len(batches))):
                if skip:
                    skip -= 1
                    continue
                local = np.sort(batches[b])
                yield assemble([(np.arange(len(local)), seg, local)], len(local), self.history, dropout, rng)


def _take(core: dict, keep: np.ndarray) -> dict:
    """The records `keep` of a chunk's core arrays."""
    out = {k: core[k][keep] for k in _PER_RECORD}
    for off_key, flats in _RAGGED_KEYS:
        o = core[off_key]
        lo, hi = o[keep], o[keep + 1]
        cnt = hi - lo
        out[off_key] = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64)
        rows = np.concatenate([np.arange(a, b) for a, b in zip(lo, hi)]) if cnt.sum() else np.zeros(0, np.int64)
        for f in flats:
            out[f] = core[f][rows]
    return out


def _cat_offsets(offs: List[np.ndarray]) -> np.ndarray:
    out, base = [np.zeros(1, np.int64)], 0
    for o in offs:
        out.append(o[1:] + base)
        base += int(o[-1])
    return np.concatenate(out)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Encode a sim.bc_corpus record directory into v2 training shards.")
    parser.add_argument("records")
    parser.add_argument("out")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    t0 = time.perf_counter()
    paths = encode_corpus_v2(args.records, args.out, args.workers)
    corpus = CorpusV2(paths)
    print(f"{len(paths)} shards, {corpus.size} records in {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

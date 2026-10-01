"""The self-play actor (Agent Training Plan, Milestones M7 and M8).
Torch-free: it talks to the network only through an `InferenceClient`
(normally a `RemoteInferenceClient` to `ml.infer_server`), so it can fork
and could run under PyPy.

**Topology.** Each actor process runs `concurrency` games at once. Every
round, each game with a pending decision contributes a search generator
(`agent.search.core.Search.search_gen`); `run_searches` advances all of
them together, so one `predict_many` per round serves every game's leaves
-- the CPU keeps advancing some games while others wait on the GPU.

**Per decision (search mode):**
- *playout cap randomization* (KataGo): with probability
  `full_fraction`, a full search (`sims_full`) whose visit distribution is
  recorded as a policy target; otherwise a cheap one (`sims_small`),
  recorded for the value target only;
- Dirichlet root noise always; move chosen from visit counts at
  temperature `tau_high` for each seat's first `temperature_moves`
  decisions, `tau_low` after;
- forced decisions are auto-resolved and never recorded.

**Per decision (DMC mode, M8):** no search -- `NetAgent(mode="q")`,
epsilon-greedy over Q; the action taken is recorded for regression toward
the game's Monte-Carlo return.

**Records.** One frame per finished game appended to the worker's own
shard (`<run>/shards/actor_<id>.bin`): a 4-byte length plus a pickle of
{meta, positions, outcome}. Each position holds the decider's compact
encoding, the target (visit distribution, or the action taken), the
multi-select candidates if any, and the privileged labels for M6 (the
opponent's true hand, the decider's next draws) -- the learner, never an
agent, reads those. A crash loses at most the in-flight games: a
half-written last frame is detected and truncated on resume, and games
already on disk are never replayed (the game index resumes from the count).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import struct
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from keyforge.config import GameConfig
from keyforge.enums import DecisionKind, Resample
from keyforge.game import Game
from keyforge.infoset import build_infoset

from .features import encode
from .search.core import Search, SearchSettings, legal_actions, run_searches
from .search.full_game import FullGame
from .search.leaf import make_evaluator
from .search.policies import make_policy
from .search.within_turn import WithinTurn

MULTI = (DecisionKind.CHOOSE_CARDS, DecisionKind.ORDER_EFFECTS)
FRAME = struct.Struct("<I")
EXIT_SHARD_BUSY = 4  # another live process owns this worker's shard: not a crash, never restarted


@dataclass
class SelfPlaySettings:
    mode: str = "search"  # search | dmc
    regime: str = "within_turn"
    leaf: str = "student"
    rollout: str = "heuristic"
    resample: str = "all"
    quiet_leaves: bool = False
    sims_full: int = 100
    sims_small: int = 25
    full_fraction: float = 0.25
    temperature_moves: int = 15
    tau_high: float = 1.0
    tau_low: float = 0.1
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.8
    dirichlet_eps: float = 0.25
    leaves_in_flight: int = 8
    max_turns: int = 60  # training-only cap (never in evaluation)
    concurrency: int = 16
    decks: tuple = ("fignor", "igor")
    epsilon: float = 0.3  # DMC exploration: start ...
    epsilon_end: float = 0.01  # ... end ...
    epsilon_decay_games: int = 20000  # ... over this many of a worker's games
    value_only_fraction: float = 0.5  # non-decider value records, as in ml/dataset.py
    enumerate_cap: int = 1024
    # Resignation (M7): off (None) until the value head is calibrated. A
    # seat whose searched value stays below -threshold for `consecutive`
    # decisions resigns -- except in `exempt_fraction` of games, which play
    # on and record whether the resignation would have been wrong.
    resign_threshold: Optional[float] = None
    resign_consecutive: int = 3
    resign_exempt_fraction: float = 0.1

    @classmethod
    def from_config(cls, cfg: dict, mode: str = "search") -> "SelfPlaySettings":
        sp, se = cfg["selfplay"], cfg["search"]
        return cls(
            mode=mode, regime=se["regime"], leaf=se["leaf"], rollout=se["rollout"], resample=se["resample"],
            quiet_leaves=bool(se.get("quiet_leaves", False)),
            sims_full=sp["playout_cap_full"], sims_small=sp["playout_cap_small"],
            full_fraction=sp["playout_cap_full_fraction"], temperature_moves=se["temperature_moves"],
            tau_high=se["temperature_high"], tau_low=se["temperature_low"], c_puct=se["c_puct"],
            dirichlet_alpha=se["dirichlet_alpha"], dirichlet_eps=se["dirichlet_eps"],
            leaves_in_flight=se["leaves_in_flight"], max_turns=sp["training_max_turns"],
            concurrency=sp["games_per_worker"], decks=tuple(cfg["pool"]["decks"]),
            epsilon=cfg["dmc"]["epsilon_start"], epsilon_end=cfg["dmc"]["epsilon_end"],
            epsilon_decay_games=max(1, cfg["dmc"]["epsilon_decay_games"] // max(1, sp["workers"])),
            enumerate_cap=cfg["network"]["enumerate_cap"],
            resign_threshold=sp.get("resign_threshold"), resign_consecutive=int(sp.get("resign_consecutive", 3)),
            resign_exempt_fraction=float(sp.get("resign_exempt_fraction", 0.1)),
        )


def game_seed(run_seed: int, worker: int, index: int) -> int:
    digest = hashlib.sha256(f"selfplay|{run_seed}|{worker}|{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 1


# ------------------------------------------------------------------ shards
def write_frame(path: str, record: dict) -> None:
    payload = pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL)
    with open(path, "ab") as f:
        f.write(FRAME.pack(len(payload)) + payload)
        f.flush()
        os.fsync(f.fileno())


def read_frames(path: str, start: int = 0):
    """Yields (end offset, record) for every complete frame from byte
    offset `start` -- a partially written last frame is skipped (and later
    truncated by `repair_shard`)."""
    if not os.path.exists(path):
        return
    with open(path, "rb") as f:
        f.seek(start)
        at = start
        while True:
            head = f.read(4)
            if len(head) < 4:
                return
            (n,) = FRAME.unpack(head)
            body = f.read(n)
            if len(body) < n:
                return
            at += 4 + n
            yield at, pickle.loads(body)


def repair_shard(path: str) -> set:
    """Truncates a half-written trailing frame; returns the game indices
    already complete on disk. Games finish out of order (many run at once),
    so resuming skips exactly these -- never "the first N"."""
    if not os.path.exists(path):
        return set()
    good, done = 0, set()
    for end, rec in read_frames(path):
        good = end
        done.add(rec["meta"]["index"])
    if os.path.getsize(path) != good:
        with open(path, "r+b") as f:
            f.truncate(good)
    return done


# ----------------------------------------------------------------- positions
def _privileged(game, viewer: int, info) -> dict:
    opp = game.players[3 - viewer]
    hand = [info.index_of_iid[c.instance_id] for c in opp.hand.cards() if c.instance_id in info.index_of_iid]
    draws = [info.index_of_iid.get(c.instance_id, -1) for c in game.players[viewer].deck.cards()[:5]]
    return {"opp_hand": hand, "next_draws": draws}


def _position(game, pid: int, decision, *, target=None, candidates=None, full: bool, turn: int, value_only: bool = False) -> dict:
    info = build_infoset(game, pid)
    return {
        "enc": encode(info),
        "seat": pid,
        "kind": decision.kind.name,
        "min_n": decision.min_n,
        "max_n": decision.max_n,
        "target": target,  # visit distribution / action taken; None = value only
        "candidates": candidates,  # index forms, for multi-select targets
        "full": full,
        "turn": turn,
        "value_only": value_only,
        "priv": _privileged(game, pid, info),
    }


def _keep(seed: int, n: int, fraction: float) -> bool:
    if fraction >= 1:
        return True
    h = hashlib.blake2b(f"{seed}|{n}".encode(), digest_size=4).digest()
    return int.from_bytes(h, "big") / 2**32 < fraction


def _pick(visits: List[int], tau: float, rng: random.Random, q: Optional[List[float]] = None) -> int:
    if tau <= 0.05:
        qq = q or [0.0] * len(visits)
        return max(range(len(visits)), key=lambda i: (visits[i], qq[i] if visits[i] else -2.0, -i))
    w = [v ** (1.0 / tau) for v in visits]
    total = sum(w)
    if total <= 0:
        return rng.randrange(len(visits))
    r = rng.random() * total
    for i, x in enumerate(w):
        r -= x
        if r <= 0:
            return i
    return len(w) - 1


class _Cap:
    """The actor's own search capability over its own game (self-play: the
    actor owns the game, but each seat's search still only ever sees
    determinized forks from that seat's viewpoint)."""

    def __init__(self, game, viewer: int):
        self._game = game
        self.viewer = viewer

    def fork_determinized(self, rng, resample=Resample.ALL, *, backend="replay"):
        return self._game.fork_determinized(self.viewer, rng, resample=resample, backend=backend)

    def infoset(self):
        return build_infoset(self._game, self.viewer)


@dataclass
class _Slot:
    index: int
    seed: int
    game: Game
    rng: random.Random
    searches: Dict[int, Search]
    positions: List[dict] = field(default_factory=list)
    decisions: Dict[int, int] = field(default_factory=lambda: {1: 0, 2: 0})
    t0: float = field(default_factory=time.time)
    low: Dict[int, int] = field(default_factory=lambda: {1: 0, 2: 0})  # consecutive low-value decisions
    exempt: bool = False
    resigned_by: Optional[int] = None
    would_resign: Optional[int] = None  # exempt games: the first seat that would have resigned

    @property
    def over(self) -> bool:
        return self.game.is_over or self.resigned_by is not None


class Actor:
    def __init__(self, client, settings: SelfPlaySettings, *, run_seed: int = 0, worker: int = 0, metrics=None):
        self.client = client
        self.s = settings
        self.run_seed = run_seed
        self.worker = worker
        self.metrics = metrics
        self.checkpoint_tag: Optional[str] = None
        self._dmc_agent = None

    def epsilon_for(self, game_index: int) -> float:
        """DMC exploration, annealed linearly by this worker's game index
        from `epsilon` to `epsilon_end` over `epsilon_decay_games` (M8)."""
        s = self.s
        frac = min(1.0, game_index / max(1, s.epsilon_decay_games))
        return s.epsilon + (s.epsilon_end - s.epsilon) * frac

    def _new_search(self, seed: int) -> Search:
        s = self.s
        policy = make_policy(s.rollout, seed=seed, client=self.client)
        regime = WithinTurn(policy) if s.regime == "within_turn" else FullGame(policy, quiet_leaves=s.quiet_leaves)
        settings = SearchSettings(
            simulations=s.sims_full, c_puct=s.c_puct, dirichlet_alpha=s.dirichlet_alpha, dirichlet_eps=s.dirichlet_eps,
            root_noise=True, leaves_in_flight=s.leaves_in_flight, resample=Resample(s.resample),
            enumerate_cap=s.enumerate_cap, reuse=False,
        )
        return Search(regime, make_evaluator(s.leaf, seed=seed), policy, settings, seed=seed)

    def _new_slot(self, index: int) -> _Slot:
        seed = game_seed(self.run_seed, self.worker, index)
        decks = self.s.decks if index % 2 == 0 else (self.s.decks[1], self.s.decks[0])
        game = Game(GameConfig(decks=decks, seed=seed, max_turns=self.s.max_turns))
        searches = {pid: self._new_search(seed + pid) for pid in (1, 2)} if self.s.mode == "search" else {}
        slot = _Slot(index=index, seed=seed, game=game, rng=random.Random(seed ^ 0xA5), searches=searches)
        slot.exempt = random.Random(seed ^ 0x5E516).random() < self.s.resign_exempt_fraction
        return slot

    def _advance_forced(self, slot: _Slot) -> None:
        g = slot.game
        while not g.is_over:
            d = g.pending_decision
            acts = legal_actions(d, self.s.enumerate_cap)
            if acts is not None and len(acts) == 1:
                g.submit(acts[0][1])
                continue
            return

    def _finish(self, slot: _Slot) -> dict:
        g = slot.game
        if slot.resigned_by is not None:
            outcome = {slot.resigned_by: -1, 3 - slot.resigned_by: 1}
        else:
            outcome = {pid: g.outcome_for(pid) for pid in (1, 2)}
        return {
            "meta": {
                "seed": slot.seed, "index": slot.index, "worker": self.worker, "decks": [str(d) for d in g.config.decks],
                "turns": g.turn_number, "reason": (g.result or {}).get("reason"), "checkpoint": self.checkpoint_tag,
                "mode": self.s.mode, "regime": self.s.regime, "seconds": time.time() - slot.t0,
                "resigned_by": slot.resigned_by, "resign_exempt": slot.exempt, "would_resign": slot.would_resign,
            },
            "positions": slot.positions,
            "outcome": outcome,
            "choice_record": list(g.choice_record),
        }

    def play(self, indices, on_game) -> int:
        """Plays the games with these indices (each fully determined by
        `(run seed, worker, index)`), calling `on_game(record)` as each
        finishes. Returns how many were played."""
        s = self.s
        queue = list(indices)
        active: List[_Slot] = []
        played = 0
        while queue or active:
            while queue and len(active) < s.concurrency:
                slot = self._new_slot(queue.pop(0))
                self._advance_forced(slot)
                active.append(slot)
            done = [sl for sl in active if sl.over]
            for sl in done:
                active.remove(sl)
                on_game(self._finish(sl))
                played += 1
                if self.metrics is not None:
                    self.metrics.count("games")
                    self.metrics.observe("game_turns", sl.game.turn_number)
                    self.metrics.observe("game_decisions", len(sl.game.choice_record))
            if not active:
                continue
            if s.mode == "dmc":
                self._dmc_round(active)
            else:
                self._search_round(active)
            if self.metrics is not None:
                self.metrics.tick()
        return played

    def _search_round(self, active: List[_Slot]) -> None:
        s = self.s
        gens, meta = [], []
        for sl in active:
            d = sl.game.pending_decision
            pid = d.player
            full = sl.rng.random() < s.full_fraction
            search = sl.searches[pid]
            search.settings.simulations = s.sims_full if full else s.sims_small
            gens.append(search.search_gen(_Cap(sl.game, pid), d))
            meta.append((sl, d, pid, full))
        t0 = time.perf_counter()
        results = run_searches(gens, self.client)
        if self.metrics is not None:
            self.metrics.observe("round_seconds", time.perf_counter() - t0)
        for (sl, d, pid, full), res in zip(meta, results):
            n = sl.decisions[pid]
            tau = s.tau_high if n < s.temperature_moves else s.tau_low
            k = _pick(res.visits, tau, sl.rng, res.q)
            total = sum(res.visits) or 1
            target = [v / total for v in res.visits]
            multi = d.kind in MULTI
            # A multi-select space over `enumerate_cap` falls back to the
            # fixed policy (index form None): there is no search target to
            # learn, only the value.
            searched = not any(f is None for f in res.index_forms)
            turn = sl.game.turn_number
            sl.positions.append(_position(
                sl.game, pid, d, target=target if (full and searched) else None,
                candidates=list(res.index_forms) if (multi and searched) else None, full=full, turn=turn,
            ))
            if _keep(sl.seed, len(sl.game.choice_record), s.value_only_fraction):
                sl.positions.append(_position(sl.game, 3 - pid, d, full=False, turn=turn, value_only=True))
            sl.decisions[pid] = n + 1
            if s.resign_threshold is not None and self._resigns(sl, pid, res.value):
                continue
            if self.metrics is not None:
                self.metrics.count("decisions")
                self.metrics.count("simulations", res.stats["simulations"])
                self.metrics.count("forks", res.stats["forks"])
                self.metrics.observe("fork_ms", 1000 * res.stats["fork_seconds"] / max(res.stats["forks"], 1))
            sl.game.submit(res.choices[k])
            self._advance_forced(sl)

    def _resigns(self, sl: _Slot, pid: int, value: float) -> bool:
        """Tracks `pid`'s run of low searched values; True if it resigns now
        (the game is then over for the actor). Exempt games note the first
        would-be resignation and play on."""
        sl.low[pid] = sl.low[pid] + 1 if value < -self.s.resign_threshold else 0
        if sl.low[pid] < self.s.resign_consecutive:
            return False
        if sl.exempt:
            if sl.would_resign is None:
                sl.would_resign = pid
            return False
        sl.resigned_by = pid
        return True

    def _dmc_round(self, active: List[_Slot]) -> None:
        from .agents.net_agent import NetAgent

        if self._dmc_agent is None:
            self._dmc_agent = NetAgent(self.client, mode="q", epsilon=self.s.epsilon, seed=self.run_seed * 131 + self.worker)
        self._dmc_agent.epsilon = self.epsilon_for(min(sl.index for sl in active))
        reqs = []
        for sl in active:
            d = sl.game.pending_decision
            reqs.append((None, d, None, _Cap(sl.game, d.player)))
        choices = self._dmc_agent.decide_many(reqs)
        for (_v, d, _b, _c), sl, choice in zip(reqs, active, choices):
            pid = d.player
            if isinstance(choice, list):
                taken = sorted(d.options.index(c) for c in choice) if d.kind == DecisionKind.CHOOSE_CARDS else [d.options.index(c) for c in choice]
            else:
                taken = [d.options.index(choice)]
            sl.positions.append(_position(sl.game, pid, d, target=taken, full=True, turn=sl.game.turn_number))
            if self.metrics is not None:
                self.metrics.count("decisions")
            sl.game.submit(choice)
            self._advance_forced(sl)


def main():
    """One actor worker process: plays its share of a run's game budget
    into its own shard, against the run's inference server."""
    from bots.inference_client import RemoteInferenceClient

    from .lifecycle import ShardBusy, exclusive, exit_with_parent
    from .telemetry import Run

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--games", type=int, required=True, help="this worker's total game budget")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--authkey", default="keyforge")
    parser.add_argument("--mode", default="search")
    args = parser.parse_args()

    run = Run.open(args.run)
    settings = SelfPlaySettings.from_config(run.config, args.mode)
    shard = os.path.join(run.artifact_dir("shards"), f"actor_{args.worker:03d}.bin")
    try:
        lock = exclusive(shard)  # noqa: F841 -- held for the life of the process
    except ShardBusy as e:
        print(f"actor {args.worker}: {e}", flush=True)
        raise SystemExit(EXIT_SHARD_BUSY)
    exit_with_parent()
    done = repair_shard(shard)
    todo = [i for i in range(args.games) if i not in done]
    client = RemoteInferenceClient((args.host, args.port), args.authkey.encode())
    for attempt in range(120):  # the server may still be loading its model
        try:
            client.predict_many([])
            break
        except (ConnectionError, OSError):
            time.sleep(1.0)
    metrics = run.metrics(f"actor-{args.worker}")
    actor = Actor(client, settings, run_seed=int(run.config["seed"]), worker=args.worker, metrics=metrics)
    best_path = os.path.join(run.root, "best.json")

    def on_game(record):
        write_frame(shard, record)
        if os.path.exists(best_path):
            with open(best_path, "r", encoding="utf-8") as f:
                actor.checkpoint_tag = json.load(f).get("checkpoint")

    if os.path.exists(best_path):
        with open(best_path, "r", encoding="utf-8") as f:
            actor.checkpoint_tag = json.load(f).get("checkpoint")
    actor.play(todo, on_game)
    metrics.flush()


if __name__ == "__main__":
    main()

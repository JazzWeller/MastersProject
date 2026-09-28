"""Multi-process paired evaluation (Agent Training Plan, M5 gate G2, the M9
diagnostics): searching agents are CPU-bound and per-game, so a paired-seed
run (sim/paired.py) is split into seed chunks across worker processes, each
playing its chunk through the concurrent driver, and the reports summed.

Agents are named by picklable `AgentSpec`s -- a `bots.registry` name plus
constructor kwargs and a privilege level -- so each worker builds its own.
Network-backed specs (`net`, `search-*` with a network leaf) name a
checkpoint and build an in-process CPU/GPU model in the worker (torch
needed there; everything else stays torch-free).
"""

from __future__ import annotations

import math
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from keyforge.enums import PrivilegeLevel

from .paired import PairedReport, run_paired


@dataclass(frozen=True)
class AgentSpec:
    name: str  # a bots.registry name ("heuristic", "search-within-turn", ...)
    kwargs: Tuple[Tuple[str, object], ...] = ()
    privilege: Optional[str] = None  # PrivilegeLevel value; None = the registered level

    @classmethod
    def of(cls, name: str, privilege: Optional[PrivilegeLevel] = None, **kwargs) -> "AgentSpec":
        return cls(name, tuple(sorted(kwargs.items())), privilege.value if privilege else None)

    def level(self) -> PrivilegeLevel:
        from bots.registry import privilege_of

        _ensure_registered()
        return PrivilegeLevel(self.privilege) if self.privilege else privilege_of(self.name)

    def build(self, seed: int):
        from bots.registry import make_agent

        _ensure_registered()
        return make_agent(self.name, seed=seed, **dict(self.kwargs))


def _ensure_registered() -> None:
    import agent.agents.registry_entries  # noqa: F401 -- registers the search agents


def _chunk_task(args) -> dict:
    a, b, seeds, deck_x, deck_y, max_turns, run_seed = args
    t0 = time.perf_counter()
    stats = {"decisions_searched": 0, "search_seconds": 0.0, "simulations": 0}
    built = []

    def make_a(i):
        agent = a.build(seed=run_seed * 1_000_003 + seeds[0] * 17 + i)
        built.append(agent)
        return agent

    rep, _results = run_paired(
        make_a, lambda i: b.build(seed=run_seed * 7_919 + seeds[0] * 31 + i), seeds, deck_x, deck_y,
        max_turns=max_turns, concurrency=1, privilege_a=a.level(), privilege_b=b.level(), run_seed=run_seed,
    )
    for agent in built:
        for k, v in getattr(agent, "search_stats", {}).items():
            stats[k] = stats.get(k, 0) + v
    out = rep.summary()
    out.update(raw=[rep.a_wins, rep.b_wins, rep.draws, rep.forfeits, rep.games], seconds=time.perf_counter() - t0,
               turns=rep.turns, by_seat=rep.by_seat, by_deck=rep.by_deck, stats=stats)
    return out


def evaluate_parallel(
    a: AgentSpec, b: AgentSpec, seeds: Sequence[int], *, deck_x="fignor", deck_y="igor", max_turns: int = 200,
    workers: int = 8, chunk: int = 4, run_seed: int = 0,
) -> dict:
    """Plays every seed's four paired games (sim/paired.py), `chunk` seeds
    per task, over a `workers`-process pool. Returns a merged summary."""
    seeds = list(seeds)
    tasks = [(a, b, seeds[i : i + chunk], deck_x, deck_y, max_turns, run_seed) for i in range(0, len(seeds), chunk)]
    t0 = time.perf_counter()
    if workers <= 1:
        parts = [_chunk_task(t) for t in tasks]
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            parts = pool.map(_chunk_task, tasks, chunksize=1)
    rep = PairedReport(games=0, a_wins=0, b_wins=0, draws=0, forfeits=0)
    stats: Dict[str, float] = {}
    for p in parts:
        w, l, d, f, g = p["raw"]
        rep.a_wins += w
        rep.b_wins += l
        rep.draws += d
        rep.forfeits += f
        rep.games += g
        rep.turns.extend(p["turns"])
        for seat, (wins, n) in p["by_seat"].items():
            cur = rep.by_seat.setdefault(int(seat), [0, 0])
            cur[0] += wins
            cur[1] += n
        for deck, (wins, n) in p["by_deck"].items():
            cur = rep.by_deck.setdefault(deck, [0, 0])
            cur[0] += wins
            cur[1] += n
        for k, v in p["stats"].items():
            stats[k] = stats.get(k, 0) + v
    out = rep.summary()
    out.update(
        agent_a=a.name, agent_b=b.name, a_kwargs=dict(a.kwargs), seconds=round(time.perf_counter() - t0, 1),
        sprt=rep.sprt(), stats=stats,
        mean_simulations_per_search=(stats["simulations"] / stats["decisions_searched"]) if stats.get("decisions_searched") else None,
        mean_seconds_per_search=(stats["search_seconds"] / stats["decisions_searched"]) if stats.get("decisions_searched") else None,
    )
    return out


def wilson_interval(wins: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)

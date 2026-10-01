"""Paired-seed evaluation through the concurrent driver (Agent Training
Plan, Milestones M3/M5/M9): the Milestone I protocol -- both seat orders x
both deck assignments, the same seed for all four -- but run with
`sim.driver.run_games`, so a batching agent (a network behind an inference
client) sees every concurrent game's decision in one `decide_many` call.

The 57.5% first-player win rate confounds seat with deck; this design
balances both for every seed, and keyed randomness keeps the four games of
a seed dealt alike for as long as the agents' play allows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from bots.base import Controller
from keyforge.config import GameConfig
from keyforge.enums import PrivilegeLevel

from .driver import GameResult, run_games
from .rate import sprt


@dataclass
class Arrangement:
    seed: int
    a_seat: int
    decks: Tuple  # (p1 deck, p2 deck)
    a_deck: object


def arrangements(seeds: Sequence[int], deck_x, deck_y) -> List[Arrangement]:
    out = []
    for s in seeds:
        for a_seat in (1, 2):
            for a_deck, b_deck in ((deck_x, deck_y), (deck_y, deck_x)):
                decks = (a_deck, b_deck) if a_seat == 1 else (b_deck, a_deck)
                out.append(Arrangement(seed=s, a_seat=a_seat, decks=decks, a_deck=a_deck))
    return out


@dataclass
class PairedReport:
    games: int
    a_wins: int
    b_wins: int
    draws: int
    forfeits: int
    by_seat: Dict[int, List[int]] = field(default_factory=dict)  # a_seat -> [wins, games]
    by_deck: Dict[str, List[int]] = field(default_factory=dict)
    turns: List[int] = field(default_factory=list)

    @property
    def score(self) -> float:
        """A's score: wins + half the draws, over games."""
        return (self.a_wins + 0.5 * self.draws) / self.games if self.games else 0.0

    @property
    def standard_error(self) -> float:
        return 0.5 / math.sqrt(self.games) if self.games else float("inf")

    def sprt(self, elo0: float = 0.0, elo1: float = 20.0) -> Optional[str]:
        return sprt(self.a_wins, self.b_wins, self.draws, elo0=elo0, elo1=elo1)

    def summary(self) -> dict:
        return {
            "games": self.games, "a_wins": self.a_wins, "b_wins": self.b_wins, "draws": self.draws,
            "forfeits": self.forfeits, "a_score": round(self.score, 4), "standard_error": round(self.standard_error, 4),
            "by_seat": {str(k): v for k, v in self.by_seat.items()}, "by_deck": self.by_deck,
            "mean_turns": round(sum(self.turns) / len(self.turns), 2) if self.turns else None,
        }


def _label(deck) -> str:
    return deck if isinstance(deck, str) else getattr(deck, "name", str(deck))


def run_paired(
    make_a: Callable[[int], Controller],
    make_b: Callable[[int], Controller],
    seeds: Sequence[int],
    deck_x="fignor",
    deck_y="igor",
    *,
    max_turns: int = 200,
    concurrency: int = 64,
    privilege_a: PrivilegeLevel = PrivilegeLevel.OBSERVATION,
    privilege_b: PrivilegeLevel = PrivilegeLevel.OBSERVATION,
    run_seed: int = 0,
) -> Tuple[PairedReport, List[GameResult]]:
    """`make_a(game_index)`/`make_b(game_index)` build (or return a shared)
    agent for that game; a shared `BatchController` is how a network agent
    gets batched across games."""
    arr = arrangements(seeds, deck_x, deck_y)

    def make_config(i: int) -> GameConfig:
        return GameConfig(decks=arr[i].decks, seed=arr[i].seed, max_turns=max_turns)

    def make_agents(i: int) -> Dict[int, Controller]:
        a_seat = arr[i].a_seat
        return {a_seat: make_a(i), 3 - a_seat: make_b(i)}

    def privilege(i: int) -> Dict[int, PrivilegeLevel]:
        a_seat = arr[i].a_seat
        return {a_seat: privilege_a, 3 - a_seat: privilege_b}

    results = run_games(len(arr), make_config, make_agents, concurrency=concurrency, privilege=privilege, run_seed=run_seed)
    rep = PairedReport(games=0, a_wins=0, b_wins=0, draws=0, forfeits=0)
    for a, r in zip(arr, results):
        rep.games += 1
        won = r.winner == a.a_seat
        lost = r.winner is not None and not won
        rep.a_wins += won
        rep.b_wins += lost
        rep.draws += r.winner is None
        rep.forfeits += r.forfeit is not None
        seat = rep.by_seat.setdefault(a.a_seat, [0, 0])
        seat[0] += won
        seat[1] += 1
        dk = rep.by_deck.setdefault(_label(a.a_deck), [0, 0])
        dk[0] += won
        dk[1] += 1
        if r.turns is not None:
            rep.turns.append(r.turns)
    return rep, results

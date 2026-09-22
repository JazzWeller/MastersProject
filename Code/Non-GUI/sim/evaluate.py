"""Pairing matrix and duplicate evaluation (Agent Interface Plan,
Milestone I): compares two agents fairly by canceling out seat and deck
bias, which the plan's own baseline measured as large and confounded
(first player wins 57.5% of the time on Fignor/Igor, single-threaded
RandomBot both seats) -- non-negotiable for any comparison this harness
reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from bots.registry import make_agent
from keyforge.config import GameConfig
from keyforge.game import Game


@dataclass
class DuplicateResult:
    """One seed, played twice with the seat/deck assignment swapped, so a
    lucky deal can't be mistaken for one agent being stronger."""

    seed: int
    deck_a: str
    deck_b: str
    winner_when_a_is_p1: Optional[int]
    winner_when_a_is_p2: Optional[int]
    agent_a_wins: int  # 0, 1, or 2
    agent_b_wins: int
    draws: int


def _play(agent_p1: str, agent_p2: str, deck_p1, deck_p2, seed: int, max_turns: int) -> Game:
    config = GameConfig(decks=(deck_p1, deck_p2), seed=seed, max_turns=max_turns)
    game = Game(config)
    controllers = {
        1: make_agent(agent_p1, seed=seed),
        2: make_agent(agent_p2, seed=(seed or 0) + 1),
    }
    while not game.is_over:
        d = game.pending_decision
        game.submit(controllers[d.player].decide(game.view_for(d.player), d))
    return game


def duplicate_evaluate(agent_a: str, agent_b: str, deck_a, deck_b, seed: int, max_turns: int = 200) -> DuplicateResult:
    """Plays `seed` twice: once with `agent_a` as p1/`deck_a` and `agent_b`
    as p2/`deck_b`, and once with the assignment swapped. Keyed randomness
    (Milestone A) keeps the deals aligned between the two for as long as
    the agents' choices agree."""
    game1 = _play(agent_a, agent_b, deck_a, deck_b, seed, max_turns)
    game2 = _play(agent_b, agent_a, deck_b, deck_a, seed, max_turns)
    w1 = game1.result.get("winner") if game1.result else None
    w2 = game2.result.get("winner") if game2.result else None
    a_wins = (1 if w1 == 1 else 0) + (1 if w2 == 2 else 0)
    b_wins = (1 if w1 == 2 else 0) + (1 if w2 == 1 else 0)
    draws = (1 if w1 is None else 0) + (1 if w2 is None else 0)
    return DuplicateResult(
        seed=seed, deck_a=deck_a, deck_b=deck_b,
        winner_when_a_is_p1=w1, winner_when_a_is_p2=w2,
        agent_a_wins=a_wins, agent_b_wins=b_wins, draws=draws,
    )


def pairing_matrix_evaluate(
    agent_a: str, agent_b: str, deck_x, deck_y, seeds, max_turns: int = 200
) -> List[DuplicateResult]:
    """Both seat orders x both deck assignments, with paired seeds: for
    each seed, a duplicate pair with A on `deck_x` and one with A on
    `deck_y` -- 4 games per seed, covering the full matrix."""
    results = []
    for seed in seeds:
        results.append(duplicate_evaluate(agent_a, agent_b, deck_x, deck_y, seed, max_turns))
        results.append(duplicate_evaluate(agent_a, agent_b, deck_y, deck_x, seed, max_turns))
    return results


def summarize(results: List[DuplicateResult]) -> Dict[str, float]:
    total = sum(r.agent_a_wins + r.agent_b_wins + r.draws for r in results)
    a_wins = sum(r.agent_a_wins for r in results)
    b_wins = sum(r.agent_b_wins for r in results)
    draws = sum(r.draws for r in results)
    return {
        "games": total,
        "agent_a_win_rate": a_wins / total if total else 0.0,
        "agent_b_win_rate": b_wins / total if total else 0.0,
        "draw_rate": draws / total if total else 0.0,
        "standard_error": 0.5 / (total ** 0.5) if total else float("inf"),
    }

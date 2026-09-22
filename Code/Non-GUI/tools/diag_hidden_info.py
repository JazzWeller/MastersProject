#!/usr/bin/env python3
"""Hidden-information diagnostic (Agent Interface Plan, Milestone J): the
same search agent (`bots.search_bot.DeterminizedRolloutBot`) with and
without privileged observation, and with each `Resample` mode (Milestone
D), against a fixed opponent. Separates "the cost of not knowing my own
draws" from "the cost of not knowing their hand", and puts an upper bound
on what belief modelling could buy: nothing beats the privileged condition,
since that IS the true hidden state.

Run from Code/Non-GUI: `python -m tools.diag_hidden_info`
"""

from __future__ import annotations

import argparse
from typing import List, Tuple

from bots.heuristic_bot import HeuristicBot
from bots.search_bot import DeterminizedRolloutBot
from keyforge.config import GameConfig
from keyforge.enums import PrivilegeLevel, Resample
from sim.driver import derive_agent_seed, run_games

CONDITIONS: List[Tuple[str, PrivilegeLevel, Resample]] = [
    ("privileged (true hidden state)", PrivilegeLevel.PRIVILEGED, Resample.ALL),
    ("search: own deck only resampled", PrivilegeLevel.SEARCH, Resample.OWN_DECK),
    ("search: opponent hand/archive/deck resampled", PrivilegeLevel.SEARCH, Resample.OPPONENT_PRIVATE),
    ("search: everything resampled", PrivilegeLevel.SEARCH, Resample.ALL),
]


def run(n_games: int = 20, n_samples: int = 8, max_turns: int = 100, run_seed: int = 0):
    report = []
    for label, level, resample in CONDITIONS:
        results = run_games(
            n_games,
            lambda i: GameConfig(decks=("fignor", "igor"), seed=i, max_turns=max_turns),
            lambda i, level=level, resample=resample: {
                1: DeterminizedRolloutBot(seed=derive_agent_seed(run_seed, i, 1), n_samples=n_samples, resample=resample),
                2: HeuristicBot(seed=derive_agent_seed(run_seed, i, 2)),
            },
            privilege={1: level, 2: PrivilegeLevel.OBSERVATION},
        )
        wins = sum(1 for r in results if r.winner == 1)
        forfeits = sum(1 for r in results if r.forfeit)
        report.append((label, wins, n_games, forfeits))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--samples", type=int, default=8, help="DeterminizedRolloutBot's n_samples")
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    report = run(args.games, args.samples, args.max_turns, args.seed)
    print(f"DeterminizedRolloutBot (seat 1, n_samples={args.samples}) vs HeuristicBot (seat 2), {args.games} games/condition:\n")
    for label, wins, n, forfeits in report:
        note = f"  [{forfeits} forfeits]" if forfeits else ""
        print(f"  {label:<46} {wins:>3}/{n} wins ({wins / n:.1%}){note}")


if __name__ == "__main__":
    main()

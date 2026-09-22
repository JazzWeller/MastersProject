#!/usr/bin/env python3
"""Search-curve diagnostic (Agent Interface Plan, Milestone J): one fixed
agent (`bots.search_bot.DeterminizedRolloutBot`) at several simulation
counts (`n_samples`, its "how many forks per decision" knob), against a
fixed opponent. A flat curve rules out search-heavy designs; a steep one
rules out search-free designs.

Run from Code/Non-GUI: `python -m tools.diag_search_curve`
"""

from __future__ import annotations

import argparse
from typing import List

from bots.heuristic_bot import HeuristicBot
from bots.search_bot import DeterminizedRolloutBot
from keyforge.config import GameConfig
from keyforge.enums import PrivilegeLevel, Resample
from sim.driver import derive_agent_seed, run_games

DEFAULT_SAMPLE_COUNTS = (1, 2, 4, 8, 16)


def run(
    n_games: int = 20,
    sample_counts: List[int] = DEFAULT_SAMPLE_COUNTS,
    max_turns: int = 100,
    run_seed: int = 0,
    privilege: PrivilegeLevel = PrivilegeLevel.SEARCH,
):
    report = []
    for n_samples in sample_counts:
        results = run_games(
            n_games,
            lambda i: GameConfig(decks=("fignor", "igor"), seed=i, max_turns=max_turns),
            lambda i, n_samples=n_samples: {
                1: DeterminizedRolloutBot(
                    seed=derive_agent_seed(run_seed, i, 1), n_samples=n_samples, resample=Resample.ALL
                ),
                2: HeuristicBot(seed=derive_agent_seed(run_seed, i, 2)),
            },
            privilege={1: privilege, 2: PrivilegeLevel.OBSERVATION},
        )
        wins = sum(1 for r in results if r.winner == 1)
        forfeits = sum(1 for r in results if r.forfeit)
        report.append((n_samples, wins, n_games, forfeits))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--samples", type=int, nargs="+", default=list(DEFAULT_SAMPLE_COUNTS))
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--privileged", action="store_true", help="use the exact fork instead of a determinized one")
    args = parser.parse_args()

    privilege = PrivilegeLevel.PRIVILEGED if args.privileged else PrivilegeLevel.SEARCH
    report = run(args.games, args.samples, args.max_turns, args.seed, privilege)
    print(f"DeterminizedRolloutBot (seat 1, {privilege.value}) vs HeuristicBot (seat 2), {args.games} games/level:\n")
    for n_samples, wins, n, forfeits in report:
        note = f"  [{forfeits} forfeits]" if forfeits else ""
        print(f"  n_samples={n_samples:<4} {wins:>3}/{n} wins ({wins / n:.1%}){note}")


if __name__ == "__main__":
    main()

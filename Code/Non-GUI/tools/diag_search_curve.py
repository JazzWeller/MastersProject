#!/usr/bin/env python3
"""Search-curve diagnostic (Agent Interface Plan, Milestone J): one fixed
agent (`bots.search_bot.DeterminizedRolloutBot`) at several simulation
counts (`n_samples`, its "how many forks per decision" knob), against a
fixed opponent. A flat curve rules out search-heavy designs; a steep one
rules out search-free designs.

Run from Code/Non-GUI: `python -m tools.diag_search_curve`

**Agent Training Plan (M9)** version, the default from the command line:
the M4/M5 `SearchAgent` at 1 / 10 / 100 / 1,000 simulations per decision
(`--sims`), paired seeds against HeuristicBot, in parallel, with the mean
seconds per searched decision at each level -- strength against compute,
never a single number. The M9 decision rules read its shape: flat from 10
to 100 makes search-free DMC the primary design; still steep at 1,000
turns on the fast fork backend. `--agent rollout` keeps the interface
plan's original DeterminizedRolloutBot version (`run`).
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


def run_search_levels(regime: str = "within_turn", levels=(1, 10, 100, 1000), n_seeds: int = 50, workers: int = 8,
                      first_seed: int = 4_000_000, max_turns: int = 200, **agent_kwargs) -> List[dict]:
    from sim.parallel_eval import AgentSpec, evaluate_parallel

    name = "search-" + regime.replace("_", "-")
    out = []
    for sims in levels:
        res = evaluate_parallel(AgentSpec.of(name, simulations=sims, **agent_kwargs), AgentSpec.of("heuristic"),
                                range(first_seed, first_seed + n_seeds), workers=workers, max_turns=max_turns)
        res["simulations"] = sims
        out.append(res)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", choices=("search", "rollout"), default="search")
    parser.add_argument("--regime", default="within_turn")
    parser.add_argument("--sims", type=int, nargs="+", default=[1, 10, 100, 1000])
    parser.add_argument("--seeds", type=int, default=50, help="paired seeds per level (search agent)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--samples", type=int, nargs="+", default=list(DEFAULT_SAMPLE_COUNTS))
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--privileged", action="store_true", help="use the exact fork instead of a determinized one")
    args = parser.parse_args()
    if args.agent == "search":
        rows = run_search_levels(args.regime, args.sims, args.seeds, args.workers, max_turns=max(args.max_turns, 200))
        print(f"SearchAgent ({args.regime}) vs HeuristicBot, {args.seeds} paired seeds x 4 games/level:\n")
        for r in rows:
            spd = r.get("mean_seconds_per_search") or 0.0
            print(f"  {r['simulations']:>5} sims  {r['a_score']:.3f} +/- {r['standard_error']:.3f}"
                  f"  ({spd * 1000:.0f} ms/search, {r['games']} games)")
        return

    privilege = PrivilegeLevel.PRIVILEGED if args.privileged else PrivilegeLevel.SEARCH
    report = run(args.games, args.samples, args.max_turns, args.seed, privilege)
    print(f"DeterminizedRolloutBot (seat 1, {privilege.value}) vs HeuristicBot (seat 2), {args.games} games/level:\n")
    for n_samples, wins, n, forfeits in report:
        note = f"  [{forfeits} forfeits]" if forfeits else ""
        print(f"  n_samples={n_samples:<4} {wins:>3}/{n} wins ({wins / n:.1%}){note}")


if __name__ == "__main__":
    main()

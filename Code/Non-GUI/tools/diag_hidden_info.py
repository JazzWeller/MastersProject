#!/usr/bin/env python3
"""Hidden-information diagnostic (Agent Interface Plan, Milestone J): the
same search agent (`bots.search_bot.DeterminizedRolloutBot`) with and
without privileged observation, and with each `Resample` mode (Milestone
D), against a fixed opponent. Separates "the cost of not knowing my own
draws" from "the cost of not knowing their hand", and puts an upper bound
on what belief modelling could buy: nothing beats the privileged condition,
since that IS the true hidden state.

Run from Code/Non-GUI: `python -m tools.diag_hidden_info`

**Agent Training Plan (M9)** version, the default from the command line:
the M4/M5 `SearchAgent` itself (`--agent search`, regime `--regime`) under
four conditions -- the exact fork (privileged: an upper bound on what all
belief modelling can ever buy), own deck only resampled (knows their hand,
not its draws), their private cards only resampled (knows its draws, not
their hand), everything resampled (the honest agent) -- each over paired
seeds against HeuristicBot, in parallel (`run_search_conditions`). The M9
decision rule reads the gap between the first and last: under 3 points,
the belief arm is dropped. `--agent rollout` keeps the interface plan's
original DeterminizedRolloutBot version (`run`).
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


SEARCH_CONDITIONS = [
    ("exact fork (privileged upper bound)", "{base}-privileged", None),
    ("own deck resampled (knows their hand)", "{base}", "own_deck"),
    ("their private cards resampled (knows its draws)", "{base}", "opponent_private"),
    ("everything resampled (the honest agent)", "{base}", "all"),
]


def run_search_conditions(regime: str = "within_turn", simulations: int = 100, n_seeds: int = 50, workers: int = 8,
                          first_seed: int = 3_000_000, max_turns: int = 200, **agent_kwargs) -> List[dict]:
    from sim.parallel_eval import AgentSpec, evaluate_parallel

    base = "search-" + regime.replace("_", "-")
    out = []
    for label, name, resample in SEARCH_CONDITIONS:
        kwargs = {"simulations": simulations, **agent_kwargs}
        if resample:
            kwargs["resample"] = resample
        res = evaluate_parallel(AgentSpec.of(name.format(base=base), **kwargs), AgentSpec.of("heuristic"),
                                range(first_seed, first_seed + n_seeds), workers=workers, max_turns=max_turns)
        res["condition"] = label
        out.append(res)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", choices=("search", "rollout"), default="search")
    parser.add_argument("--regime", default="within_turn")
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=50, help="paired seeds per condition (search agent)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--samples", type=int, default=8, help="DeterminizedRolloutBot's n_samples")
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.agent == "search":
        rows = run_search_conditions(args.regime, args.sims, args.seeds, args.workers, max_turns=max(args.max_turns, 200))
        print(f"SearchAgent ({args.regime}, {args.sims} sims) vs HeuristicBot, {args.seeds} paired seeds x 4 games/condition:\n")
        for r in rows:
            print(f"  {r['condition']:<50} {r['a_score']:.3f} +/- {r['standard_error']:.3f}  ({r['games']} games, {r['forfeits']} forfeits)")
        gap = rows[0]["a_score"] - rows[-1]["a_score"]
        verdict = "drop the belief arm (< 3 points)" if gap < 0.03 else "belief modelling has room to help"
        print(f"\n  hidden-information gap (exact - honest): {gap:+.3f} -> {verdict}")
        return

    report = run(args.games, args.samples, args.max_turns, args.seed)
    print(f"DeterminizedRolloutBot (seat 1, n_samples={args.samples}) vs HeuristicBot (seat 2), {args.games} games/condition:\n")
    for label, wins, n, forfeits in report:
        note = f"  [{forfeits} forfeits]" if forfeits else ""
        print(f"  {label:<46} {wins:>3}/{n} wins ({wins / n:.1%}){note}")


if __name__ == "__main__":
    main()

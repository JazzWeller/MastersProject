#!/usr/bin/env python3
"""Repeatable engine throughput benchmark (Agent Interface Plan, Milestone 0).

Reports decisions/sec, games/sec, the decision-kind histogram, choice-space
percentiles, replay-fork cost at 25/50/75/100% of a game, and the cost of a
driver-detached `copy.deepcopy` of a game's data state. Every later milestone
in Code/AGENT_INTERFACE_PLAN.md re-runs this so a regression on the hot path
of every future self-play run gets caught immediately, not after weeks of
training time have been spent on top of it.

Run from `Code/Non-GUI`: `python -m tools.bench_engine`
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from typing import Dict, List, Optional

from bots.heuristic_bot import HeuristicBot
from bots.random_bot import RandomBot
from keyforge.config import GameConfig
from keyforge.enums import DecisionKind
from keyforge.game import Game
from keyforge.replay import replay

BOTS = {"random": RandomBot, "heuristic": HeuristicBot}


def _choice_space(kind: DecisionKind, n_options: int, min_n: int, max_n: int) -> int:
    """The number of distinct legal submissions for one decision -- not just
    `len(options)`, since CHOOSE_CARDS/ORDER_EFFECTS submit a list."""
    if n_options == 0:
        return 1
    if kind == DecisionKind.ORDER_EFFECTS:
        return math.factorial(n_options)
    if kind == DecisionKind.CHOOSE_CARDS:
        hi = min(max_n, n_options)
        lo = min(min_n, hi)
        return sum(math.comb(n_options, k) for k in range(lo, hi + 1))
    return n_options


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] + (s[c] - s[f]) * (k - f)


def play_games(n_games: int, seed: Optional[int], decks, bot_name: str, build_views: bool, max_turns: int):
    """Plays `n_games` games single-threaded, returning per-decision stats and
    one representative finished game (config + choice_record) for the fork
    and copy benchmarks below."""
    bot_cls = BOTS[bot_name]
    # Milestone L "lazy observations": a harness shouldn't pay build_view()'s
    # cost for an agent that never looks at its `view` argument (RandomBot).
    # `build_views=False` (--compare-views' second run) still forces it off
    # regardless, to isolate build_view()'s cost in isolation.
    bot_wants_view = build_views and getattr(bot_cls, "needs_view", True)
    kind_counts: Dict[DecisionKind, int] = {}
    space_by_kind: Dict[DecisionKind, List[int]] = {}
    total_decisions = 0
    total_turns = 0
    representative = None

    t0 = time.perf_counter()
    for i in range(n_games):
        game_seed = (seed + i) if seed is not None else None
        config = GameConfig(decks=decks, seed=game_seed, max_turns=max_turns)
        game = Game(config)
        controllers = {
            1: bot_cls(seed=game_seed),
            2: bot_cls(seed=(game_seed or 0) + 1 if game_seed is not None else None),
        }
        while not game.is_over:
            d = game.pending_decision
            view = game.view_for(d.player) if bot_wants_view else None
            choice = controllers[d.player].decide(view, d)
            n_options = len(d.options)
            kind_counts[d.kind] = kind_counts.get(d.kind, 0) + 1
            space_by_kind.setdefault(d.kind, []).append(_choice_space(d.kind, n_options, d.min_n, d.max_n))
            total_decisions += 1
            game.submit(choice)
        total_turns += game.turn_number
        representative = (config, list(game.choice_record))
    elapsed = time.perf_counter() - t0

    return {
        "elapsed": elapsed,
        "n_games": n_games,
        "total_decisions": total_decisions,
        "total_turns": total_turns,
        "kind_counts": kind_counts,
        "space_by_kind": space_by_kind,
        "representative": representative,
    }


def bench_replay_fork(config: GameConfig, record: List) -> Dict[str, float]:
    n = len(record)
    results = {}
    for frac in (0.25, 0.50, 0.75, 1.0):
        upto = max(1, round(n * frac))
        t0 = time.perf_counter()
        replay(config, record, upto=upto)
        results[f"{int(frac * 100)}%"] = (time.perf_counter() - t0) * 1000.0
    return results


def bench_copy(config: GameConfig, record: List) -> Dict[str, float]:
    """Cost of a plain `copy.deepcopy` once the suspended-generator driver is
    detached -- `Game` itself can never be deepcopied (`_driver` holds a live
    generator), so this measures the data-state-only figure the plan's
    'Verified engine facts' cites as an upper bound, not a working fork."""
    mid = max(1, len(record) // 2)
    game = replay(config, record, upto=mid)
    driver = game._driver
    game._driver = None
    t0 = time.perf_counter()
    copy.deepcopy(game)
    elapsed = (time.perf_counter() - t0) * 1000.0
    game._driver = driver
    return {"at_50%_ms": elapsed, "decision_index": mid}


def format_report(with_views: dict, without_views: Optional[dict], fork: dict, copy_cost: dict) -> str:
    lines = []
    for label, stats in (("with views", with_views), ("without views", without_views)):
        if stats is None:
            continue
        dec_per_sec = stats["total_decisions"] / stats["elapsed"] if stats["elapsed"] else 0.0
        games_per_sec = stats["n_games"] / stats["elapsed"] if stats["elapsed"] else 0.0
        avg_turns = stats["total_turns"] / stats["n_games"] if stats["n_games"] else 0.0
        avg_decisions = stats["total_decisions"] / stats["n_games"] if stats["n_games"] else 0.0
        lines.append(f"=== {stats['n_games']} games, {label} ===")
        lines.append(f"  decisions/sec: {dec_per_sec:,.0f}   games/sec: {games_per_sec:,.1f}")
        lines.append(f"  avg decisions/game: {avg_decisions:.1f}   avg turns/game: {avg_turns:.1f}")
        lines.append("  decision kind          %of all   max space   p99 space")
        total = stats["total_decisions"] or 1
        for kind, count in sorted(stats["kind_counts"].items(), key=lambda kv: -kv[1]):
            spaces = stats["space_by_kind"][kind]
            pct = 100.0 * count / total
            lines.append(
                f"    {kind.name:<20}  {pct:5.1f}%   {max(spaces):>9}   {_percentile(spaces, 99):>9.0f}"
            )
        lines.append("")

    lines.append("=== replay-fork cost (fresh Game.__init__ + N submits) ===")
    for label, ms in fork.items():
        lines.append(f"  at {label:>4} of game: {ms:.3f} ms")
    lines.append("")
    lines.append("=== driver-detached deepcopy cost (data state only, not a working fork) ===")
    lines.append(f"  at decision {copy_cost['decision_index']}: {copy_cost['at_50%_ms']:.3f} ms")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bot", choices=sorted(BOTS), default="random")
    parser.add_argument("--p1-deck", default="fignor")
    parser.add_argument("--p2-deck", default="igor")
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--compare-views", action="store_true", help="also run with build_view() skipped, to isolate its cost")
    parser.add_argument("--json", default=None, help="write raw per-run stats to this path as JSON")
    args = parser.parse_args(argv)

    decks = (args.p1_deck, args.p2_deck)
    with_views = play_games(args.games, args.seed, decks, args.bot, build_views=True, max_turns=args.max_turns)
    without_views = None
    if args.compare_views:
        without_views = play_games(args.games, args.seed, decks, args.bot, build_views=False, max_turns=args.max_turns)

    config, record = with_views["representative"]
    fork = bench_replay_fork(config, record)
    copy_cost = bench_copy(config, record)

    print(format_report(with_views, without_views, fork, copy_cost))

    if args.json:
        payload = {
            "with_views": {
                "elapsed": with_views["elapsed"],
                "n_games": with_views["n_games"],
                "total_decisions": with_views["total_decisions"],
                "total_turns": with_views["total_turns"],
                "kind_counts": {k.name: v for k, v in with_views["kind_counts"].items()},
            },
            "fork_ms": fork,
            "copy_ms": copy_cost,
        }
        if without_views:
            payload["without_views"] = {
                "elapsed": without_views["elapsed"],
                "total_decisions": without_views["total_decisions"],
            }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()

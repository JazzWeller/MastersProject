#!/usr/bin/env python3
"""A rating ladder: every pair of players plays `tools.eval_v2`'s paired
match, and one Bradley-Terry fit turns the results into Elo ratings
(after Tier 2 round 3, where both arms beat HeuristicBot ~96% of the time
and that one opponent stopped telling networks apart).

    python -m tools.ladder --name tier2 --players heuristic v1:tier0b-a0 v1:tier2-v1-s2 v2:tier2-v2-s2 \\
        --sims 100 --seeds 100 --workers 16

Players are `eval_v2`'s (`v1:<run>`, `v2:<run>`, `<run>@final`, `<run>@p<N>`,
a bot name, any of them with `#regime=...,leaf=...,sims=...` options), and
`--deck-set N` plays every match on held-out random decks (M11). Each pair is one `eval_v2` run with its default output under
`$KEYFORGE_DATA/evals/`, so a pair already played -- by an earlier ladder or
an earlier evaluation, in either order -- is read, not replayed, and an
interrupted ladder resumes where it stopped (SIGTERM ends the running match,
which resumes from its own partial file). `--skip-missing` drops a player
whose run or checkpoint doesn't exist yet, so one ladder definition can name
runs still to come.

The fit: the strength of each player maximizes the likelihood of every
pair's score (wins plus half the draws, over games), with the first player
fixed at 0 Elo; the +/- is one standard error from the fit's information
matrix. Written to `$KEYFORGE_DATA/evals/ladder-<name>.json` and printed.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import signal
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAUSED = 75


def _out_name(a: str, b: str, sims: int, seeds: int, first_seed: int, deck_set: int = 0, deck_seed: int = 2) -> str:
    """`eval_v2`'s default output file for a against b."""
    from sim import data_root
    from tools.eval_v2 import out_name

    return data_root.resolve(os.path.join("evals", out_name(a, b, sims, seeds, first_seed, False, deck_set, deck_seed)))


def _available(player: str) -> bool:
    if ":" not in player:
        return True
    from tools.eval_v2 import _checkpoint, player_options

    try:
        _checkpoint(player_options(player)[0].split(":", 1)[1])
        return True
    except (FileNotFoundError, StopIteration, KeyError, IndexError, json.JSONDecodeError):
        return False


def result_for(a: str, b: str, sims: int, seeds: int, first_seed: int, deck_set: int = 0,
               deck_seed: int = 2) -> Optional[Tuple[float, int]]:
    """(a's points, games) from a finished match in either order, or None."""
    for x, y, flip in ((a, b, False), (b, a, True)):
        path = _out_name(x, y, sims, seeds, first_seed, deck_set, deck_seed)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                r = json.load(f)
            points = r["a_wins"] + 0.5 * r["draws"]
            return (r["games"] - points if flip else points), r["games"]
    return None


def fit(players: Sequence[str], results: Dict[Tuple[int, int], Tuple[float, int]], iterations: int = 2000):
    """Bradley-Terry by minorization-maximization (Hunter 2004), then Elo
    with player 0 at 0 and standard errors from the observed information.
    `results[(i, j)]` = (i's points against j, games)."""
    n = len(players)
    wins = [0.0] * n
    games = [[0.0] * n for _ in range(n)]
    for (i, j), (p, g) in results.items():
        wins[i] += p
        wins[j] += g - p
        games[i][j] += g
        games[j][i] += g
    # A player with no points (or all of them) has no finite rating; half a
    # point either way keeps the fit finite (it reads as "at least this far").
    w = [min(max(x, 0.5), sum(games[i]) - 0.5) if sum(games[i]) else 0.0 for i, x in enumerate(wins)]
    gamma = [1.0] * n
    for _ in range(iterations):
        new = []
        for i in range(n):
            den = sum(games[i][j] / (gamma[i] + gamma[j]) for j in range(n) if j != i and games[i][j])
            new.append(w[i] / den if den else gamma[i])
        s = new[0]
        new = [x / s for x in new]
        done = max(abs(math.log(a / b)) for a, b in zip(new, gamma)) < 1e-10
        gamma = new
        if done:
            break
    theta = [math.log(x) for x in gamma]
    # Observed information over theta (natural log scale), player 0 fixed.
    info = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and games[i][j]:
                p = 1.0 / (1.0 + math.exp(theta[j] - theta[i]))
                v = games[i][j] * p * (1 - p)
                info[i][i] += v
                info[i][j] -= v
    sub = [row[1:] for row in info[1:]]
    cov = _inverse(sub) if n > 1 else []
    k = 400.0 / math.log(10.0)
    elo = [k * t for t in theta]
    se = [0.0] + [k * math.sqrt(max(cov[i][i], 0.0)) for i in range(n - 1)]
    return elo, se


def _inverse(m: List[List[float]]) -> List[List[float]]:
    n = len(m)
    a = [list(row) + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(m)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(a[r][c]))
        if abs(a[piv][c]) < 1e-12:
            raise ValueError("the ladder's results don't connect every player")
        a[c], a[piv] = a[piv], a[c]
        d = a[c][c]
        a[c] = [x / d for x in a[c]]
        for r in range(n):
            if r != c and a[r][c]:
                f = a[r][c]
                a[r] = [x - f * y for x, y in zip(a[r], a[c])]
    return [row[n:] for row in a]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--players", nargs="+", required=True, help="the first is the 0-Elo anchor")
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--first-seed", type=int, default=3_000_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--deck-set", type=int, default=0, help="every match on N held-out random deck pairs (eval_v2)")
    parser.add_argument("--deck-seed", type=int, default=2)
    args = parser.parse_args()

    players = [p for p in args.players if not args.skip_missing or _available(p)]
    dropped = [p for p in args.players if p not in players]
    if dropped:
        print(f"[ladder] skipping (not there yet): {', '.join(dropped)}", flush=True)
    child: List[Optional[subprocess.Popen]] = [None]

    def on_term(_signum, _frame):
        if child[0] is not None and child[0].poll() is None:
            child[0].terminate()  # eval_v2 keeps its finished units
            child[0].wait()
        raise SystemExit(PAUSED)

    signal.signal(signal.SIGTERM, on_term)
    pairs = list(itertools.combinations(range(len(players)), 2))
    results: Dict[Tuple[int, int], Tuple[float, int]] = {}
    for k, (i, j) in enumerate(pairs, 1):
        a, b = players[i], players[j]
        where = (args.sims, args.seeds, args.first_seed, args.deck_set, args.deck_seed)
        r = result_for(a, b, *where)
        if r is None:
            print(f"[ladder] {k}/{len(pairs)}: {a} vs {b}", flush=True)
            child[0] = subprocess.Popen(
                [sys.executable, "-m", "tools.eval_v2", "--a", a, "--b", b, "--sims", str(args.sims),
                 "--seeds", str(args.seeds), "--first-seed", str(args.first_seed), "--workers", str(args.workers),
                 "--deck-set", str(args.deck_set), "--deck-seed", str(args.deck_seed)],
                cwd=HERE)
            code = child[0].wait()
            child[0] = None
            if code != 0:
                raise SystemExit(code)
            r = result_for(a, b, *where)
        else:
            print(f"[ladder] {k}/{len(pairs)}: {a} vs {b} already played ({r[0] / r[1]:.3f})", flush=True)
        results[(i, j)] = r

    elo, se = fit(players, results)
    order = sorted(range(len(players)), key=lambda i: -elo[i])
    table = [{"player": players[i], "elo": round(elo[i], 1), "se": round(se[i], 1)} for i in order]
    matches = [{"a": players[i], "b": players[j], "a_score": round(p / g, 4), "games": g}
               for (i, j), (p, g) in results.items()]
    report = {"name": args.name, "anchor": players[0], "sims": args.sims, "seeds": args.seeds, "deck_set": args.deck_set,
              "first_seed": args.first_seed, "ratings": table, "matches": matches, "skipped": dropped}
    from sim import data_root

    out = data_root.resolve(os.path.join("evals", f"ladder-{args.name}.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(f"\nladder {args.name} ({args.sims} sims, {args.seeds} paired seeds a match; {players[0]} = 0)")
    for row in table:
        print(f"  {row['elo']:8.1f} +/- {row['se']:5.1f}  {row['player']}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()

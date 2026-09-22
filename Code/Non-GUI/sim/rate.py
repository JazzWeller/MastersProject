"""Bradley-Terry ratings with confidence intervals, an explicit
non-transitivity report, and SPRT early stopping (Agent Interface Plan,
Milestone I). Pure standard library, no numpy/scipy -- keyforge/bots/sim
stay importable under PyPy, per the plan's own scope note.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class Rating:
    name: str
    rating: float  # log-strength; the anchor is pinned at 0.0
    games: int
    wins: float  # a draw counts as half a win


def bradley_terry(
    results: Iterable[Tuple[str, str, float]],
    *,
    anchor: Optional[str] = None,
    iterations: int = 200,
    tol: float = 1e-9,
) -> Dict[str, Rating]:
    """`results` is an iterable of `(name_a, name_b, score_a)`, `score_a`
    being 1.0/0.0/0.5 for a win/loss/draw. Fits each name's Bradley-Terry
    strength via the standard minorization-maximization iteration (Hunter,
    2004): monotonically convergent, no learning rate to tune.
    `anchor` (default: the first name seen) is pinned at rating 0.0, since
    Bradley-Terry strengths are only defined up to a multiplicative (here,
    additive on the log scale) constant.
    """
    wins: Dict[str, float] = defaultdict(float)
    total_games: Dict[str, int] = defaultdict(int)
    pair_games: Dict[Tuple[str, str], float] = defaultdict(float)
    names: List[str] = []
    seen = set()

    for a, b, score_a in results:
        for n in (a, b):
            if n not in seen:
                seen.add(n)
                names.append(n)
        pair_games[(a, b)] += 1
        pair_games[(b, a)] += 1
        total_games[a] += 1
        total_games[b] += 1
        wins[a] += score_a
        wins[b] += 1 - score_a

    if not names:
        return {}
    if anchor is None:
        anchor = names[0]

    strength = {n: 1.0 for n in names}
    for _ in range(iterations):
        new_strength = {}
        for a in names:
            denominator = 0.0
            for b in names:
                if a == b:
                    continue
                n_ab = pair_games.get((a, b), 0.0)
                if n_ab:
                    denominator += n_ab / (strength[a] + strength[b])
            new_strength[a] = max(wins[a] / denominator, 1e-9) if denominator > 0 else strength[a]
        # Normalize by the geometric mean each step, not by the anchor's
        # own value: a competitor with zero wins converges toward the
        # 1e-9 floor, and rescaling by THAT every iteration (instead of a
        # stable reference) blows up everyone else's strength over
        # `iterations` repeated divisions. The anchor shift happens once,
        # after convergence, below -- a single additive move on the log
        # scale, which changes no gap between any two ratings.
        log_mean = sum(math.log(v) for v in new_strength.values()) / len(new_strength)
        gmean = math.exp(log_mean)
        for n in names:
            new_strength[n] /= gmean
        max_delta = max(abs(new_strength[n] - strength[n]) for n in names)
        strength = new_strength
        if max_delta < tol:
            break

    anchor_log = math.log(strength[anchor])
    return {
        n: Rating(name=n, rating=math.log(strength[n]) - anchor_log, games=total_games[n], wins=wins[n]) for n in names
    }


def standard_error(n_games: int) -> float:
    """~0.5/sqrt(N): about 1,000 games to resolve a 3-point win-rate gap
    and 10,000 for 1 point (the plan's own sample-size guidance)."""
    return 0.5 / math.sqrt(n_games) if n_games > 0 else float("inf")


def games_needed_for_gap(win_rate_gap: float) -> float:
    """Roughly how many games to resolve a win-rate gap of `win_rate_gap`
    (e.g. 0.03 for "3 points") at about two standard errors (`2 *
    standard_error(N) ~= win_rate_gap`, solved for N) -- the plan's own
    figures (~1,000 games for a 3-point gap, ~10,000 for 1 point) are at
    this threshold, not one standard error."""
    if win_rate_gap <= 0:
        return float("inf")
    return math.ceil((1.0 / win_rate_gap) ** 2)


def non_transitivity_report(ratings: Dict[str, Rating], pairwise_win_rate: Dict[Tuple[str, str], float]) -> List[str]:
    """Flags pairs whose head-to-head result disagrees with their fitted
    rating order -- Bradley-Terry assumes transitivity (A>B, B>C => A>C),
    which a real agent population doesn't have to satisfy, and a rating
    alone can't tell you it failed."""
    notes = []
    names = sorted(ratings, key=lambda n: -ratings[n].rating)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            wr = pairwise_win_rate.get((a, b))
            if wr is None:
                continue
            if ratings[a].rating > ratings[b].rating and wr < 0.5:
                notes.append(f"{a} rates above {b} but lost their head-to-head ({wr:.1%} win rate for {a})")
    return notes


def _elo_to_prob(elo_diff: float) -> float:
    return 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))


def sprt(
    wins: int, losses: int, draws: int, *, elo0: float = 0.0, elo1: float = 5.0, alpha: float = 0.05, beta: float = 0.05
) -> Optional[str]:
    """A sequential probability ratio test over win/loss/draw counts,
    testing H0 ("the true elo difference is `elo0`") against H1 ("it's
    `elo1`") -- standard practice in chess-engine testing (cutechess-cli's
    SPRT), which is what lets a clear-cut A-vs-B comparison stop on a
    fraction of a fixed-N budget. Returns `"H0"`/`"H1"` once decided, or
    `None` to keep playing. Draws count as half a win toward each
    hypothesis's likelihood -- an accepted approximation (not a full
    trinomial win/draw/loss model), matching common practice.
    """
    p0, p1 = _elo_to_prob(elo0), _elo_to_prob(elo1)
    w, l = wins + 0.5 * draws, losses + 0.5 * draws
    if w + l == 0:
        return None
    llr = w * math.log(p1 / p0) + l * math.log((1 - p1) / (1 - p0))
    lower_bound = math.log(beta / (1 - alpha))
    upper_bound = math.log((1 - beta) / alpha)
    if llr >= upper_bound:
        return "H1"
    if llr <= lower_bound:
        return "H0"
    return None

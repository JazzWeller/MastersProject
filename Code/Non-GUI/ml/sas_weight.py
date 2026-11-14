"""Outcome targets weighted by how surprising each game's result was, given the decks' SAS ratings
(`selfplay.sas_weighting`; the plateau-sasw arms).

A game between decks rated a and b (seat 1, seat 2) is expected to score

    E1 = 1 / (1 + 10 ** (-k (a - b) / 400))

for seat 1, k being Elo per SAS point. Every position of a decided game gets the weight 2 |s1 - E1|, s1 seat 1's
score (1 or 0): twice the result's surprise, the same for both seats. A draw (the turn limit, which says nothing
about the decks) keeps weight 1. Even decks weigh 1 whatever happened (as
without weighting); the stronger deck winning weighs less, the weaker deck winning more (up to 2). The weights
scale only the outcome-regression losses (value, oracle), as a weighted mean over the batch, so their size is
unchanged; the policy (search visit) targets don't depend on the outcome and stay as they are. Validation stays
unweighted, so held-out losses compare with an unweighted run's.

k comes from the config, or (`"fit:<run>"`) from a logistic fit to that run's own self-play games (the higher
deck's score against the gap, no intercept: the conditions these games are played in), cached in
`<run>/sas_elo.json`.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Optional


def expected(gap: float, elo_per_sas: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo_per_sas * gap / 400.0))


class SasWeigher:
    def __init__(self, deck_list: str, elo_per_sas: float):
        from sim.decklists import sas_of

        self.deck_list = deck_list
        self.k = float(elo_per_sas)
        self._sas = lambda name: sas_of(deck_list, name)

    def game_weight(self, record: dict) -> float:
        d1, d2 = record["meta"]["decks"]
        e1 = expected(self._sas(d1) - self._sas(d2), self.k)
        z1 = record["outcome"][1] if 1 in record["outcome"] else record["outcome"]["1"]
        if float(z1) == 0.0:
            return 1.0
        s1 = (float(z1) + 1.0) / 2.0
        return 2.0 * abs(s1 - e1)


def fit_from_shards(shard_dir: str, deck_list: str, max_games: int = 10_000) -> dict:
    """k from a run's self-play games: up to `max_games`, spread over its shards."""
    from agent.selfplay import read_frames
    from sim.decklists import sas_of
    from tools.sas_report import fit_logistic

    shards = sorted(glob.glob(os.path.join(shard_dir, "actor_*.bin")))
    if not shards:
        raise FileNotFoundError(f"no shards in {shard_dir}")
    per = max(1, max_games // len(shards))
    xs, ys = [], []
    for path in shards:
        for n, (_end, rec) in enumerate(read_frames(path)):
            if n >= per:
                break
            d1, d2 = rec["meta"]["decks"]
            gap = sas_of(deck_list, d1) - sas_of(deck_list, d2)
            if gap == 0:
                continue
            z1 = rec["outcome"][1] if 1 in rec["outcome"] else rec["outcome"]["1"]
            s1 = (float(z1) + 1.0) / 2.0
            xs.append(abs(gap))
            ys.append(s1 if gap > 0 else 1.0 - s1)
    if len(xs) < 200:
        raise ValueError(f"only {len(xs)} games with unequal decks in {shard_dir}")
    _b0, k, _s0, se = fit_logistic(xs, ys, intercept=False)
    return {"elo_per_sas": round(k, 3), "se": round(se, 3), "games": len(xs), "deck_list": deck_list,
            "higher_deck_score": round(sum(ys) / len(ys), 4)}


def resolve(spec: Optional[dict], runs_root: str) -> Optional[SasWeigher]:
    """The weigher for `selfplay.sas_weighting` (None: off)."""
    if not spec:
        return None
    deck_list = spec.get("deck_list", "phase3")
    k = spec.get("elo_per_sas")
    if isinstance(k, str) and k.startswith("fit:"):
        src = k[4:]
        cache = os.path.join(runs_root, src, "sas_elo.json")
        if os.path.exists(cache):
            with open(cache, encoding="utf-8") as f:
                fit = json.load(f)
        else:
            fit = fit_from_shards(os.path.join(runs_root, src, "shards"), deck_list)
            tmp = f"{cache}.{os.getpid()}.tmp"  # several arms may fit at once: each writes whole, the last one stays
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(fit, f, indent=1)
            os.replace(tmp, cache)
        k = fit["elo_per_sas"]
    if k is None:
        raise ValueError("sas_weighting needs elo_per_sas (a number or \"fit:<run>\")")
    return SasWeigher(deck_list, float(k))

"""What a SAS advantage is worth, from `eval-sas-*`'s evaluations (`tools.eval_v2 --sas-gap`, archon or adaptive).

    python -m tools.sas_report <eval json or directory> ... [--deck-list phase3] [--out sas_report.json]

Reads every evaluation that has `sas_gap` set and keeps each game (or match) with both decks' SAS ratings:

- a mirror (the same player on both sides): how often the higher-SAS deck wins, per gap band, and a logistic fit
  of Elo per SAS point;
- a player against another: its score holding the higher and the lower deck, and a two-parameter fit
  (its Elo at equal decks, Elo per SAS point), whose ratio is its skill in SAS points;
- adaptive: how often a match goes to game 3, the winning bids, whether the higher-SAS deck is the one bid on,
  how bids track the gap, and how the bid winner does in game 3 (a fair auction leaves it near 50%).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from typing import Dict, List, Optional, Tuple

ELO = 400.0 / math.log(10.0)


def _score(winner: Optional[str]) -> float:
    return 1.0 if winner == "a" else (0.0 if winner == "b" else 0.5)


def fit_logistic(xs: List[float], ys: List[float], intercept: bool = True) -> Tuple[float, float, float, float]:
    """Max-likelihood p = sigmoid(b0 + b1 x) for scores y in [0, 1] (Newton). Returns (b0, b1, se0, se1) in Elo."""
    b0 = b1 = 0.0
    for _ in range(50):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for x, y in zip(xs, ys):
            p = 1.0 / (1.0 + math.exp(-(b0 + b1 * x)))
            w = p * (1 - p)
            g0 += y - p
            g1 += (y - p) * x
            h00 += w
            h01 += w * x
            h11 += w * x * x
        if intercept:
            det = h00 * h11 - h01 * h01
            if det <= 0:
                break
            d0 = (h11 * g0 - h01 * g1) / det
            d1 = (h00 * g1 - h01 * g0) / det
        else:
            d0, d1 = 0.0, (g1 / h11 if h11 > 0 else 0.0)
        b0 += d0
        b1 += d1
        if abs(d0) + abs(d1) < 1e-10:
            break
    if intercept:
        det = h00 * h11 - h01 * h01
        se0, se1 = (math.sqrt(h11 / det), math.sqrt(h00 / det)) if det > 0 else (float("inf"), float("inf"))
    else:
        se0, se1 = 0.0, (1 / math.sqrt(h11) if h11 > 0 else float("inf"))
    return b0 * ELO, b1 * ELO, se0 * ELO, se1 * ELO


def _elo(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return ELO * math.log(p / (1 - p))


def _mean_se(per_unit: List[float]) -> Tuple[float, float]:
    n = len(per_unit)
    if not n:
        return float("nan"), float("nan")
    m = sum(per_unit) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in per_unit) / (n - 1)) if n > 1 else float("nan")
    return m, sd / math.sqrt(n)


def load(paths: List[str]) -> List[dict]:
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, "*.json"))) if os.path.isdir(p) else [p]
    out = []
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            continue
        if isinstance(d, dict) and d.get("sas_gap") and d.get("units"):
            d["_file"] = os.path.basename(f)
            out.append(d)
    return out


def _base(spec: str) -> str:
    return spec.split("@")[0]


def analyse(ev: dict, sas) -> dict:
    mirror = ev["a"] == ev["b"]
    units = []  # per unit: list of (d = a_sas - b_sas, score of a, record)
    for u in ev["units"]:
        rows = []
        for r in u.get("records") or []:
            d = sas(r["a_deck"]) - sas(r["b_deck"])
            rows.append((d, _score(r["winner"]), r))
        if rows:
            units.append(rows)
    allrows = [x for rows in units for x in rows]
    gaps = [abs(d) for d, _s, _r in allrows]
    res = {"file": ev["_file"], "a": ev["a"], "b": ev["b"], "format": ev.get("format", "archon"), "gap": ev["sas_gap"],
           "units": len(units), "results": len(allrows), "mean_gap": round(sum(gaps) / len(gaps), 2) if gaps else None,
           "a_score": ev.get("a_score")}
    if mirror:
        # the higher deck's score, per unit (both sides are the same player)
        per = []
        for rows in units:
            hi = [s if d > 0 else 1 - s for d, s, _r in rows if d != 0]
            if hi:
                per.append(sum(hi) / len(hi))
        m, se = _mean_se(per)
        res["higher_deck_score"] = round(m, 4)
        res["higher_deck_se"] = round(se, 4)
        res["higher_deck_elo"] = round(_elo(m), 1)
    else:
        for side, keep in (("with_higher", lambda d: d > 0), ("with_lower", lambda d: d < 0)):
            per = []
            for rows in units:
                ss = [s for d, s, _r in rows if keep(d)]
                if ss:
                    per.append(sum(ss) / len(ss))
            m, se = _mean_se(per)
            res[f"a_score_{side}"] = round(m, 4)
            res[f"a_score_{side}_se"] = round(se, 4)
    if res["format"] == "adaptive":
        recs = [r for _d, _s, r in allrows]
        bids = [(d, r) for d, _s, r in allrows if r.get("bid")]
        res["game3_rate"] = round(len(bids) / len(recs), 4) if recs else None
        if bids:
            chains = [r["bid"]["chains"] for _d, r in bids]
            res["mean_bid"] = round(sum(chains) / len(chains), 2)
            res["bid_zero_rate"] = round(sum(1 for c in chains if c == 0) / len(chains), 4)
            # was the proven deck the higher-SAS one? (a's deck in game 1 is a_deck)
            hi_proven = [((r["bid"]["deck"] == r["a_deck"]) == (d > 0)) for d, r in bids if d != 0]
            res["proven_is_higher_rate"] = round(sum(hi_proven) / len(hi_proven), 4) if hi_proven else None
            g3 = [r for _d, r in bids if len(r.get("games", [])) == 3]
            pilot_won = [r["games"][2]["winner"] == r["bid"]["winner"] for r in g3 if r["games"][2]["winner"]]
            res["bid_winner_game3_rate"] = round(sum(pilot_won) / len(pilot_won), 4) if pilot_won else None
            res["a_won_bid_rate"] = round(sum(1 for _d, r in bids if r["bid"]["winner"] == "a") / len(bids), 4)
    return res, allrows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--deck-list", default="phase3")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    from sim.decklists import sas_of

    def sas(name):
        return sas_of(args.deck_list, name)

    evals = load(args.paths)
    if not evals:
        raise SystemExit("no evaluations with a SAS gap found")
    rows, pooled = [], {}
    for ev in evals:
        r, allrows = analyse(ev, sas)
        rows.append(r)
        key = (ev["a"], ev["b"], r["format"])
        pooled.setdefault(key, []).extend(allrows)
    fits = []
    for (a, b, fmt), allrows in sorted(pooled.items()):
        xs = [d for d, _s, _r in allrows]
        ys = [s for _d, s, _r in allrows]
        if a == b:
            # each result seen from the higher deck's side; no intercept by symmetry
            xs2 = [abs(d) for d in xs if d != 0]
            ys2 = [s if d > 0 else 1 - s for d, s in zip(xs, ys) if d != 0]
            _b0, k, _s0, sek = fit_logistic(xs2, ys2, intercept=False)
            fits.append({"a": a, "b": b, "format": fmt, "elo_per_sas": round(k, 2), "se": round(sek, 2),
                         "results": len(xs2)})
        else:
            e, k, se_e, se_k = fit_logistic(xs, ys)
            fits.append({"a": a, "b": b, "format": fmt, "elo_at_equal_decks": round(e, 1), "se_elo": round(se_e, 1),
                         "elo_per_sas": round(k, 2), "se_per_sas": round(se_k, 2), "results": len(xs),
                         "skill_in_sas": round(e / k, 1) if k > 0 else None})
    order = {"0-5": 0, "5-10": 1, "10-20": 2}
    rows.sort(key=lambda r: (r["format"], r["a"], r["b"], order.get(r["gap"], 9), r["gap"]))
    for r in rows:
        print(json.dumps(r))
    print()
    for f in fits:
        print(json.dumps(f))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"bands": rows, "fits": fits}, fh, indent=1)
        print(f"written to {args.out}")


if __name__ == "__main__":
    main()

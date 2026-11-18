"""The report's four tables, built from the rating ladders and the SAS report under `results/`.

    python -m tools.report_tables [--results results] [--out results/tables]

Writes `<out>/tables.md` (and `<out>/tables.xlsx` when openpyxl is installed). Each ladder is read from whichever copy
under `<results>/*/evals/` holds the most matches, so the tables fill in as new ladders are exported; a cell whose
ladder or player is missing stays empty. Names come from `tools.report_names`.

1. Strength by deck set: each agent's Elo on the fixed, random, real test and real train decks.
2. Search at play time: three networks from no search to 1,000 simulations, and searching the true hidden state.
3. Design and training choices: the Elo difference each makes, within one ladder, on real and fixed decks.
4. Deck strength (SAS) and the Adaptive format.

Elo has HeuristicBot at 0 throughout (a ladder anchored elsewhere is shifted). Every standard error -- of a rating
against HeuristicBot, of a difference, of a mean of differences -- comes from the ladder fit's full covariance
(refitted from its matches), so ratings that share opponents are not treated as independent.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from typing import Dict, List, Optional, Tuple

from tools.report_names import DECK_SETS, name

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H = "heuristic"


# ---------------------------------------------------------------------- ladders
def load_ladders(results: str) -> Dict[str, dict]:
    """Ladder name -> the copy with the most matches (ties: the last path in sorted order)."""
    best: Dict[str, Tuple[int, str, dict]] = {}
    for path in sorted(glob.glob(os.path.join(results, "*", "evals", "ladder-*.json"))):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        n = len(d.get("matches", []))
        if d["name"] not in best or n >= best[d["name"]][0]:
            best[d["name"]] = (n, path, d)
    return {k: v[2] for k, v in best.items()}


class Ladders:
    def __init__(self, ladders: Dict[str, dict]):
        self.l = ladders
        self._cov: Dict[str, Tuple[Dict[str, int], list]] = {}

    def cov(self, ladder: str):
        """(player -> index, Elo covariance) from the ladder's matches, refitted with `tools.ladder.fit`."""
        if ladder not in self._cov:
            from tools.ladder import fit

            d = self.l[ladder]
            players = [d["anchor"]] + [r["player"] for r in d["ratings"] if r["player"] != d["anchor"]]
            idx = {p: i for i, p in enumerate(players)}
            results = {}
            for m in d["matches"]:
                i, j = idx[m["a"]], idx[m["b"]]
                p, g = results.get((i, j), (0.0, 0))
                results[(i, j)] = (p + m["a_score"] * m["games"], g + m["games"])
            _elo, _se, c = fit(players, results, with_cov=True)
            self._cov[ladder] = (idx, c)
        return self._cov[ladder]

    def var(self, ladder: str, weights: Dict[str, float]) -> float:
        """The variance of sum(w * elo(player)) in one ladder."""
        idx, c = self.cov(ladder)
        ws = [(idx[p], w) for p, w in weights.items() if w]
        return max(sum(wi * wj * c[i][j] for i, wi in ws for j, wj in ws), 0.0)

    def _r(self, ladder: str, player: str) -> Optional[dict]:
        d = self.l.get(ladder)
        if not d:
            return None
        for r in d["ratings"]:
            if r["player"] == player:
                return r
        return None

    def elo(self, ladder: str, player: str) -> Optional[Tuple[float, float]]:
        """(Elo with HeuristicBot at 0, standard error), or None."""
        r, h = self._r(ladder, player), self._r(ladder, H)
        if r is None or h is None:
            return None
        if player == H:
            return (0.0, 0.0)
        return (r["elo"] - h["elo"], math.sqrt(self.var(ladder, {player: 1.0, H: -1.0})))

    def diff(self, pairs: List[Tuple[str, str, str]]) -> Optional[Tuple[float, float]]:
        """The mean of elo(a) - elo(b) over (ladder, a, b) pairs; None unless every pair is rated."""
        if not pairs:
            return None
        n = len(pairs)
        total, weights = 0.0, {}  # ladder -> player -> weight in the mean
        for ladder, a, b in pairs:
            ra, rb = self._r(ladder, a), self._r(ladder, b)
            if ra is None or rb is None:
                return None
            total += ra["elo"] - rb["elo"]
            w = weights.setdefault(ladder, {})
            w[a] = w.get(a, 0.0) + 1.0 / n
            w[b] = w.get(b, 0.0) - 1.0 / n
        return (total / n, math.sqrt(sum(self.var(lad, w) for lad, w in weights.items())))


# ---------------------------------------------------------------------- tables
# A table is (title, note, header, rows); a row is a list of cells, each a string or an (estimate, se) pair or None.
T1_AGENTS = [("v1:tier0b-a0", "Imitating HeuristicBot", "Compact"), ("v2:tier0b2-a7", "Imitating HeuristicBot", "Full"),
             ("v1:tier2-v1-s3", "Self-play, fixed decks", "Compact"), ("v2:tier2-v2-s3", "Self-play, fixed decks", "Full"),
             ("v1:m11-v1", "Self-play, random decks", "Compact"), ("v2:m11-v2", "Self-play, random decks", "Full"),
             ("v1:real-v1", "Self-play, real decks (train)", "Compact"), ("v2:real-v2", "Self-play, real decks (train)", "Full")]
T1_LADDERS = [("baseline", "fixed"), ("m11", "random"), ("real-test", "real-test"), ("real-train", "real-train")]


def table1(L: Ladders):
    rows = [[name(p), trained, inputs, *[L.elo(lad, p) for lad, _ in T1_LADDERS]] for p, trained, inputs in T1_AGENTS]
    return ("Table 1. Strength by deck set",
            "Elo, HeuristicBot = 0, 100 simulations. Compare within a column: each column is its own ladder. "
            "Random decks: 32 held-out pairs, 256 games a match; otherwise 400.",
            ["Agent", "Trained by", "Inputs", *[DECK_SETS[k] for _, k in T1_LADDERS]], rows)


T2_SETTINGS = [("#search=none", "None (network alone)"), ("#sims=10", "10 simulations"), ("#sims=30", "30 simulations"),
               ("", "100 simulations"), ("#sims=300", "300 simulations"), ("#sims=1000", "1,000 simulations"),
               ("#resample=exact", "100 simulations, sees hidden cards")]
T2_NETS = [("anatomy", "v1:real-v1", "real-test"), ("diag", "v1:tier2-v1-s3", "fixed"), ("anatomy-bc", "v1:tier0b-a0", "fixed")]


def table2(L: Ladders):
    rows = [[label, *[L.elo(lad, base + opt) for lad, base, _ in T2_NETS]] for opt, label in T2_SETTINGS]
    return ("Table 2. Search at play time",
            "Elo, HeuristicBot = 0, 400 games a match. Each column is one network in its own ladder.",
            ["Search at play time", *[f"{name(b)}, {DECK_SETS[k].lower()}" for _, b, k in T2_NETS]], rows)


def _m9(train, regime, leaf, seed=1):
    return f"v1:m9-{train}-{seed}#regime={regime},leaf={leaf}"


_R1 = "v1:real-v1"
_ARMS = ("ctrl", "nogate", "sims300")
T3_ROWS = [
    ("Within-turn search at play time", "Full-game search",
     [("design-real", _R1, f"{_R1}#regime=full_game")],
     [("m9", _m9(t, "within_turn", "student"), _m9(t, "full_game", "student")) for t in ("wt", "fg")]),
    ("Belief-oracle leaf", "Student-network leaf",
     [("design-real", f"{_R1}#leaf=belief_oracle", _R1)],
     [("m9", _m9(t, r, "belief_oracle"), _m9(t, r, "student")) for t in ("wt", "fg") for r in ("within_turn", "full_game")]),
    ("Within-turn search during training", "Full-game search during training",
     [("m9-real", f"v1:m9-wt-{k}", f"v1:m9-fg-{k}") for k in (1, 2)],
     [("baseline", f"v1:m9-wt-{k}", f"v1:m9-fg-{k}") for k in (1, 2)]),
    ("Full inputs (v2)", "Compact inputs (v1), the three lineages",
     [("real-test", f"v2:{a}", f"v1:{b}") for a, b in (("tier2-v2-s3", "tier2-v1-s3"), ("m11-v2", "m11-v1"), ("real-v2", "real-v1"))],
     [("baseline", f"v2:{a}", f"v1:{b}") for a, b in (("tier2-v2-s3", "tier2-v1-s3"), ("m11-v2", "m11-v1"), ("real-v2", "real-v1"))]),
    ("4× bigger network (256×8)", "128×4, same data",
     [("size", "v1:size-256@final", "v1:size-128@final")], [("size-fi", "v1:size-256@final", "v1:size-128@final")]),
    ("+ 20k self-play games", "Real-deck agent",
     [("plateau-sasw", "v1:plateau-ctrl@final", _R1)], [("plateau-sasw-fi", "v1:plateau-ctrl@final", _R1)]),
    ("+ 20k games, no gating", "Real-deck agent",
     [("plateau-sasw", "v1:plateau-nogate@final", _R1)], [("plateau-sasw-fi", "v1:plateau-nogate@final", _R1)]),
    ("+ 20k games, 3× training search", "Real-deck agent",
     [("plateau-sasw", "v1:plateau-sims300@final", _R1)], [("plateau-sasw-fi", "v1:plateau-sims300@final", _R1)]),
    ("SAS-weighted outcome targets", "The same three runs unweighted",
     [("plateau-sasw", f"v1:plateau-sasw-{a}@final", f"v1:plateau-{a}@final") for a in _ARMS],
     [("plateau-sasw-fi", f"v1:plateau-sasw-{a}@final", f"v1:plateau-{a}@final") for a in _ARMS]),
]


def table3(L: Ladders):
    rows = [[label, vs, L.diff(real), L.diff(fixed)] for label, vs, real, fixed in T3_ROWS]
    return ("Table 3. Design and training choices",
            "Elo difference the change makes, measured within one ladder (several pairs: their mean). "
            "Fixed-deck play-time rows average M9's two training regimes; the training row its two seeds.",
            ["Change", "Compared with", DECK_SETS["real-test"], DECK_SETS["fixed"]], rows)


def _pct(x):
    return None if x is None else f"{100 * x:.1f}%"


def table4(sas: Optional[dict]):
    note = "Real test decks, 100 deck pairs per band; the agent is the Real-deck agent at 100 simulations."
    head = ["SAS gap", "Mean gap", "Higher deck's edge, agent vs itself (Elo)",
            "vs HeuristicBot with the higher deck", "vs HeuristicBot with the lower deck",
            "Adaptive vs HeuristicBot, match wins", "Adaptive mirror: reaches game 3",
            "Mean bid (chains)", "Bid winner wins game 3"]
    if not sas:
        return ("Table 4. Deck strength (SAS) and the Adaptive format", note + " (no SAS report found)", head, [])
    bands = {}
    for b in sas["bands"]:
        key = ("mirror" if b["a"] == b["b"] else "heuristic", b["format"])
        bands.setdefault(b["gap"], {})[key] = b
    order = sorted(bands, key=lambda g: float(g.split("-")[0]))
    rows = []
    for g in order:
        m = bands[g].get(("mirror", "archon"), {})
        h = bands[g].get(("heuristic", "archon"), {})
        ha = bands[g].get(("heuristic", "adaptive"), {})
        ma = bands[g].get(("mirror", "adaptive"), {})
        edge = (m["higher_deck_elo"], 400 / math.log(10) * m["higher_deck_se"] /
                (m["higher_deck_score"] * (1 - m["higher_deck_score"]))) if m else None
        gap = m.get("mean_gap") or h.get("mean_gap")
        rows.append([g, None if gap is None else round(gap, 1), edge, _pct(h.get("a_score_with_higher")),
                     _pct(h.get("a_score_with_lower")), _pct(ha.get("a_score")), _pct(ma.get("game3_rate")),
                     ma.get("mean_bid"), _pct(ma.get("bid_winner_game3_rate"))])
    fits = {(f["b"] == f["a"], f["format"]): f for f in sas.get("fits", [])}
    mf, hf = fits.get((True, "archon")), fits.get((False, "archon"))
    if mf and hf:
        note += (f" Pooled: {mf['elo_per_sas']:.1f} ± {mf['se']:.1f} Elo per SAS point (agent vs itself); skill over "
                 f"HeuristicBot {hf['elo_at_equal_decks']:.0f} ± {hf['se_elo']:.0f} Elo at equal decks, worth "
                 f"{hf['skill_in_sas']:.0f} SAS points.")
    return ("Table 4. Deck strength (SAS) and the Adaptive format", note, head, rows)


# ---------------------------------------------------------------------- output
def cell_text(c) -> str:
    if c is None:
        return ""
    if isinstance(c, tuple):
        v, se = round(c[0]), c[1]
        return f"{v:+d} ± {se:.0f}" if v else f"0 ± {se:.0f}"
    return str(c)


def _plain(c) -> str:
    """Absolute Elo without a sign (tables 1, 2)."""
    if isinstance(c, tuple):
        v, se = c
        return f"{v:.0f}" if se == 0 else f"{v:.0f} ± {se:.0f}"
    return cell_text(c)


def markdown(tables) -> str:
    out = []
    for i, (title, note, head, rows) in enumerate(tables):
        fmt = _plain if i < 2 else cell_text
        out += [f"## {title}", "", note, "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        out += ["| " + " | ".join(fmt(c) for c in r) + " |" for r in rows]
        out.append("")
    return "\n".join(out)


def xlsx(tables, path: str) -> bool:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError:
        return False
    wb = Workbook()
    wb.remove(wb.active)
    F = "Arial"
    for i, (title, note, head, rows) in enumerate(tables):
        ws = wb.create_sheet(f"Table {i + 1}")
        ws["A1"] = title
        ws["A1"].font = Font(name=F, bold=True, size=13)
        ws["A2"] = note
        ws["A2"].font = Font(name=F, italic=True, size=9, color="595959")
        for k, h in enumerate(head, 1):
            c = ws.cell(4, k, h)
            c.font = Font(name=F, bold=True, color="FFFFFF", size=10)
            c.fill = PatternFill("solid", start_color="1F3864")
            c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.row_dimensions[4].height = 45
        for n, r in enumerate(rows, 5):
            for k, c in enumerate(r, 1):
                if isinstance(c, tuple):
                    # the estimate stays a number (sortable); its standard error is shown by the number format
                    v = ws.cell(n, k, round(c[0]))
                    se = f'" ± {c[1]:.0f}"' if c[1] else ""
                    v.number_format = f"+0{se};-0{se};0{se}" if i >= 2 else f"0{se}"
                else:
                    v = ws.cell(n, k, c)
                v.font = Font(name=F, size=10, bold=(k == 1))
                v.alignment = Alignment(horizontal="left" if k == 1 else "center", wrap_text=k <= 2)
        ws.column_dimensions["A"].width = 38
        for k in range(2, len(head) + 1):
            ws.column_dimensions[get_column_letter(k)].width = 30 if (i == 2 and k == 2) else 18
        ws.freeze_panes = "B5"
    wb.save(path)
    return True


def build(results: str):
    L = Ladders(load_ladders(results))
    sas = None
    p = os.path.join(results, "sas", "sas_report.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            sas = json.load(f)
    return [table1(L), table2(L), table3(L), table4(sas)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=os.path.join(HERE, "results"))
    parser.add_argument("--out", default=os.path.join(HERE, "results", "tables"))
    args = parser.parse_args()
    tables = build(args.results)
    os.makedirs(args.out, exist_ok=True)
    md = os.path.join(args.out, "tables.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write(markdown(tables))
    print("wrote", md)
    xl = os.path.join(args.out, "tables.xlsx")
    print("wrote", xl) if xlsx(tables, xl) else print("openpyxl not installed: no tables.xlsx")
    empty = sum(1 for t in tables for r in t[3] for c in r if c is None or c == "")
    print(f"{empty} empty cells")


if __name__ == "__main__":
    main()

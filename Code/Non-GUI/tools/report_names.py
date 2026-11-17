"""Report names for agents, play-time settings and deck sets: the one place a name is chosen.

`name("v1:real-v1#sims=1000")` -> "Real-deck agent · 1,000 sims". Run IDs stay what they are everywhere else (files,
ladders, run.py); only tables written for a reader go through here. "Compact" inputs are network v1 (cards by identity
and a few attributes), "Full" network v2 (adds what each player knows about hidden cards and all static card meaning).
"""

from __future__ import annotations

RUNS = {
    "heuristic": "HeuristicBot",
    "tier0b-a0": "Imitation (compact)",
    "tier0b2-a7": "Imitation (full)",
    "tier2-v1-s1": "Specialist (compact), round 1",
    "tier2-v1-s2": "Specialist (compact), round 2",
    "tier2-v1-s3": "Specialist (compact)",
    "tier2-v2-s1": "Specialist (full), round 1",
    "tier2-v2-s2": "Specialist (full), round 2",
    "tier2-v2-s3": "Specialist (full)",
    "m11-v1": "Generalist (compact)",
    "m11-v2": "Generalist (full)",
    "real-v1": "Real-deck agent",
    "real-v2": "Real-deck agent (full)",
    "size-bc-128": "Imitation, real decks, small net",
    "size-bc-256": "Imitation, real decks, large net",
    "size-128": "Real-deck agent retrained, small net (128×4)",
    "size-256": "Real-deck agent retrained, large net (256×8)",
    "plateau-ctrl": "Real-deck agent + 20k games",
    "plateau-nogate": "Real-deck agent + 20k games, no gating",
    "plateau-sims300": "Real-deck agent + 20k games, 3× training search",
    "plateau-sasw-ctrl": "Real-deck agent + 20k games, SAS-weighted",
    "plateau-sasw-nogate": "Real-deck agent + 20k games, no gating, SAS-weighted",
    "plateau-sasw-sims300": "Real-deck agent + 20k games, 3× training search, SAS-weighted",
    "m9-wt-1": "Specialist, within-turn training, seed 1",
    "m9-wt-2": "Specialist, within-turn training, seed 2",
    "m9-fg-1": "Specialist, full-game training, seed 1",
    "m9-fg-2": "Specialist, full-game training, seed 2",
}

# Play-time options (`#key=value` in a player spec), appended after a dot. Defaults (within-turn search, the student
# leaf, 100 simulations, sampled hidden cards) add nothing.
OPTIONS = {
    ("search", "none"): "no search",
    ("resample", "exact"): "sees hidden cards",
    ("regime", "full_game"): "full-game search",
    ("leaf", "belief_oracle"): "oracle leaf",
}
DEFAULTS = {("regime", "within_turn"), ("leaf", "student"), ("sims", "100")}

DECK_SETS = {
    "fixed": "Fixed decks",
    "random": "Random decks",
    "real-test": "Real decks (test)",
    "real-train": "Real decks (train)",
}


def name(spec: str) -> str:
    """A player spec as `tools.ladder` writes it (`v1:<run>[@final|@p<N>][#k=v,...]`, or a bot) -> its report name."""
    base, _, opts = spec.partition("#")
    run = base.split(":", 1)[-1]
    run, _, at = run.partition("@")
    out = RUNS.get(run, run)
    if at and at != "final":
        out += f" @{at}"
    parts = []
    for kv in filter(None, opts.split(",")):
        k, _, v = kv.partition("=")
        if (k, v) in DEFAULTS:
            continue
        if k == "sims":
            parts.append(f"{int(v):,} sims")
        else:
            parts.append(OPTIONS.get((k, v), f"{k}={v}"))
    return " · ".join([out, *parts])

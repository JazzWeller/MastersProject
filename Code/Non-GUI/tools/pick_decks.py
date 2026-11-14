"""Pick a random pair of real decks whose SAS ratings are close, and write each as a deck JSON the GUI loads.

    python -m tools.pick_decks --out-dir <dir> [--max-diff 3] [--split test] [--seed N]

Decks come from `keyforge/cards/data/decklists/phase3.json` (Decks of KeyForge, https://decksofkeyforge.com).
Prints the two file paths, one per line, then a line describing the pair.
"""

from __future__ import annotations

import argparse
import json
import os
import random

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DECKLIST = os.path.join(HERE, "keyforge", "cards", "data", "decklists", "phase3.json")


def pick(decks, max_diff: float, rng: random.Random):
    """Your deck uniformly at random (among decks with a partner), then the bot's uniformly among the decks within
    `max_diff` SAS of it -- so every SAS is played, not mostly the crowded ones."""
    partners = {id(a): [b for b in decks if b is not a and abs(a["sas"] - b["sas"]) <= max_diff] for a in decks}
    mine = [a for a in decks if partners[id(a)]]
    if not mine:
        raise SystemExit(f"no two decks within {max_diff} SAS")
    a = rng.choice(mine)
    return a, rng.choice(partners[id(a)])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-diff", type=float, default=3.0, help="largest SAS difference allowed")
    ap.add_argument("--split", default="test", choices=["test", "train", "all"], help="test = decks no agent trained on")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)
    with open(DECKLIST, encoding="utf-8") as f:
        decks = [d for d in json.load(f)["decks"] if args.split == "all" or d.get("split") == args.split]
    a, b = pick(decks, args.max_diff, random.Random(args.seed))
    os.makedirs(args.out_dir, exist_ok=True)
    paths = []
    for role, d in (("you", a), ("bot", b)):
        path = os.path.join(args.out_dir, f"{role}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"name": d["name"], "pods": d["pods"]}, f, indent=2, ensure_ascii=False)
        paths.append(path)
    print(paths[0])
    print(paths[1])
    print(f"you: {a['name']} (SAS {a['sas']:g}, {'/'.join(a['pods'])})  |  bot: {b['name']} (SAS {b['sas']:g}, {'/'.join(b['pods'])})")


if __name__ == "__main__":
    main()

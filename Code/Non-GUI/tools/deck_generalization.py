#!/usr/bin/env python3
"""The held-out-deck generalization experiment (Agent Training Plan,
Milestone M11, "the thesis payoff"): behaviour-clone on N random deck
pairs, then play the resulting agent (no search) against HeuristicBot on
random decks **never seen in training**, for several N -- win rate on
held-out decks as a function of the number of training decks.

    python -m tools.deck_generalization --run gen-decks --n-decks 1 4 16 64

Training decks come from a fixed deck set per N (`sim.bc_corpus`'s
`fixed_random` source, seed 1); evaluation decks from a disjoint set (seed
2), so the held-out set is clean by construction. Each N gets its own
corpus directory -- shards are partitioned by deck set, never filtered
after the fact. Writes `deck_generalization.json`/`.md` into the run.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time

import torch

from agent import config as config_mod
from agent.telemetry import Run
from ml.bc_train import train
from ml.dataset import Corpus
from ml.evaluate import evaluate_model
from ml.infer_server import TorchModel
from sim.bc_corpus import fixed_deck_set
from tools.run_screens import stage_data


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--config", default="tier0_bc.json")
    parser.add_argument("--n-decks", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--games", type=int, default=4000, help="training games per N")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--eval-decks", type=int, default=16, help="held-out decks (pairs drawn from them)")
    parser.add_argument("--eval-seeds", type=int, default=25, help="paired seeds per held-out deck pair")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run = Run.create(args.run, args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_path = os.path.join(run.root, "deck_generalization.json")
    report = json.load(open(out_path)) if os.path.exists(out_path) else {"points": {}}
    held = fixed_deck_set(args.eval_decks, deck_seed=2)
    pairs = [(held[k], held[(k + 1) % len(held)]) for k in range(len(held))]
    for n in args.n_decks:
        if str(n) in report["points"]:
            continue
        cfg = copy.deepcopy(run.config)
        pool = dict(cfg["pool"], deck_source="fixed_random", n_decks=n, deck_seed=1)
        cfg["bc"]["epochs"] = args.epochs
        data = stage_data(cfg, args.workers, tag=f"decks-{n}", counts={"heuristic": args.games}, pool=pool)
        model, rep = train(cfg, Corpus.from_dir(data), device=device)
        tm = TorchModel(model, str(device))
        wins = games = 0.0
        t0 = time.time()
        for k, (dx, dy) in enumerate(pairs):
            r = evaluate_model(tm, "heuristic", n_seeds=args.eval_seeds, first_seed=7_000_000 + 1000 * k, deck_x=dx, deck_y=dy)
            wins += r["a_wins"] + 0.5 * r["draws"]
            games += r["games"]
        point = {"n_decks": n, "heldout_score": round(wins / games, 4), "games": int(games), "seconds": round(time.time() - t0, 1),
                 "train_top1_action": rep["top1_by_kind"].get("CHOOSE_ACTION", {}).get("accuracy")}
        report["points"][str(n)] = point
        run.journal("deck_generalization_point", game_index=0, **point)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[M11] {n} training decks -> {point['heldout_score']:.3f} on held-out decks ({point['games']} games)", flush=True)
    lines = ["# Held-out-deck generalization (M11)", "", "| Training decks | Score vs HeuristicBot on held-out decks | games |", "|---|---|---|"]
    for n, p in sorted(report["points"].items(), key=lambda kv: int(kv[0])):
        lines.append(f"| {n} | {p['heldout_score']:.3f} | {p['games']} |")
    with open(os.path.join(run.root, "deck_generalization.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

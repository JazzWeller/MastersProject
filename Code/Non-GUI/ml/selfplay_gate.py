"""Checkpoint gating for self-play (Agent Training Plan, M7), as its own
process so the learner never stops for it.

    python -m ml.selfplay_gate --run wt-s0 --candidate <hash> --best <hash> --number 3 --out <json>

A gate is up to `gate_games` games of search (tens of minutes); run inside
the learner's loop it stopped both training and shard ingest for that long
while the actors kept producing. The learner starts this with the
candidate it just saved, keeps training, and reads the verdict file when
the process exits.

Seeds are a function of the run seed and the gate number -- a gate re-run
after a resume replays the same games -- and never of the clock.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from typing import Optional, Tuple

GATE_SPRT = (0.0, 35.0)  # elo: "no better" vs "~55% against the best"


def gate_verdict(rep) -> Optional[str]:
    """The SPRT's verdict if it decided within the game budget; otherwise a
    fixed-N test at the budget -- promote only if the score clears 0.5 by
    two standard errors. Returns "H1" (promote), "H0" or None."""
    v = rep.sprt(*GATE_SPRT)
    if v is not None:
        return v
    if rep.games and rep.score - 0.5 > 2 * rep.standard_error:
        return "H1"
    return None


def gate_seeds(run_seed: int, number: int, n: int) -> range:
    digest = hashlib.sha256(f"gate|{run_seed}|{number}".encode()).digest()
    first = 10_000_000 + (int.from_bytes(digest[:4], "big") % 1_000_000) * 1_000
    return range(first, first + n)


def run_gate(candidate_path: str, best_path: str, cfg: dict, *, sims: int, mode: str, number: int,
             device: str = "cuda", concurrency: int = 64) -> Tuple[Optional[str], object]:
    """SPRT of the candidate against the best so far, both searching at
    `sims` (search mode) or both search-free (DMC)."""
    from .arena import Player, play_paired
    from .checkpoints import load_model
    from .infer_server import TorchModel

    sp, se = cfg["selfplay"], cfg["search"]
    cand_net, _m, _o = load_model(candidate_path)
    best_net, _m, _o = load_model(best_path)
    cand = TorchModel(cand_net, device)
    incumbent = TorchModel(best_net, device)
    kind = "net" if mode == "dmc" else "search"
    head = "q" if mode == "dmc" else "policy"
    quiet = bool(se.get("quiet_leaves", False))
    a = Player("candidate", kind, cand, regime=se["regime"], leaf=se["leaf"], simulations=sims, resample=se["resample"],
               mode=head, quiet_leaves=quiet)
    b = Player("best", kind, incumbent, regime=se["regime"], leaf=se["leaf"], simulations=sims, resample=se["resample"],
               mode=head, quiet_leaves=quiet)
    seeds = gate_seeds(int(cfg["seed"]), number, max(1, int(sp["gate_games"]) // 4))
    rep = play_paired(a, b, seeds, cfg["pool"]["decks"][0], cfg["pool"]["decks"][1],
                      max_turns=200, concurrency=concurrency, sprt=GATE_SPRT, min_games=40)
    return gate_verdict(rep), rep


def main():
    from agent.lifecycle import exit_with_parent
    from agent.telemetry import Run

    from .checkpoints import CheckpointStore

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--best", required=True)
    parser.add_argument("--number", type=int, required=True)
    parser.add_argument("--sims", type=int, default=50)
    parser.add_argument("--mode", default="search")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    exit_with_parent()

    run = Run.open(args.run)
    store = CheckpointStore(os.path.join(run.root, "checkpoints"))
    t0 = time.time()
    verdict, rep = run_gate(store.path_of(args.candidate), store.path_of(args.best), run.config, sims=args.sims,
                            mode=args.mode, number=args.number, device=args.device)
    result = {
        "number": args.number, "candidate": args.candidate, "best": args.best, "verdict": verdict,
        "score": rep.score, "games": rep.games, "seconds": round(time.time() - t0, 1), "report": rep.summary(),
    }
    with open(args.out + ".tmp", "w", encoding="utf-8") as f:
        json.dump(result, f)
    os.replace(args.out + ".tmp", args.out)


if __name__ == "__main__":
    main()

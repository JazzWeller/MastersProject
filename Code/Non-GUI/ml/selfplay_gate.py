"""Checkpoint gating for self-play (Agent Training Plan, M7), as its own
process so the learner never stops for it.

    python -m ml.selfplay_gate --run wt-s0 --candidate <hash> --best <hash> --number 3 --out <json>

A gate is up to `gate_games` games of search, over `selfplay.gate_workers`
processes (`run_gate`). Run inside the learner's loop it stopped both
training and shard ingest for that long while the actors kept producing. The learner starts this with the
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

GATE_SPRT = (0.0, 35.0)  # elo: "no better" vs "~55% against the best" (the default: `selfplay.gate_sprt`)
GATE_FALLBACK_Z = 2.0  # the fixed-N test's standard errors (`selfplay.gate_fallback_z`)


def gate_rule(cfg: dict) -> Tuple[Tuple[float, float], float]:
    """A run's (SPRT elo bounds, fixed-N z). Tier 2 round 4 gained ~10-20
    Elo per 1,000 steps: against (0, 35) and z 2, 15 of v1's 20 gates ended
    undecided at 0.50-0.55, and the actors played a stale network."""
    sp = cfg.get("selfplay", {})
    elo0, elo1 = sp.get("gate_sprt") or GATE_SPRT
    return (float(elo0), float(elo1)), float(sp.get("gate_fallback_z", GATE_FALLBACK_Z))


def gate_verdict(rep, sprt: Tuple[float, float] = GATE_SPRT, z: float = GATE_FALLBACK_Z) -> Optional[str]:
    """The SPRT's verdict if it decided within the game budget; otherwise a
    fixed-N test at the budget -- promote only if the score clears 0.5 by
    `z` standard errors. Returns "H1" (promote), "H0" or None."""
    v = rep.sprt(*sprt)
    if v is not None:
        return v
    if rep.games and rep.score - 0.5 > z * rep.standard_error:
        return "H1"
    return None


def gate_seeds(run_seed: int, number: int, n: int) -> range:
    digest = hashlib.sha256(f"gate|{run_seed}|{number}".encode()).digest()
    first = 10_000_000 + (int.from_bytes(digest[:4], "big") % 1_000_000) * 1_000
    return range(first, first + n)


def _players(candidate_path: str, best_path: str, cfg: dict, sims: int, mode: str, device: str):
    """The candidate and the incumbent as arena players, both searching at
    `sims` (search mode) or both search-free (DMC)."""
    from .arena import Player

    se = cfg["search"]
    v2 = int(cfg.get("observation", {}).get("version", 1)) == 2
    extra = {}
    if v2:  # network v2: both players search with v2 leaves, in the run's determinization
        from agent.search.leaf_v2 import history_form

        from .checkpoints import load_model_v2
        from .infer_v2 import TorchModelV2

        cand_net, _m = load_model_v2(candidate_path)
        best_net, _m = load_model_v2(best_path)
        cand, incumbent = TorchModelV2(cand_net, device), TorchModelV2(best_net, device)
        extra = {"leaf": "student_v2", "history": history_form(cand_net.history_arch),
                 "determinization": se.get("determinization", "chance_exact")}
    else:
        from .checkpoints import load_model
        from .infer_server import TorchModel

        cand_net, _m, _o = load_model(candidate_path)
        best_net, _m, _o = load_model(best_path)
        cand, incumbent = TorchModel(cand_net, device), TorchModel(best_net, device)
        extra = {"leaf": se["leaf"]}
    kind = "net" if mode == "dmc" else "search"
    head = "q" if mode == "dmc" else "policy"
    quiet = bool(se.get("quiet_leaves", False))
    a = Player("candidate", kind, cand, regime=se["regime"], simulations=sims, resample=se["resample"],
               mode=head, quiet_leaves=quiet, **extra)
    b = Player("best", kind, incumbent, regime=se["regime"], simulations=sims, resample=se["resample"],
               mode=head, quiet_leaves=quiet, **extra)
    return a, b


_WORKER = None  # (candidate, best, pool) in a gate worker process
UNIT_SEEDS = 4  # seeds per unit of work in a parallel gate (16 games; the SPRT is read after each)


def _worker_init(spec) -> None:
    global _WORKER
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the gate process decides when to stop
    candidate_path, best_path, cfg, sims, mode, device = spec
    _WORKER = (*_players(candidate_path, best_path, cfg, sims, mode, device), cfg["pool"])


def unit_decks(pool: dict, first_seed: int):
    """A unit's deck pair: the run's two decks, or (any other deck source,
    M11) a pair drawn from the unit's first seed, the same on every replay."""
    import random

    from sim.bc_corpus import decks_for

    if pool.get("deck_source", "presets") == "presets":
        return tuple(pool["decks"])
    return decks_for(pool, 0, random.Random(f"gate-decks|{first_seed}"))


def _play_unit(a, b, pool: dict, seeds):
    from .arena import play_paired

    dx, dy = unit_decks(pool, seeds[0])
    return play_paired(a, b, seeds, dx, dy, max_turns=200, concurrency=4 * len(seeds))


def _worker_unit(seeds):
    a, b, pool = _WORKER
    return _play_unit(a, b, pool, seeds)


def merge_reports(into, rep) -> None:
    """Adds one paired report's games into another (whole seeds both)."""
    for k in ("games", "a_wins", "b_wins", "draws", "forfeits"):
        setattr(into, k, getattr(into, k) + getattr(rep, k))
    for mine, theirs in ((into.by_seat, rep.by_seat), (into.by_deck, rep.by_deck)):
        for key, (w, n) in theirs.items():
            cur = mine.setdefault(key, [0, 0])
            cur[0] += w
            cur[1] += n
    into.turns.extend(rep.turns)


def run_gate(candidate_path: str, best_path: str, cfg: dict, *, sims: int, mode: str, number: int,
             device: str = "cuda", concurrency: int = 64, workers: int = 1) -> Tuple[Optional[str], object]:
    """SPRT of the candidate against the best so far (`gate_rule`), the
    seeds played `UNIT_SEEDS` at a time -- each unit on its own deck pair
    when the pool isn't two fixed decks -- and the test read after each
    finished unit, on whole seeds only as in `play_paired`. With `workers`
    > 1 (`selfplay.gate_workers`) the units run over that many processes: a
    gate beside the actors was one process among ~30 and took 1-4 h in Tier
    2 round 3."""
    from sim.paired import PairedReport

    sp = cfg["selfplay"]
    sprt, z = gate_rule(cfg)
    seeds = list(gate_seeds(int(cfg["seed"]), number, max(1, int(sp["gate_games"]) // 4)))
    # One process plays bigger units, so as many games share its GPU batches as before (`concurrency`).
    size = UNIT_SEEDS if workers > 1 else max(UNIT_SEEDS, concurrency // 4)
    units = [seeds[k:k + size] for k in range(0, len(seeds), size)]
    total = PairedReport(games=0, a_wins=0, b_wins=0, draws=0, forfeits=0)

    def decided() -> bool:
        return total.games >= 40 and total.sprt(*sprt) is not None

    if workers <= 1:
        a, b = _players(candidate_path, best_path, cfg, sims, mode, device)
        for unit in units:
            merge_reports(total, _play_unit(a, b, cfg["pool"], unit))
            if decided():
                break
        return gate_verdict(total, sprt, z), total
    import multiprocessing as mp

    spec = (candidate_path, best_path, cfg, sims, mode, device)
    with mp.get_context("spawn").Pool(min(workers, len(units)), initializer=_worker_init, initargs=(spec,)) as pool:
        for rep in pool.imap_unordered(_worker_unit, units):  # leaving the `with` stops the rest
            merge_reports(total, rep)
            if decided():
                break
    return gate_verdict(total, sprt, z), total


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
                            mode=args.mode, number=args.number, device=args.device,
                            workers=int(run.config["selfplay"].get("gate_workers", 1)))
    result = {
        "number": args.number, "candidate": args.candidate, "best": args.best, "verdict": verdict,
        "score": rep.score, "games": rep.games, "seconds": round(time.time() - t0, 1), "report": rep.summary(),
    }
    with open(args.out + ".tmp", "w", encoding="utf-8") as f:
        json.dump(result, f)
    os.replace(args.out + ".tmp", args.out)


if __name__ == "__main__":
    main()

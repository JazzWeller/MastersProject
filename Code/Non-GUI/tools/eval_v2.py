#!/usr/bin/env python3
"""Search strength of trained networks, v2 or v1, on paired seeds (Agent
Observation Plan, O10's search rungs). WSL, CUDA.

    python -m tools.eval_v2 --a v2:tier0b2-a8 --b heuristic --sims 100 --seeds 100 --workers 8
    python -m tools.eval_v2 --a v2:tier0b2-a6 --b v1:tier0b-a0 --sims 100 --seeds 100

A player is `v2:<run id>` (its network-v2 checkpoint -- a BC run's, or a
self-play run's best promoted one; `<run id>@final` its last -- searched with the
`student_v2` leaf in its history's request form), `v1:<run id>` (a v1
checkpoint, the `student` leaf) or a bot name (`heuristic`). Both searchers
use the same regime, simulations and O3's `chance_exact` determinization.
`--a-exact` hands player a's search the true hidden state instead (the G3
diagnostic: its score minus the honest player's on the same seeds is the
hidden-information gap).
Each seed is played in both seat orders and both deck assignments
(`ml.arena.play_paired`); seeds are played `--unit` at a time over
`--workers` processes, each with its own copy of the networks on the GPU.

A network player may end in `#key=value,...` to play differently from the
command line's settings (the M9 matrix and its diagnostics):
`regime=within_turn|full_game`, `leaf=student|belief_oracle` (v1),
`sims=<n>`, `resample=exact` (the true hidden state), or `search=none`
(v1: the network's policy alone, no search) -- e.g.
`v1:m9-fg-1#regime=within_turn,leaf=belief_oracle`.

`--deck-set N` plays each unit of seeds on one of N random deck pairs from
a fixed deck set (`sim.bc_corpus.fixed_deck_set`, seed `--deck-seed`, 2 by
default: never a training deck, M11) instead of Fignor and Igor.

Pausable: each finished unit is appended to `<out>.partial.jsonl`, so a run
stopped any way (Ctrl-C, SIGTERM, `tools/pause_training.sh`) and started
again with the same arguments plays only the units still missing. `--out`
defaults to `$KEYFORGE_DATA/evals/<a>__<b>__<sims>sims_<seeds>x<first seed>.json`.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time


def _checkpoint(run_id: str) -> str:
    """A run's network: a BC run's (`bc_report.json`), or a self-play run's
    best promoted one (`best.json`); `<run>@final` is a self-play run's last
    learner checkpoint, promoted or not, and `<run>@p<N>` its N-th promoted
    one (`@p0`: the warm start)."""
    from sim import data_root

    run_id, _, which = run_id.partition("@")
    root = data_root.resolve(os.path.join("runs", run_id))
    if which == "final":
        with open(os.path.join(root, "journal.jsonl"), "r", encoding="utf-8") as f:
            ckpt = [e for e in map(json.loads, f) if e["event"] == "run_finished"][-1]["final"]
    elif which.startswith("p") and which[1:].isdigit():
        with open(os.path.join(root, "journal.jsonl"), "r", encoding="utf-8") as f:
            promoted = [e["checkpoint"] for e in map(json.loads, f) if e["event"] == "promoted"]
        if int(which[1:]) >= len(promoted):
            raise FileNotFoundError(f"{run_id} has {len(promoted) - 1} promotions, not {which[1:]}")
        ckpt = promoted[int(which[1:])]
    elif os.path.exists(os.path.join(root, "bc_report.json")):
        with open(os.path.join(root, "bc_report.json"), "r", encoding="utf-8") as f:
            ckpt = json.load(f)["checkpoint"]
    else:
        with open(os.path.join(root, "best.json"), "r", encoding="utf-8") as f:
            ckpt = json.load(f)["checkpoint"]
    store = os.path.join(root, "checkpoints")
    return next(os.path.join(store, n) for n in os.listdir(store) if n.startswith(ckpt))


def _resample(args, name: str) -> str:
    return "exact" if name == "a" and getattr(args, "a_exact", False) else "all"


PLAYER_OPTIONS = ("regime", "leaf", "sims", "resample", "search")


def player_options(spec: str):
    """`kind:run#k=v,...` -> (`kind:run`, {k: v})."""
    base, _, rest = spec.partition("#")
    opts = dict(kv.split("=", 1) for kv in rest.split(",") if kv)
    unknown = set(opts) - set(PLAYER_OPTIONS)
    if unknown:
        raise ValueError(f"player {spec!r}: unknown options {sorted(unknown)} (have {PLAYER_OPTIONS})")
    return base, opts


def _player(spec: str, args, name: str):
    from ml.arena import Player

    if ":" not in spec:
        return Player(name=name, kind="bot", bot=spec)
    base, opts = player_options(spec)
    kind, run_id = base.split(":", 1)
    regime = opts.get("regime", args.regime)
    sims = int(opts.get("sims", args.sims))
    resample = opts.get("resample", _resample(args, name))
    if kind == "v2":
        from agent.search.leaf_v2 import history_form
        from ml.checkpoints import load_model_v2
        from ml.infer_v2 import TorchModelV2

        if opts.get("search") == "none" or opts.get("leaf", "student") != "student":
            raise ValueError(f"{spec!r}: v2 players search with the student leaf only")
        net, _meta = load_model_v2(_checkpoint(run_id))
        return Player(name=name, kind="search", model=TorchModelV2(net, "cuda", compile=False), regime=regime,
                      leaf="student_v2", history=history_form(net.history_arch), simulations=sims,
                      determinization="chance_exact", resample=resample)
    if kind == "v1":
        from ml.infer_server import load_torch_model

        model = load_torch_model(_checkpoint(run_id), "cuda")
        if opts.get("search") == "none":
            return Player(name=name, kind="net", model=model, mode="policy")
        return Player(name=name, kind="search", model=model, regime=regime, leaf=opts.get("leaf", "student"),
                      simulations=sims, determinization="chance_exact", resample=resample)
    raise ValueError(f"unknown player {spec!r} (v2:<run> | v1:<run> | a bot name)")


_PLAYERS = None


def _init(args):
    global _PLAYERS
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the parent decides when to stop
    _PLAYERS = (_player(args.a, args, "a"), _player(args.b, args, "b"))


def _work(task):
    seeds, args = task
    from ml.arena import play_paired

    a, b = _PLAYERS
    rep = play_paired(a, b, seeds, *unit_decks(args, seeds[0]), max_turns=args.max_turns, concurrency=args.concurrency)
    return {"seeds": list(seeds), "games": rep.games, "a_wins": rep.a_wins, "b_wins": rep.b_wins, "draws": rep.draws,
            "forfeits": rep.forfeits, "turns": rep.turns, "by_seat": {str(k): v for k, v in rep.by_seat.items()}}


def unit_decks(args, first_seed: int):
    """The deck pair a unit starting at `first_seed` plays: Fignor and Igor,
    or (`--deck-set N`) the unit's pair from a fixed set of N random decks
    -- a function of the seed alone, so a resumed evaluation and every
    player pairing meet the same decks."""
    if not getattr(args, "deck_set", 0):
        return "fignor", "igor"
    from sim.bc_corpus import fixed_deck_set

    decks = fixed_deck_set(args.deck_set, deck_seed=args.deck_seed)
    k = (first_seed - args.first_seed) // args.unit
    return decks[k % len(decks)], decks[(k + 1 + k // len(decks)) % len(decks)]


def out_name(a: str, b: str, sims: int, seeds: int, first_seed: int, a_exact: bool = False, deck_set: int = 0,
             deck_seed: int = 2) -> str:
    """The default output file's name (also how `tools.ladder` finds a match)."""
    name = f"{a}{'-exact' if a_exact else ''}__{b}__{sims}sims_{seeds}x{first_seed}"
    if deck_set:
        name += f"_decks{deck_set}s{deck_seed}"
    return name.replace(":", "-") + ".json"


def _default_out(args) -> str:
    from sim import data_root

    return data_root.resolve(os.path.join("evals", out_name(args.a, args.b, args.sims, args.seeds, args.first_seed,
                                                            args.a_exact, args.deck_set, args.deck_seed)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", default="heuristic")
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--regime", default="within_turn", choices=("within_turn", "full_game"))
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--first-seed", type=int, default=3_000_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=16, help="games in flight per worker")
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--a-exact", action="store_true",
                        help="player a searches the true hidden state (G3: its score minus the honest one's on the "
                             "same seeds is the hidden-information gap)")
    parser.add_argument("--deck-set", type=int, default=0, help="play on N held-out random deck pairs (M11)")
    parser.add_argument("--deck-seed", type=int, default=2, help="the deck set's seed (training sets use others)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--unit", type=int, default=4, help="seeds per unit of work (the pause granularity)")
    args = parser.parse_args()
    out = args.out or _default_out(args)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    partial = out + ".partial.jsonl"
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    parts = []
    if os.path.exists(partial):
        with open(partial, "r", encoding="utf-8") as f:
            parts = [json.loads(line) for line in f if line.strip()]
    done = {s for p in parts for s in p["seeds"]}
    todo = [s for s in seeds if s not in done]
    units = [todo[k:k + args.unit] for k in range(0, len(todo), args.unit)]
    if done:
        print(f"resuming: {len(seeds) - len(todo)} of {len(seeds)} seeds already played ({partial})", flush=True)
    import signal

    def on_term(_signum, _frame):  # leave through the pool's `with` (it terminates the workers)
        raise SystemExit(75)

    signal.signal(signal.SIGTERM, on_term)
    t0 = time.time()
    if units:
        with mp.get_context("spawn").Pool(min(args.workers, len(units)), initializer=_init, initargs=(args,)) as pool:
            with open(partial, "a", encoding="utf-8") as f:
                for p in pool.imap_unordered(_work, [(u, args) for u in units]):
                    f.write(json.dumps(p) + "\n")
                    f.flush()
                    parts.append(p)
    tot = {k: sum(p[k] for p in parts) for k in ("games", "a_wins", "b_wins", "draws", "forfeits")}
    score = (tot["a_wins"] + 0.5 * tot["draws"]) / tot["games"] if tot["games"] else 0.0
    from sim.parallel_eval import wilson_interval

    lo, hi = wilson_interval(tot["a_wins"] + 0.5 * tot["draws"], tot["games"])
    turns = [t for p in parts for t in p["turns"]]
    res = dict(tot, a=args.a, a_exact=bool(args.a_exact), b=args.b, sims=args.sims, regime=args.regime, seeds=args.seeds,
               deck_set=args.deck_set, deck_seed=args.deck_seed if args.deck_set else None, a_score=round(score, 4),
               interval95=[round(lo, 4), round(hi, 4)], mean_turns=round(sum(turns) / len(turns), 2) if turns else None,
               seconds=round(time.time() - t0, 1))
    print(json.dumps(res, indent=1))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    os.remove(partial)
    print(f"written to {out}")


if __name__ == "__main__":
    main()

"""Agent Observation Plan, Milestone O3: the determinization acceptance
checks, over fuzz games.

    python -m tools.o3_acceptance --games 200 --calibration 400 [--workers 8]

- **Constraints** (`--games` per pool, the pools of tools/o1_acceptance.py):
  every world drawn by `constrained`, `chance_exact` and `belief` (random
  weights) satisfies the viewer's knowledge -- each card's mask, each zone's
  count, each known deck position -- and leaves the viewer's projection
  unchanged; the opponent's tracker inside the world is sound; and how
  often `relabel` is accepted.
- **Calibration** (`--calibration` games, both players epsilon-greedy
  HeuristicBots, as in the BC corpus): the log-loss of `chance_exact`'s
  P(hand) for every opponent card in a hidden zone at every decision,
  against the uniform prior |hand| / |hidden| (the plan's 0.541).
- **Cost**: microseconds per sample, against a replay fork.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from typing import List

from keyforge.determinize import chance_filter, sample_world, apply_world
from keyforge.enums import Resample
from keyforge.game import Game
from keyforge.knowledge import tracker_for
from tools.o1_acceptance import POOLS, bots_for, game_config

SAMPLERS = ("constrained", "chance_exact", "belief")


def _where(game):
    from tests._journal_check import actual_locations

    return actual_locations(game)


def check_world(game, v: int, world, label: str) -> List[str]:
    """`world` (a determinized fork for `v`) against what `v` knows in
    `game`."""
    problems = []
    t = tracker_for(game, v)
    where = _where(world)
    for iid, m in t.mask.items():
        if where[iid] not in m:
            problems.append(f"{label}: card {iid} is in {where[iid]}, outside its mask {sorted(m, key=repr)}")
    for pid in (1, 2):
        deck = [c.instance_id for c in world.players[pid].deck.cards()]
        for i, x in enumerate(t.top[pid]):
            if x is not None and (i >= len(deck) or deck[i] != x):
                problems.append(f"{label}: deck {pid} top should be {t.top[pid]}, is {deck[:len(t.top[pid])]}")
                break
        b = t.bottom[pid]
        for i, x in enumerate(b):
            if x is not None and deck[len(deck) - len(b) + i] != x:
                problems.append(f"{label}: deck {pid} bottom should be {b}")
                break
        for kind in ("hand", "archive", "deck"):
            n_true = len(getattr(game.players[pid], kind))
            n_world = len(getattr(world.players[pid], kind))
            if n_true != n_world:
                problems.append(f"{label}: {kind} {pid} has {n_world} cards, should have {n_true}")
    if world.projected(v) != game.projected(v):
        problems.append(f"{label}: the viewer's projection changed")
    o = 3 - v
    to = tracker_for(world, o)
    for iid, m in to.mask.items():
        if where[iid] not in m:
            problems.append(f"{label}: inside the world, the opponent's tracker has card {iid} outside its mask")
            break
    return problems


def check_game(config, i: int, every: int = 10, stats=None) -> List[str]:
    game = Game(config)
    bots = bots_for(i, config.seed)
    n = 0
    rng = random.Random(config.seed)
    while not game.is_over:
        if n % every == 0 and game.copy_anywhere:
            for v in (1, 2):
                for sampler in SAMPLERS:
                    weights = None
                    if sampler == "belief":
                        w = {iid: rng.random() for iid in game._cards_by_id}
                        weights = w.get
                    history = "relabel" if sampler == "chance_exact" else "drop"
                    world = game.fork_determinized(v, random.Random(rng.random()), Resample.ALL, backend="copy",
                                                   sampler=sampler, weights=weights, history=history)
                    if stats is not None and history == "relabel":
                        stats["relabel_tried"] += 1
                        stats["relabel_ok"] += "_world_cut" not in world.__dict__
                    problems = check_world(game, v, world, f"{sampler} for viewer {v} after {n} decisions")
                    if problems:
                        return problems
                    if stats is not None:
                        stats["worlds"] += 1
        d = game.pending_decision
        game.submit(bots[d.player].decide(game.view_for(d.player), d))
        n += 1
    return []


def calibration_game(i: int):
    """(n, chance_exact log-loss sum, uniform log-loss sum) over one game."""
    from sim.bc_corpus import EpsilonGreedy
    from keyforge.cards.decks import DECKS, random_deck

    rng = random.Random(f"calibration-{i}")
    presets = sorted(DECKS)
    decks = (rng.choice(presets), rng.choice(presets)) if i % 2 else (random_deck(rng, "R1"), random_deck(rng, "R2"))
    from keyforge.config import GameConfig

    game = Game(GameConfig(decks=decks, seed=rng.getrandbits(31), max_turns=120))
    bots = {1: EpsilonGreedy(rng.getrandbits(31), 0.2), 2: EpsilonGreedy(rng.getrandbits(31), 0.2)}
    n = 0
    ll_c = ll_u = 0.0
    eps = 1e-6
    while not game.is_over:
        d = game.pending_decision
        v = d.player
        o = 3 - v
        p = game.players[o]
        hidden = p.hand.cards() + p.archive.cards() + p.deck.cards()
        if hidden:
            f = chance_filter(game, v)
            hand = {c.instance_id for c in p.hand.cards()}
            u = len(hand) / len(hidden)
            for c in hidden:
                pz = f.p_zone(c.instance_id)
                q = pz[0] if pz is not None else u
                y = c.instance_id in hand
                q = min(max(q, eps), 1 - eps)
                uu = min(max(u, eps), 1 - eps)
                ll_c -= math.log(q) if y else math.log(1 - q)
                ll_u -= math.log(uu) if y else math.log(1 - uu)
                n += 1
        game.submit(bots[v].decide(game.view_for(v), d))
    return n, ll_c, ll_u


def cost(n_positions: int = 20) -> dict:
    out = {s: [] for s in ("uniform",) + SAMPLERS}
    fork = []
    for i in range(n_positions):
        config = game_config("random", 500 + i)
        game = Game(config)
        bots = bots_for(i, config.seed)
        for _ in range(60 + 7 * i):
            if game.is_over:
                break
            d = game.pending_decision
            game.submit(bots[d.player].decide(game.view_for(d.player), d))
        if game.is_over:
            continue
        rng = random.Random(i)
        tracker_for(game, 1)
        chance_filter(game, 1)
        for s in out:
            t0 = time.perf_counter()
            for _ in range(50):
                if s == "uniform":
                    w = game.copy()
                    w._redeal(1, rng, Resample.ALL, "uniform")
                else:
                    sample_world(game, 1, rng, Resample.ALL, s, (lambda x: 1.0) if s == "belief" else None)
            dt = (time.perf_counter() - t0) / 50
            out[s].append(dt)
        t0 = time.perf_counter()
        game.fork_by_replay()
        fork.append(time.perf_counter() - t0)
    mean = lambda xs: sum(xs) / len(xs) * 1e6 if xs else 0
    return {"replay_fork_us": round(mean(fork)), **{f"{s}_us": round(mean(v)) for s, v in out.items()}}


def _check_worker(args):
    pool, i, every = args
    stats = {"worlds": 0, "relabel_tried": 0, "relabel_ok": 0}
    try:
        problems = check_game(game_config(pool, i), i, every, stats)
    except Exception as ex:
        import traceback
        problems = [f"raised {type(ex).__name__}: {ex}\n{traceback.format_exc()}"]
    return pool, i, problems, stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--games", type=int, default=20, help="constraint games per pool")
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--calibration", type=int, default=100, help="calibration games")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)
    t0 = time.time()
    pool_map = map
    p = None
    if args.workers > 1:
        import multiprocessing as mp
        p = mp.Pool(args.workers)
        pool_map = lambda f, xs: p.imap_unordered(f, xs, chunksize=2)
    failures = 0
    totals = {"worlds": 0, "relabel_tried": 0, "relabel_ok": 0}
    jobs = [(pool, i, args.every) for pool in POOLS for i in range(args.games)]
    for pool, i, problems, stats in pool_map(_check_worker, jobs):
        for k in totals:
            totals[k] += stats[k]
        if problems:
            failures += 1
            print(f"{pool} game {i}:", *problems[:3], sep="\n  ", flush=True)
    print(f"constraints: {len(jobs)} games, {totals['worlds']} worlds, {failures} games with problems; "
          f"relabel accepted {totals['relabel_ok']} of {totals['relabel_tried']}")
    n = ll_c = ll_u = 0
    for k, (dn, dc, du) in enumerate(pool_map(calibration_game, range(args.calibration))):
        n, ll_c, ll_u = n + dn, ll_c + dc, ll_u + du
    if n:
        print(f"calibration: {n} card-positions over {args.calibration} games: P(hand) log-loss chance_exact "
              f"{ll_c / n:.4f}, uniform {ll_u / n:.4f}")
    print("cost:", cost())
    print(f"{time.time() - t0:.0f} s")
    if p is not None:
        p.close()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

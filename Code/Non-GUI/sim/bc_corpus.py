"""The behaviour-cloning corpus (Agent Training Plan, Milestone M3): replay
records with teacher labels.

Three sources, per the plan: `heuristic` (HeuristicBot mirror games -- the
imitation target), `epsilon` (an epsilon-greedy HeuristicBot -- state
coverage the pure heuristic never visits) and `random` (RandomBot -- the
near-random states early self-play actually occupies). In every source the
**label** at a decision is what HeuristicBot would choose there; only the
*behaviour* that moves the game on differs. For `heuristic` the two
coincide, so its records carry no separate label list.

Records, not tensors (the plan's storage trade-off): each line is a replay
record -- the resolved config (full decklists, version-stamped), the
behaviour's choice record, the labels, the outcome -- a few KB per game,
regenerated in seconds and never stale when the encoder changes.
`ml/dataset.py` turns them into encoded training shards.

Deck sources: `presets` (the config's two decks, seat order alternating by
game), `random` (a fresh random legal deck per seat per game, optionally
with some cards held out entirely -- M3 Screen 5) and `alliance`.

Run from Code/Non-GUI: `python -m sim.bc_corpus --config tier0_bc_smoke.json
--out bc/smoke` (a relative `--out` lands under $KEYFORGE_DATA).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from bots.base import Controller
from bots.heuristic_bot import HeuristicBot
from bots.random_bot import RandomBot
from keyforge.cards.card_data import CARD_DEFS
from keyforge.cards.decks import ALL_HOUSES, CARDS_PER_POD, PODS_PER_DECK, Deck, build_alliance_deck, resolve_deck
from keyforge.config import GameConfig
from keyforge.game import Game
from keyforge.replay import config_to_dict, encode_choice
from sim import data_root

SOURCES = ("heuristic", "epsilon", "random")
GAMES_PER_SHARD = 250


def game_seed(run_seed: int, source: str, index: int) -> int:
    digest = hashlib.sha256(f"bc|{run_seed}|{source}|{index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") >> 1


class EpsilonGreedy(Controller):
    """HeuristicBot, except with probability `epsilon` a uniformly random
    legal choice instead."""

    def __init__(self, seed: int, epsilon: float):
        self.heuristic = HeuristicBot(seed=seed)
        self.random = RandomBot(seed=seed + 7)
        self.rng = random.Random(seed ^ 0x5EED)
        self.epsilon = epsilon

    def decide(self, view, decision, budget=None, capability=None):
        if self.rng.random() < self.epsilon:
            return self.random.decide(view, decision)
        return self.heuristic.decide(view, decision)


def _names_by_house(excluded: frozenset) -> Dict:
    out: Dict = {}
    for name, cdef in CARD_DEFS.items():
        if name not in excluded:
            out.setdefault(cdef.house, []).append(name)
    return out


def restricted_random_deck(rng: random.Random, name: str, excluded: frozenset = frozenset(), required: frozenset = frozenset()) -> Deck:
    """A random legal deck (3 houses, 12 cards each, with replacement) that
    never contains an `excluded` card -- and, if `required` is given,
    contains at least one card from it (redrawn until it does; used for
    the held-out evaluation decks of M3 Screen 5)."""
    pool = _names_by_house(excluded)
    houses = [h for h in ALL_HOUSES if pool.get(h)]
    for _ in range(1000):
        chosen = rng.sample(houses, PODS_PER_DECK)
        pods = {h: [rng.choice(pool[h]) for _ in range(CARDS_PER_POD)] for h in chosen}
        if not required or any(n in required for names in pods.values() for n in names):
            return Deck(name=name, pods=pods)
    raise ValueError("restricted_random_deck: no deck with a required card in 1000 draws")


def fixed_deck_set(n: int, deck_seed=0, excluded: frozenset = frozenset()) -> list:
    """`n` random legal decks, reproducible from `deck_seed` -- a deck set
    with a different seed never shares a deck with this one except by
    coincidence, which is what makes a held-out evaluation set clean by
    construction (M11)."""
    return [restricted_random_deck(random.Random(f"deckset|{deck_seed}|{k}"), f"D{deck_seed}-{k}", excluded) for k in range(n)]


def held_out_cards(seed: int, fraction: float = 0.2) -> frozenset:
    """A reproducible set of cards held out of training for Screen 5."""
    names = sorted(CARD_DEFS)
    rng = random.Random(f"heldout|{seed}")
    k = int(round(fraction * len(names)))
    return frozenset(rng.sample(names, k))


def _decks_for(pool: dict, index: int, rng: random.Random, excluded: frozenset, required: frozenset) -> Tuple:
    source = pool.get("deck_source", "presets")
    if source == "presets":
        a, b = pool["decks"]
        return (a, b) if index % 2 == 0 else (b, a)
    if source == "random":
        return (
            restricted_random_deck(rng, "R1", excluded, required),
            restricted_random_deck(rng, "R2", excluded, required),
        )
    if source == "fixed_random":
        # M11's held-out-deck experiment: games only ever use decks from a
        # fixed, reproducible set of `n_decks` random decks.
        deck_set = fixed_deck_set(int(pool["n_decks"]), pool.get("deck_seed", 0), excluded)
        i, j = rng.randrange(len(deck_set)), rng.randrange(len(deck_set))
        return deck_set[i], deck_set[j]
    if source == "alliance":
        presets = list(pool.get("alliance_sources") or ["fignor", "igor", "stonewall", "starfall", "vigil", "thornwood"])
        def one(label):
            houses = rng.sample(ALL_HOUSES, PODS_PER_DECK)
            srcs = {}
            for h in houses:
                having = [p for p in presets if h in resolve_deck(p).pods]
                if not having:
                    return None
                srcs[h] = rng.choice(having)
            return build_alliance_deck(label, srcs)
        decks = []
        while len(decks) < 2:
            d = one(f"A{len(decks) + 1}")
            if d is not None:
                decks.append(d)
        return tuple(decks)
    raise ValueError(f"unknown deck_source {source!r}")


def play_labelled(config: GameConfig, source: str, seed: int, epsilon: float = 0.1) -> dict:
    game = Game(config)
    teachers = {pid: HeuristicBot(seed=seed + pid) for pid in (1, 2)}
    if source == "heuristic":
        behaviour = teachers
    elif source == "epsilon":
        behaviour = {pid: EpsilonGreedy(seed + 11 * pid, epsilon) for pid in (1, 2)}
    elif source == "random":
        behaviour = {pid: RandomBot(seed=seed + 13 * pid) for pid in (1, 2)}
    else:
        raise ValueError(f"unknown source {source!r}")
    labels: Optional[List] = None if source == "heuristic" else []
    while not game.is_over:
        d = game.pending_decision
        view = game.view_for(d.player)
        if labels is None:
            choice = teachers[d.player].decide(view, d)
        else:
            labels.append(encode_choice(d, teachers[d.player].decide(view, d)))
            choice = behaviour[d.player].decide(view, d)
        game.submit(choice)
    return {
        "source": source,
        "seed": seed,
        "config": config_to_dict(config),
        "record": game.choice_record,
        "labels": labels,
        "outcome": {"1": game.outcome_for(1), "2": game.outcome_for(2)},
        "turns": game.turn_number,
        "reason": (game.result or {}).get("reason"),
    }


def _shard_task(args) -> Tuple[str, int, float]:
    path, source, start, count, run_seed, pool, epsilon, excluded, required = args
    if os.path.exists(path):
        return path, 0, 0.0
    t0 = time.perf_counter()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for index in range(start, start + count):
            seed = game_seed(run_seed, source, index)
            rng = random.Random(seed)
            decks = _decks_for(pool, index, rng, frozenset(excluded), frozenset(required))
            config = GameConfig(decks=decks, seed=seed, max_turns=pool.get("max_turns", 200))
            rec = play_labelled(config, source, seed, epsilon)
            rec["index"] = index
            f.write(json.dumps(rec, separators=(",", ":")))
            f.write("\n")
    os.replace(tmp, path)
    return path, count, time.perf_counter() - t0


def generate(
    out_dir: str, counts: Dict[str, int], *, run_seed: int = 0, pool: Optional[dict] = None, epsilon: float = 0.1,
    workers: int = 5, excluded: Iterable[str] = (), required: Iterable[str] = (), prefix: str = "",
) -> List[str]:
    """Writes `<out_dir>/<prefix><source>_<shard>.jsonl` shards, skipping
    any already complete (so an interrupted run resumes). Returns every
    shard path, complete or freshly written."""
    out_dir = data_root.resolve(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    pool = dict(pool or {"decks": ["fignor", "igor"], "deck_source": "presets", "max_turns": 200})
    tasks = []
    for source in SOURCES:
        n = int(counts.get(source, 0))
        for shard, start in enumerate(range(0, n, GAMES_PER_SHARD)):
            path = os.path.join(out_dir, f"{prefix}{source}_{shard:04d}.jsonl")
            tasks.append((path, source, start, min(GAMES_PER_SHARD, n - start), run_seed, pool, epsilon, sorted(excluded), sorted(required)))
    if workers <= 1:
        results = [_shard_task(t) for t in tasks]
    else:
        with mp.get_context("spawn").Pool(workers) as p:
            results = p.map(_shard_task, tasks, chunksize=1)
    return [r[0] for r in results]


def read_records(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    from agent import config as config_mod

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="tier0_bc.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    cfg = config_mod.resolve(args.config)
    bc = cfg["bc"]
    counts = {"heuristic": bc["heuristic_games"], "epsilon": bc["epsilon_games"], "random": bc["random_games"]}
    t0 = time.perf_counter()
    paths = generate(args.out, counts, run_seed=cfg["seed"], pool=cfg["pool"], epsilon=bc["epsilon"], workers=args.workers or bc["workers"])
    print(f"{len(paths)} shards, {sum(counts.values())} games in {time.perf_counter() - t0:.1f}s -> {data_root.resolve(args.out)}")


if __name__ == "__main__":
    main()

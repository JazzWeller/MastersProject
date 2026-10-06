"""Agent Observation Plan, Milestone O1: the acceptance checks for the zone
journal and the projection, over fuzz games.

    python -m tools.o1_acceptance --games 1000 [--workers 8]

runs `--games` games per pool (Phase 1.1: Fignor/Igor; Phase 2: the
Dis/Logos/Shadows presets; Phase 3: the presets with the other houses;
random decks), each played by RandomBot or HeuristicBot, and checks:

- **completeness**: after every decision, where the journal puts each card
  is where it is (`tests/_journal_check.py`), and the privileged
  projection's hands are the hands;
- **sigma-invariance**: permuting each player's cards that a viewer never
  saw, through the privileged journal, log and decision records, leaves
  that viewer's projection identical;
- **no over-hiding**: every entry with a public source or destination
  shows its card to both viewers;
- **determinism**: a replay of the game's choices gives the same journal
  and decisions, and so does a copy at a decision played on.

`tests/test_agent_observation_o1.py` runs the same checks on a few games.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from types import SimpleNamespace
from typing import Dict, List, Optional, Set

from bots.heuristic_bot import HeuristicBot
from bots.random_bot import RandomBot
from keyforge.cards.decks import random_deck
from keyforge.config import GameConfig
from keyforge.game import Game
from keyforge.journal import CARD_VERBS, LIMBO, PUBLIC_KINDS
from keyforge.log import LogEvent
from keyforge.projection import Projector
from keyforge.replay import decode_choice

POOLS = {
    "phase1": [("fignor", "igor")],
    "phase2": ["cinder", "fignor", "gambit", "igor", "riftwalker", "wraith"],
    "phase3": ["starfall", "stonewall", "thornwood", "vigil"],
    "random": None,
}
# The log fields that name a card, by the field holding its instance id.
_NAME_OF_IID_FIELD = {"iid": "card", "iids": "cards"}


def game_config(pool: str, i: int) -> GameConfig:
    rng = random.Random(f"{pool}-{i}")
    if pool == "phase1":
        decks = ("fignor", "igor") if i % 2 == 0 else ("igor", "fignor")
    elif pool == "random":
        decks = (random_deck(rng, "R1"), random_deck(rng, "R2"))
    else:
        decks = (rng.choice(POOLS[pool]), rng.choice(POOLS[pool]))
    return GameConfig(decks=decks, seed=rng.getrandbits(31), max_turns=120)


def bots_for(i: int, seed: int):
    if i % 2:
        return {1: HeuristicBot(seed=seed), 2: HeuristicBot(seed=seed + 1)}
    return {1: RandomBot(seed=seed), 2: RandomBot(seed=seed + 1)}


# ------------------------------------------------------------------ play ----


class Played:
    """One game, played with the per-decision checks, plus what the later
    checks need: the cards each viewer could see at any point."""

    def __init__(self, config: GameConfig, bots, check_every: int = 1):
        from tests._journal_check import mismatches

        self.config = config
        self.game = game = Game(config)
        self.seen: Dict[int, Set[int]] = {1: set(), 2: set()}
        self.problems: List[str] = []
        self.copy_at: Optional[int] = None
        n = 0
        privileged = Projector(None)
        while not game.is_over:
            self._look(game)
            d = game.pending_decision
            for o in d.options:
                card = getattr(o, "card", o)
                iid = getattr(card, "instance_id", None)
                if iid is not None:
                    self.seen[d.player].add(iid)
            if n % check_every == 0:
                for m in mismatches(game)[:3]:
                    self.problems.append(f"after {n} decisions: {m}")
                privileged.update(game.journal, game.log.events)
                for pid, p in game.players.items():
                    hand = {c.instance_id for c in p.hand.cards()}
                    if privileged.hands[pid] != hand:
                        self.problems.append(f"after {n} decisions: the projection's hand {pid} is "
                                             f"{sorted(privileged.hands[pid])}, the hand {sorted(hand)}")
            if self.problems:
                return
            game.submit(bots[d.player].decide(game.view_for(d.player), d))
            n += 1
        self._look(game)

    def _look(self, game) -> None:
        for v in (1, 2):
            seen = self.seen[v]
            for pid, p in game.players.items():
                for c in p.discard.cards() + p.purged.cards() + p.play_area.all_cards():
                    seen.add(c.instance_id)
                    for u in getattr(c.type_object, "upgrades", None) or ():
                        seen.add(u.instance_id)
                    for u in c.under_cards:
                        if u.owner == v:
                            seen.add(u.instance_id)
                if pid == v or v in p.hand_revealed_to:
                    seen.update(c.instance_id for c in p.hand.cards())
                if pid == v:
                    seen.update(c.instance_id for c in p.archive.cards())
        for e in game.log.events:
            if e.kind in ("reveal", "reveal_top", "help_from_future_self", "timetraveler_shuffle"):
                for f in ("iid", "iids"):
                    v = e.data.get(f)
                    for iid in (v if isinstance(v, list) else [v]):
                        if isinstance(iid, int):
                            for w in e.visible_to:
                                self.seen[w].add(iid)


# ------------------------------------------------------------- checks ----


def _remap(sigma, x):
    return sigma.get(x, x) if isinstance(x, int) and not isinstance(x, bool) else x


def _zone(sigma, z):
    if z[0] in ("attached", "under"):
        return (z[0], _remap(sigma, z[1]))
    return z


def _form(sigma, f):
    if f[0] in CARD_VERBS:
        return (f[0], _remap(sigma, f[1]))
    if f[0] == "seq":
        return ("seq", tuple(_form(sigma, x) for x in f[1]))
    if f[0] == "trigger":
        return ("trigger", (_remap(sigma, f[1][0]), f[1][1]))
    return f


def permuted_sources(game, sigma: Dict[int, int]):
    """The game's privileged journal, log and decision records with every
    card relabelled by `sigma` (names follow their instance ids)."""
    name_of = {iid: c.name for iid, c in game._cards_by_id.items()}
    j = game.journal
    entries = []
    for e in j.entries:
        pos = _remap(sigma, e[9]) if e[7] == "swap" else e[9]
        entries.append((e[0], e[1], _remap(sigma, e[2]), e[3], e[4], _zone(sigma, e[5]), _zone(sigma, e[6]), e[7], e[8],
                        pos, _remap(sigma, e[10])))
    events = []
    for ev in game.log.events:
        data = dict(ev.data)
        for f, name_field in _NAME_OF_IID_FIELD.items():
            if f not in data:
                continue
            v = data[f]
            if isinstance(v, list):
                data[f] = [_remap(sigma, x) for x in v]
                if isinstance(data.get(name_field), list) and len(data[name_field]) == len(v):
                    data[name_field] = [name_of.get(x, n) for x, n in zip(data[f], data[name_field])]
            else:
                data[f] = _remap(sigma, v)
                if isinstance(data.get(name_field), str):
                    data[name_field] = name_of.get(data[f], data[name_field])
        events.append(LogEvent(ev.kind, data, visible_to=ev.visible_to, private=ev.private))
    decisions = []
    for r in j.records():
        n_e, n_l, player, kind, intent, source, affects, optional, chosen, offered = r
        decisions.append((n_e, n_l, player, kind, intent, _remap(sigma, source), affects, optional,
                          tuple(_form(sigma, f) for f in chosen), tuple(_form(sigma, f) for f in offered)))
    return SimpleNamespace(entries=entries, marks=j.marks, decisions=decisions), events


def check_sigma(played: Played, rng: random.Random) -> List[str]:
    game = played.game
    problems = []
    for v in (1, 2):
        hidden_by_owner: Dict[int, List[int]] = {1: [], 2: []}
        for iid, c in sorted(game._cards_by_id.items()):
            if iid not in played.seen[v]:
                hidden_by_owner[c.owner].append(iid)
        sigma: Dict[int, int] = {}
        for owner, iids in hidden_by_owner.items():
            shuffled = list(iids)
            rng.shuffle(shuffled)
            sigma.update(zip(iids, shuffled))
        if not any(a != b for a, b in sigma.items()):
            continue
        base = Projector(v).update(game.journal, game.log.events)
        journal, events = permuted_sources(game, sigma)
        moved = Projector(v).update(journal, events)
        if base != moved:
            for i, (a, b) in enumerate(zip(base, moved)):
                if a != b:
                    problems.append(f"viewer {v}, item {i}: {a} became {b} under sigma")
                    break
            else:
                problems.append(f"viewer {v}: {len(base)} items became {len(moved)}")
    return problems


def check_no_over_hiding(game) -> List[str]:
    problems = []
    for v in (1, 2):
        items = [x for x in Projector(v).update(game.journal, game.log.events) if x[0] == "zone"]
        by_seq = {x[1]: x for x in items}
        for e in game.journal.entries:
            if e[2] is None or e[5] == e[6]:
                continue
            if e[5][0] in PUBLIC_KINDS and e[5] != LIMBO or e[6][0] in PUBLIC_KINDS and e[6] != LIMBO:
                if by_seq[e[0]][3] is None:
                    problems.append(f"viewer {v}: entry {e} hides a card with a public side")
    return problems


def _journal_state(game):
    return (list(game.journal.entries), list(game.journal.records()))


def check_determinism(played: Played, rng: random.Random) -> List[str]:
    game = played.game
    problems = []
    replay = Game(played.config)
    record = game.choice_record
    k = rng.randrange(len(record)) if record else 0
    copy = None
    for i, encoded in enumerate(record):
        if i >= k and copy is None and replay.copy_anywhere:  # (native execution copies at boundaries only)
            copy, k = replay.copy(), i
        replay.submit(decode_choice(replay.pending_decision, encoded))
    if _journal_state(replay) != _journal_state(game):
        problems.append("a replay's journal differs")
    if copy is not None:
        for encoded in record[k:]:
            copy.submit(decode_choice(copy.pending_decision, encoded))
        if _journal_state(copy) != _journal_state(game):
            problems.append(f"a copy at decision {k}, played on, has a different journal")
    return problems


def check_game(config: GameConfig, i: int, check_every: int = 1) -> List[str]:
    played = Played(config, bots_for(i, config.seed), check_every=check_every)
    if played.problems:
        return played.problems
    rng = random.Random(config.seed)
    return check_sigma(played, rng) + check_no_over_hiding(played.game) + check_determinism(played, rng)


def _worker(args):
    pool, i, check_every = args
    config = game_config(pool, i)
    try:
        problems = check_game(config, i, check_every)
    except Exception as ex:  # report, don't stop the sweep
        problems = [f"raised {type(ex).__name__}: {ex}"]
    return pool, i, problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--games", type=int, default=100, help="games per pool")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--check-every", type=int, default=1, help="completeness check every N decisions")
    args = ap.parse_args(argv)
    jobs = [(pool, i, args.check_every) for pool in POOLS for i in range(args.games)]
    t0 = time.time()
    failures = 0
    if args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(args.workers) as p:
            results = p.imap_unordered(_worker, jobs, chunksize=4)
            for pool, i, problems in results:
                if problems:
                    failures += 1
                    print(f"{pool} game {i}:", *problems[:3], sep="\n  ", flush=True)
    else:
        for job in jobs:
            pool, i, problems = _worker(job)
            if problems:
                failures += 1
                print(f"{pool} game {i}:", *problems[:3], sep="\n  ", flush=True)
    print(f"{len(jobs)} games ({args.games} per pool), {failures} with problems, {time.time() - t0:.0f} s")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

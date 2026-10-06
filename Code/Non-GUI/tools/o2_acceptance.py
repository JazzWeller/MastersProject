"""Agent Observation Plan, Milestone O2: the knowledge tracker's soundness
fuzz.

    python -m tools.o2_acceptance --games 1000 [--workers 8]

plays `--games` games per pool (the pools of tools/o1_acceptance.py) and at
every decision checks, for both viewers, that

- every card's true zone is in the tracker's mask (so every exact claim is
  true),
- every known deck position is true,
- the zone counts are the true counts,
- and the same for the second-order tracker (what the opponent knows about
  the viewer's own cards);
- and every `--brute-every` decisions, that each mask contains every zone
  some world consistent with the viewer's projection puts the card in
  (`world_masks`, a DP over the hidden moves) -- and reports how often a
  mask is wider than the worlds.

It also reports how much is known: the share of the opponent's hidden
cards the tracker places exactly, and the mean mask size.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List, Optional

from keyforge.game import Game
from keyforge.journal import LIMBO
from keyforge.knowledge import second_order_tracker, tracker_for
from tools.o1_acceptance import POOLS, bots_for, game_config


def _actual(game):
    from tests._journal_check import actual_locations

    return actual_locations(game)


def check_tracker(game, t, where, label: str, cards=None) -> List[str]:
    problems = []
    for iid in (cards if cards is not None else t.mask):
        z = where.get(iid)
        if z not in t.mask[iid]:
            problems.append(f"{label}: card {iid} is in {z}, the tracker says {sorted(t.mask[iid], key=repr)}")
    for pid, p in game.players.items():
        deck = [c.instance_id for c in p.deck.cards()]
        for i, x in enumerate(t.top[pid]):
            if x is not None and (i >= len(deck) or deck[i] != x):
                problems.append(f"{label}: deck {pid} top {t.top[pid]}, actually {deck[:len(t.top[pid])]}")
                break
        b = t.bottom[pid]
        for i, x in enumerate(b):
            j = len(deck) - len(b) + i
            if x is not None and (j < 0 or deck[j] != x):
                problems.append(f"{label}: deck {pid} bottom {b}, actually {deck[-len(b):] if b else []}")
                break
    if cards is None:
        for zone, n in t.counts.items():
            true = sum(1 for z in where.values() if z == zone)
            if zone[0] != "setup" and n != true:
                problems.append(f"{label}: zone {zone} count {n}, actually {true}")
    return problems


_ANON = "anon"


def world_masks(game, v: int, labels: List[int], max_states: int = 50000) -> Optional[dict]:
    """Every zone each of `labels` (cards of `v`'s opponent) can be in, over
    all worlds consistent with what `v` saw: a DP over the opponent's
    journal entries whose state is where each label is plus what is known of
    each deck's ends. A label is anonymous (None) until `v` first sees it --
    until then it is interchangeable with every other card `v` hasn't seen
    -- and then must have been somewhere an anonymous card could be. A move
    `v` didn't see takes, from its zone, any label there or an anonymous
    card (if the zone holds one then: zone counts are public); a draw from
    a deck end known to hold a card takes that card; a move or note `v`
    saw, and a hand revealed to `v`, must match. Every opponent card `v`
    ever saw must be a label (an anonymous card has no history to check).
    None if the DP outgrows `max_states`."""
    o = 3 - v
    items = game.projected(v)
    shown = {x[1]: x[3] for x in items if x[0] == "zone"}
    index = {x: i for i, x in enumerate(labels)}
    deck = ("deck", o)
    counts = {}  # the opponent's cards per zone, as the game goes (public)
    true_where = {}
    states = {(tuple(None for _ in labels), ())}

    def anon(z, zone):
        return counts.get(zone, 0) - sum(1 for zz in z if zz == zone)

    def first_seen(states, i, zone):
        """Label `i` seen in `zone`: from anonymous, it was one of the
        anonymous cards there."""
        out = set()
        for z, ends in states:
            if z[i] is None:
                if zone[0] != "limbo" and anon(z, zone) <= 0:
                    continue
                z = z[:i] + (zone,) + z[i + 1:]
            elif z[i] != zone and zone[0] != "limbo":
                continue
            out.add((z, ends))
        return out

    for e in game.journal.entries:
        seq, iid, owner, frm, to, op, pos = e[0], e[2], e[3], e[5], e[6], e[7], e[9]
        if op == "shuffle":
            if frm == deck:
                states = {(z, ()) for z, _ in states}
            continue
        if op == "reveal_hand":
            if frm == ("hand", o) and v == pos:
                hand = {i for i, x in enumerate(labels) if true_where.get(x) == frm}
                for i in hand:
                    states = first_seen(states, i, frm)
                states = {(z, ends) for z, ends in states if {i for i, zz in enumerate(z) if zz == frm} == hand}
            continue
        if owner != o or op == "unreveal_hand":
            continue
        if frm[0] == "setup":
            counts[to] = counts.get(to, 0) + 1
            true_where[iid] = to
            continue
        seen_iid = shown.get(seq)
        if frm == to:  # a note
            if seen_iid is not None and frm != LIMBO:  # (a note in limbo is about a card mid-move)
                who = iid if iid in index else _ANON
                if iid in index:
                    states = first_seen(states, index[iid], frm)
                if frm == deck and pos == "top":  # the top card, now known
                    states = {(z, ends[:-1] + (who,) if ends else (who,)) for z, ends in states}
            continue
        if seen_iid is not None and seen_iid in index:
            states = first_seen(states, index[seen_iid], frm)
        new = set()
        for z, ends in states:
            options = []
            if seen_iid is not None:
                options.append(seen_iid if seen_iid in index else _ANON)
            elif frm == deck and pos == "top" and ends:
                options.append(ends[-1])
            else:
                options.extend(labels[i] for i, zz in enumerate(z) if zz == frm)
                options.append(_ANON)
            for who in options:
                if who == _ANON:
                    if anon(z, frm) <= 0 and frm[0] != "limbo":
                        continue
                elif z[index[who]] != frm:
                    continue
                nz = z
                if who != _ANON:
                    i = index[who]
                    nz = z[:i] + (to,) + z[i + 1:]
                nt = list(ends)
                if frm == deck:
                    if pos == "top" and nt:
                        nt.pop()
                    elif pos not in ("top", "bottom"):  # (a position may be the destination's flank)
                        nt = [t for t in nt if t != who] if who != _ANON else []
                if to == deck:
                    if pos == "top":
                        nt.append(who)
                    else:
                        nt = []
                new.add((nz, tuple(nt)))
        states = new
        if len(states) > max_states:
            return None
        counts[frm] = counts.get(frm, 0) - 1
        counts[to] = counts.get(to, 0) + 1
        true_where[iid] = to
    out = {x: set() for x in labels}
    for z, _ in states:
        for x, zz in zip(labels, z):
            if zz is not None:
                out[x].add(zz)
    return out


def check_brute_force(game, v: int, limit: int = 40) -> List[str]:
    """The tracker's masks against `world_masks` for the opponent's hidden
    cards `v` has seen (positions where the DP grows too large skipped).
    Returns "unsound: ..." for a mask missing a zone some world puts the
    card in, and "wider: ..." for one wider than the worlds."""
    o = 3 - v
    where = _actual(game)
    t = tracker_for(game, v)
    items = game.projected(v)
    seen = set()
    for x in items:
        if x[0] == "zone" and x[3] is not None:
            seen.add(x[3])
        elif x[0] == "reveal":
            seen.add(x[2])
    hidden = [iid for iid, z in sorted(where.items()) if t.owner[iid] == o and iid in seen
              and z[0] in ("hand", "archive", "deck")]
    labels = sorted(iid for iid in seen if t.owner.get(iid) == o)
    if not hidden or len(labels) > limit:
        return []
    possible = world_masks(game, v, labels)
    if possible is None:
        return []
    problems = []
    for x in hidden:
        m = set(t.mask[x])
        if where[x] not in possible[x]:
            problems.append(f"oracle: viewer {v}: card {x} is in {where[x]}, no world has it there")
        if not possible[x] <= m:
            problems.append(f"unsound: viewer {v}: card {x}'s mask {sorted(m)} misses {sorted(possible[x] - m)}")
        elif m != possible[x]:
            problems.append(f"wider: viewer {v}: card {x}'s mask {sorted(m)}, the worlds' {sorted(possible[x])}")
    return problems


class Stats:
    def __init__(self):
        self.hidden = 0
        self.exact = 0
        self.mask_total = 0
        self.compared = 0
        self.wider = 0

    def add(self, game, t, where):
        v = t.viewer
        for iid, m in t.mask.items():
            if t.owner[iid] == v:
                continue
            z = where[iid]
            if z[0] in ("hand", "archive", "deck"):
                self.hidden += 1
                self.mask_total += len(m)
                if len(m) == 1:
                    self.exact += 1


def check_game(config, i: int, check_every: int = 1, stats: Stats = None, brute_every: int = 0) -> List[str]:
    game = Game(config)
    bots = bots_for(i, config.seed)
    n = 0
    while not game.is_over:
        d = game.pending_decision
        if n % check_every == 0:
            where = _actual(game)
            for v in (1, 2):
                t = tracker_for(game, v)
                problems = check_tracker(game, t, where, f"viewer {v} after {n} decisions")
                s = second_order_tracker(game, v)
                own = [iid for iid, o in t.owner.items() if o == v]
                problems += check_tracker(game, s, where, f"viewer {v} second-order after {n} decisions", own)
                if brute_every and n % brute_every == 0:
                    found = check_brute_force(game, v)
                    if stats is not None:
                        stats.compared += 1
                        stats.wider += sum(1 for p in found if p.startswith("wider"))
                    problems += [p for p in found if not p.startswith("wider")]
                if problems:
                    return problems
                if stats is not None:
                    stats.add(game, t, where)
        game.submit(bots[d.player].decide(game.view_for(d.player), d))
        n += 1
    return []


def _worker(args):
    pool, i, check_every, brute_every = args
    stats = Stats()
    try:
        problems = check_game(game_config(pool, i), i, check_every, stats, brute_every)
    except Exception as ex:
        import traceback
        problems = [f"raised {type(ex).__name__}: {ex}\n{traceback.format_exc()}"]
    return pool, i, problems, (stats.hidden, stats.exact, stats.mask_total, stats.compared, stats.wider)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--games", type=int, default=100, help="games per pool")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--check-every", type=int, default=1)
    ap.add_argument("--brute-every", type=int, default=10, help="compare with the worlds every N decisions (0: never)")
    args = ap.parse_args(argv)
    jobs = [(pool, i, args.check_every, args.brute_every) for pool in POOLS for i in range(args.games)]
    t0 = time.time()
    failures = 0
    hidden = exact = mask_total = compared = wider = 0
    if args.workers > 1:
        import multiprocessing as mp
        pool = mp.Pool(args.workers)
        results = pool.imap_unordered(_worker, jobs, chunksize=4)
    else:
        results = map(_worker, jobs)
    for pool_name, i, problems, (h, e, m, c, w) in results:
        hidden, exact, mask_total = hidden + h, exact + e, mask_total + m
        compared, wider = compared + c, wider + w
        if problems:
            failures += 1
            print(f"{pool_name} game {i}:", *problems[:3], sep="\n  ", flush=True)
    print(f"{len(jobs)} games ({args.games} per pool), {failures} with problems, {time.time() - t0:.0f} s")
    if hidden:
        print(f"opponent's hidden cards placed exactly: {exact / hidden:.1%}; mean mask size {mask_total / hidden:.2f}")
    print(f"compared with the worlds at {compared} positions; masks wider than the worlds: {wider}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

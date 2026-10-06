"""Knowledge-consistent determinization and exact priors (Agent Observation
Plan, Milestone O3).

A *world* is an assignment of the cards a viewer can't see to the zones they
can't see into: the opponent's hand, archive and deck (each deck in an
order), and the order of the viewer's own deck. `sample_world` draws one;
`Game.fork_determinized(viewer, rng, sampler=...)` builds it as a game.

| Sampler | Distribution |
|---|---|
| `uniform` | the legacy re-deal: the opponent's hidden cards (less a revealed hand) shuffled among their zones |
| `constrained` | uniform over assignments consistent with the viewer's knowledge tracker (keyforge/knowledge.py): every mask, every zone count, every known deck position |
| `chance_exact` | the posterior when draws and shuffles are random and the opponent's choices carry no information (`ChanceFilter`) |
| `belief` | `chance_exact`'s support, the hand weighted by a caller's per-card weights (the network's belief head) |

**`constrained`.** Cards fall into classes by mask. A DP over the classes
counts the assignments for each split of a class's cards between the zones
(a product of multinomials, the deck taking the rest); a split is sampled
from it, then the cards of each class are assigned uniformly.

**`chance_exact`** (`ChanceFilter`): the opponent's hidden cards are kept
as *atoms* of exchangeable cards -- at first their whole deck, then every
card the viewer has seen enter their hidden pool is an atom of its own --
and the filter keeps the exact joint distribution of how many cards of
each atom are in each hidden zone. A move the viewer doesn't see takes a
uniformly random card of its zone (a draw, a face-down archive); a card
the viewer sees leave or be revealed in a zone is conditioned on being
there. P(hand), P(archive), P(deck) and P(next draw) of every card follow,
and are what O2 exports.

**Known deck ends** are kept by every sampler but `uniform`; the rest of a
deck is in uniform random order.
"""

from __future__ import annotations

import math
import random
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from .journal import LIMBO, PUBLIC_KINDS

Zone = Tuple[str, Optional[int]]

SAMPLERS = ("uniform", "constrained", "chance_exact", "belief")


# ------------------------------------------------------------ the filter ----


class ChanceFilter:
    """The `chance_exact` posterior over the opponent's hidden zones, for
    one viewer, fed by that viewer's projection (`update`).

    State: `atoms` (each a list of instance ids, exchangeable) and `dist`, a
    dict from a state -- per atom, a tuple of counts in the four slots
    `HAND, ARCHIVE, DECK, ELSEWHERE` (elsewhere: any other zone hidden from
    the viewer, under a card) -- to its probability.

    Exact under its model, with one approximation: a card put face down on
    a deck end without the viewer seeing which (rare) is drawn later as a
    random card, not as that one."""

    __slots__ = ("viewer", "opp", "atoms", "atom_of", "dist", "cursor", "top", "bottom", "_dealt")

    def __init__(self, viewer: int, decklists):
        self.viewer = viewer
        self.opp = 3 - viewer
        self.atoms: List[List[int]] = []
        self.atom_of: Dict[int, int] = {}
        self.dist: Dict[tuple, float] = {(): 1.0}
        self.cursor = 0
        self.top: List[Optional[int]] = []
        self.bottom: List[Optional[int]] = []
        self._dealt = list(decklists[self.opp])

    def slot(self, zone: Zone) -> Optional[int]:
        """The slot of a zone hidden from the viewer that holds the
        opponent's cards, else None (public, or the viewer's own)."""
        kind, key = zone
        if kind in PUBLIC_KINDS or kind == "setup":
            return None
        if key == self.opp:
            return _SLOTS.get(kind)
        if kind == "under":
            return ELSEWHERE
        return None

    # ---------------------------------------------------------- queries ----

    def p_zone(self, iid: int) -> Optional[Tuple[float, float, float, float]]:
        """(P(hand), P(archive), P(deck), P(elsewhere)) for a card in the
        opponent's hidden pool, else None."""
        a = self.atom_of.get(iid)
        if a is None:
            return None
        size = len(self.atoms[a])
        out = [0.0, 0.0, 0.0, 0.0]
        for state, p in self.dist.items():
            c = state[a]
            for k in range(4):
                out[k] += p * c[k] / size
        return tuple(out)

    def p_next_draw(self, iid: int) -> float:
        a = self.atom_of.get(iid)
        if a is None:
            return 0.0
        if self.top and self.top[0] is not None:
            return 1.0 if self.top[0] == iid else 0.0
        known_bottom = {x for x in self.bottom if x is not None}
        if iid in known_bottom:
            return 0.0
        size = len(self.atoms[a])
        total = 0.0
        for state, p in self.dist.items():
            n_deck = sum(c[DECK] for c in state) - len(known_bottom)
            if n_deck > 0:
                total += p * (state[a][DECK] / size) / n_deck
        return total

    # ----------------------------------------------------------- update ----

    def update(self, items) -> None:
        for i in range(self.cursor, len(items)):
            self.feed(items[i])
        self.cursor = len(items)

    def feed(self, item) -> None:
        tag = item[0]
        if tag == "zone":
            self._zone(item)
        elif tag == "reveal":
            _, seq, iid, zone, code = item
            k = self.slot(zone)
            if k is not None:
                self._condition(iid, k)
        elif tag == "decision":
            for iid, zone in item[10]:
                k = self.slot(zone)
                if k is not None:
                    self._condition(iid, k)

    def _normalize(self) -> None:
        total = sum(self.dist.values())
        if total <= 0:
            raise ValueError("chance filter: no consistent world")
        if abs(total - 1.0) > 1e-12:
            inv = 1.0 / total
            self.dist = {s: p * inv for s, p in self.dist.items()}

    def _set(self, a: int, counts: Tuple[int, int, int, int]) -> None:
        """Atom `a`'s counts are now `counts` in every state."""
        new: Dict[tuple, float] = {}
        for s, p in self.dist.items():
            ns = s[:a] + (counts,) + s[a + 1:]
            new[ns] = new.get(ns, 0.0) + p
        self.dist = new

    def _alone(self, iid: int) -> int:
        """Makes `iid` an atom of its own, where it is (the caller conditions
        on where). Returns that atom."""
        a = self.atom_of[iid]
        if len(self.atoms[a]) == 1:
            return a
        size = len(self.atoms[a])
        # the card is in slot k with probability c[k] / size
        new: Dict[tuple, float] = {}
        for s, p in self.dist.items():
            c = s[a]
            for k in range(4):
                if c[k]:
                    rest = list(c)
                    rest[k] -= 1
                    one = [0, 0, 0, 0]
                    one[k] = 1
                    ns = s[:a] + (tuple(rest),) + s[a + 1:] + (tuple(one),)
                    new[ns] = new.get(ns, 0.0) + p * c[k] / size
        self.dist = new
        self.atoms[a].remove(iid)
        b = len(self.atoms)
        self.atoms.append([iid])
        self.atom_of[iid] = b
        return b

    def _condition(self, iid: int, k: int) -> None:
        """`iid` (an opponent card in the pool) is known to be in slot `k`."""
        if iid not in self.atom_of:
            return
        a = self._alone(iid)
        self.dist = {s: p for s, p in self.dist.items() if s[a][k] == 1}
        self._normalize()

    def _drop(self, iid: int) -> None:
        a = self._alone(iid)
        self._set(a, (0, 0, 0, 0))
        self.atoms[a] = []
        del self.atom_of[iid]

    def _place(self, iid: int, k: int) -> None:
        """`iid`, which the viewer saw, is now in slot `k` (new to the pool,
        or moved within it)."""
        one = [0, 0, 0, 0]
        one[k] = 1
        if iid in self.atom_of:
            self._set(self._alone(iid), tuple(one))
            return
        a = len(self.atoms)
        self.atoms.append([iid])
        self.atom_of[iid] = a
        self.dist = {s + (tuple(one),): p for s, p in self.dist.items()}

    def _random_move(self, f: int, t: Optional[int], exclude: frozenset) -> None:
        """A card the viewer didn't see, uniformly random among the cards in
        slot `f` but `exclude` (known deck ends that can't be it), goes to
        slot `t`."""
        if t is None:
            raise ValueError("chance filter: a hidden card left the pool unseen")
        excl = {self.atom_of[x] for x in exclude if x in self.atom_of}
        new: Dict[tuple, float] = {}
        for s, p in self.dist.items():
            total = sum(c[f] for c in s) - len(excl)
            if total <= 0:
                continue
            for a, c in enumerate(s):
                n = c[f] - (1 if a in excl else 0)
                if n <= 0:
                    continue
                counts = list(c)
                counts[f] -= 1
                counts[t] += 1
                ns = s[:a] + (tuple(counts),) + s[a + 1:]
                new[ns] = new.get(ns, 0.0) + p * n / total
        self.dist = new
        self._normalize()

    def _zone(self, item) -> None:
        _, seq, turn, iid, owner, controller, frm, to, op, epoch, position, cause, code = item
        if op in ("reveal_hand", "unreveal_hand"):
            return
        deck = ("deck", self.opp)
        if op == "shuffle":
            if frm == deck:
                self.top, self.bottom = [], []
                self._merge_deck()
            return
        if frm[0] == "setup":
            if frm[1] == self.opp and self._dealt is not None:
                a = len(self.atoms)
                self.atoms.append(self._dealt)
                for x in self._dealt:
                    self.atom_of[x] = a
                self.dist = {s + ((0, 0, len(self._dealt), 0),): p for s, p in self.dist.items()}
                self._dealt = None
            return
        if owner != self.opp:  # (an "under" zone is keyed by its host, not by whose card it is)
            return
        f, t = self.slot(frm), self.slot(to)
        if frm == to:  # a note
            if iid is not None and f is not None:
                self._condition(iid, f)
                if frm == deck and position == "top":
                    if self.top:
                        self.top[0] = iid
                    else:
                        self.top.append(iid)
            return
        if f is None and t is None:
            return
        end = position if position in ("top", "bottom") else None
        if frm == deck:  # the deck's known ends
            if end == "top" and self.top:
                x = self.top.pop(0)
                if iid is None:
                    iid = x
            elif end == "bottom" and self.bottom:
                x = self.bottom.pop()
                if iid is None:
                    iid = x
            elif end is None:
                if iid is None:
                    self.top, self.bottom = [], []
                else:
                    self.top = [x for x in self.top if x != iid]
                    self.bottom = [x for x in self.bottom if x != iid]
        if iid is not None:
            if f is not None:
                self._condition(iid, f)
            if t is None:
                if iid in self.atom_of:
                    self._drop(iid)
            else:
                self._place(iid, t)
        else:
            exclude = frozenset()
            if frm == deck and end is not None:
                other = self.bottom if end == "top" else self.top
                exclude = frozenset(x for x in other if x is not None)
            self._random_move(f, t, exclude)
        if to == deck:
            if end == "top":
                self.top.insert(0, iid)
            elif end == "bottom":
                self.bottom.append(iid)
            else:
                self.top, self.bottom = [], []

    def _merge_deck(self) -> None:
        """After a shuffle, the atoms certainly in the deck are one atom: the
        shuffle made their cards exchangeable (the plan's "a shuffle merges
        groups in the deck"). Empty atoms go too."""
        n = len(self.atoms)
        sure = [a for a in range(n) if self.atoms[a]
                and all(s[a] == (0, 0, len(self.atoms[a]), 0) for s in self.dist)]
        keep = [a for a in range(n) if self.atoms[a] and a not in sure]
        if len(sure) <= 1 and len(keep) + len(sure) == n:
            return
        merged = [x for a in sure for x in self.atoms[a]]
        atoms = [self.atoms[a] for a in keep] + ([merged] if merged else [])
        tail = ((0, 0, len(merged), 0),) if merged else ()
        new: Dict[tuple, float] = {}
        for st, p in self.dist.items():
            ns = tuple(st[a] for a in keep) + tail
            new[ns] = new.get(ns, 0.0) + p
        self.atoms = atoms
        self.atom_of = {x: i for i, members in enumerate(atoms) for x in members}
        self.dist = new

    def copy(self) -> "ChanceFilter":
        new = ChanceFilter.__new__(ChanceFilter)
        new.viewer, new.opp = self.viewer, self.opp
        new.atoms = [list(a) for a in self.atoms]
        new.atom_of = dict(self.atom_of)
        new.dist = dict(self.dist)
        new.cursor = self.cursor
        new.top, new.bottom = list(self.top), list(self.bottom)
        new._dealt = self._dealt
        return new


HAND, ARCHIVE, DECK, ELSEWHERE = 0, 1, 2, 3
_SLOTS = {"hand": HAND, "archive": ARCHIVE, "deck": DECK, "under": ELSEWHERE}


def chance_filter(game, viewer: int) -> ChanceFilter:
    """`viewer`'s chance filter, brought up to date (kept on the game, copied
    with it, like the tracker)."""
    cache = game.__dict__.get("_chance")
    if cache is None:
        cache = game.__dict__["_chance"] = {}
    f = cache.get(viewer)
    if f is None:
        decklists = {pid: [c.instance_id for c in p.all_cards] for pid, p in game.players.items()}
        f = cache[viewer] = ChanceFilter(viewer, decklists)
    f.update(game.projected(viewer))
    return f


# ------------------------------------------------------------ sampling ----


def _hidden_under(game, viewer: int) -> List[Tuple[object, int]]:
    """The slots (host card, index) of the opponent's cards under cards,
    hidden from `viewer`."""
    o = 3 - viewer
    slots = []
    # every card, not just those in play: a host between zones (being
    # destroyed) still holds the cards under it
    for host in sorted(game._cards_by_id.values(), key=lambda c: c.instance_id):
        for k, u in enumerate(host.under_cards):
            if u.owner == o:
                slots.append((host, k))
    return slots


def _opponent_zones(game, viewer: int):
    """The opponent's hidden zones as they truly are: hand, archive, deck
    (top first) and the cards under cards, as lists of cards; and whether
    the hand is revealed to `viewer`."""
    p = game.players[3 - viewer]
    under = [host.under_cards[k] for host, k in _hidden_under(game, viewer)]
    return p.hand.cards(), p.archive.cards(), p.deck.cards(), under, viewer in p.hand_revealed_to


def _deck_order(cards: List, top: Sequence, bottom: Sequence, rng: random.Random, by_iid) -> List:
    """`cards` in a uniformly random order, but the known `top` cards first
    and the known `bottom` cards last (None in either: unknown)."""
    top_known = [by_iid[x] if x is not None else None for x in top]
    bottom_known = [by_iid[x] if x is not None else None for x in bottom]
    fixed = {id(c) for c in top_known + bottom_known if c is not None}
    rest = [c for c in cards if id(c) not in fixed]
    rng.shuffle(rest)
    head = [c if c is not None else rest.pop() for c in top_known]
    tail = [c if c is not None else rest.pop() for c in reversed(bottom_known)]
    return head + rest + tail[::-1]


def _multinomial(n: int, parts: Sequence[int]) -> int:
    out = math.factorial(n)
    for k in parts:
        out //= math.factorial(k)
    return out


def _splits(size: int, allowed: Tuple[int, ...], caps: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    """Every way to put `size` cards in the zones `allowed`, at most each
    zone's count in each."""
    import itertools

    nz = len(caps)
    if not allowed:
        return [(0,) * nz] if size == 0 else []
    *free, last = allowed
    out = []
    for ns in itertools.product(*(range(min(size, caps[k]) + 1) for k in free)):
        rest = size - sum(ns)
        if 0 <= rest <= caps[last]:
            counts = [0] * nz
            for k, n in zip(free, ns):
                counts[k] = n
            counts[last] = rest
            out.append(tuple(counts))
    return out


class _Plan:
    """The DP of `split_constrained` for one (classes, zone counts): the
    options per class and the memoized count of completions."""

    __slots__ = ("options", "memo", "n")

    def __init__(self, classes, caps):
        self.options = [[(s, _multinomial(size, s)) for s in _splits(size, tuple(sorted(allowed)), caps)]
                        for size, allowed in classes]
        self.memo: Dict[tuple, int] = {}
        self.n = len(classes)

    def ways(self, i: int, rem: tuple) -> int:
        if i == self.n:
            return 1 if not any(rem) else 0
        key = (i, rem)
        n = self.memo.get(key)
        if n is None:
            n = 0
            for s, w in self.options[i]:
                if all(a <= b for a, b in zip(s, rem)):
                    n += w * self.ways(i + 1, tuple(b - a for a, b in zip(s, rem)))
            self.memo[key] = n
        return n


_PLANS: Dict[tuple, _Plan] = {}


def split_constrained(classes, caps, rng: random.Random):
    """Samples how many cards of each class go to each zone, uniformly over
    the assignments of cards to zones that fill each zone to its count and
    keep each card in its class's zones. `classes`: [(size, allowed zone
    indices)]; `caps`: the zone counts. Returns per class a tuple of counts.
    (The DP is cached: search draws many worlds from one position.)"""
    caps = tuple(caps)
    key = (tuple((size, tuple(sorted(allowed))) for size, allowed in classes), caps)
    plan = _PLANS.get(key)
    if plan is None:
        if len(_PLANS) > 512:
            _PLANS.clear()
        plan = _PLANS[key] = _Plan(classes, caps)
    rem = caps
    if plan.ways(0, rem) == 0:
        raise ValueError("constrained sampler: no assignment is consistent with the knowledge")
    out = []
    for i in range(plan.n):
        choices = []
        for s, w in plan.options[i]:
            if all(a <= b for a, b in zip(s, rem)):
                n = w * plan.ways(i + 1, tuple(b - a for a, b in zip(s, rem)))
                if n:
                    choices.append((s, n))
        r = rng.randrange(sum(n for _, n in choices))
        for s, n in choices:
            if r < n:
                break
            r -= n
        out.append(s)
        rem = tuple(b - a for a, b in zip(s, rem))
    return out


def _assign(members: List, counts: Sequence[int], rng: random.Random, weights=None) -> List[List]:
    """Splits `members` into groups of `counts`: uniformly, or, with
    `weights` (instance id -> weight), the first group (the hand) drawn by
    weight without replacement."""
    members = list(members)
    groups = []
    counts = list(counts)
    if weights is not None and counts[0]:
        first = []
        for _ in range(counts[0]):
            ws = [max(weights(c.instance_id), 1e-12) for c in members]
            r = rng.random() * sum(ws)
            j = 0
            for j, w in enumerate(ws):
                if r < w:
                    break
                r -= w
            first.append(members.pop(j))
        groups.append(first)
        counts = counts[1:]
    rng.shuffle(members)
    k = 0
    for n in counts:
        groups.append(members[k:k + n])
        k += n
    return groups


def sample_opponent(game, viewer: int, rng: random.Random, sampler: str, weights=None):
    """One world's opponent hidden zones -- (hand, archive, deck top first,
    under) as lists of cards -- from `sampler`."""
    from .knowledge import tracker_for

    hand, archive, deck, under, revealed = _opponent_zones(game, viewer)
    o = 3 - viewer
    by_iid = game._cards_by_id
    if sampler == "uniform":
        fixed = hand if revealed else []
        pool = ([] if revealed else list(hand)) + archive + deck
        rng.shuffle(pool)
        n_hand = len(hand) - len(fixed)
        return fixed + pool[:n_hand], pool[n_hand:n_hand + len(archive)], pool[n_hand + len(archive):], list(under)
    # cards the opponent doesn't own (one of the viewer's, archived from
    # play into the opponent's archive) stay where they are; only the
    # opponent's own cards are dealt
    fixed = [[c for c in z if c.owner != o] for z in (hand, archive, deck, under)]
    free = [[c for c in z if c.owner == o] for z in (hand, archive, deck, under)]
    caps = tuple(len(z) for z in free)
    pool = free[0] + free[1] + free[2] + free[3]
    out = [list(fixed[0]), list(fixed[1]), list(fixed[2]), list(fixed[3])]
    if sampler == "constrained":
        t = tracker_for(game, viewer)
        slots = {("hand", o): HAND, ("archive", o): ARCHIVE, ("deck", o): DECK}
        by_mask: Dict[FrozenSet[int], List] = {}
        for c in pool:
            idx = frozenset(slots.get(z, ELSEWHERE) for z in t.mask[c.instance_id]
                            if z in slots or z[0] == "under")
            by_mask.setdefault(idx, []).append(c)
        keys = sorted(by_mask, key=lambda k: (len(by_mask[k]), sorted(k)))
        splits = split_constrained([(len(by_mask[k]), k) for k in keys], caps, rng)
        for k, s in zip(keys, splits):
            for z, group in enumerate(_assign(by_mask[k], s, rng)):
                out[z].extend(group)
        top, bottom = t.top[o], t.bottom[o]
    elif sampler in ("chance_exact", "belief"):
        f = chance_filter(game, viewer)
        r = rng.random()
        state = None
        for state, p in f.dist.items():
            if r < p:
                break
            r -= p
        for a, members in enumerate(f.atoms):
            if members:
                groups = _assign([by_iid[x] for x in members], state[a], rng,
                                 weights if sampler == "belief" else None)
                for z, group in enumerate(groups):
                    out[z].extend(group)
        top, bottom = f.top, f.bottom
    else:
        raise ValueError(f"unknown sampler {sampler!r} (one of {SAMPLERS})")
    rng.shuffle(out[HAND])
    rng.shuffle(out[ARCHIVE])
    rng.shuffle(out[ELSEWHERE])
    return out[HAND], out[ARCHIVE], _deck_order(out[DECK], top, bottom, rng, by_iid), out[ELSEWHERE]


def sample_own_deck(game, viewer: int, rng: random.Random, sampler: str) -> List:
    """The order of `viewer`'s own deck in one world."""
    from .knowledge import tracker_for

    cards = game.players[viewer].deck.cards()
    if sampler == "uniform":
        rng.shuffle(cards)
        return cards
    t = tracker_for(game, viewer)
    return _deck_order(cards, t.top[viewer], t.bottom[viewer], rng, game._cards_by_id)


def sample_world(game, viewer: int, rng: random.Random, resample, sampler: str, weights=None):
    """One world for `viewer` (`resample`: keyforge.enums.Resample) as
    instance ids: (opponent's (hand, archive, deck, under) or None, own deck
    or None)."""
    from .enums import Resample

    opponent = own = None
    if resample in (Resample.OPPONENT_PRIVATE, Resample.ALL):
        opponent = tuple([c.instance_id for c in zone] for zone in sample_opponent(game, viewer, rng, sampler, weights))
    if resample in (Resample.OWN_DECK, Resample.ALL):
        own = [c.instance_id for c in sample_own_deck(game, viewer, rng, sampler)]
    return opponent, own


def apply_world(game, viewer: int, opponent=None, own_deck=None) -> Dict:
    """Puts a sampled world (`sample_world`'s instance ids) in place,
    without journaling (a re-deal is not a game event). Returns sigma: each
    card -> the card now in its place."""
    import collections

    by_iid = game._cards_by_id
    if opponent is not None:
        opponent = tuple([by_iid[x] for x in zone] for zone in opponent)
    if own_deck is not None:
        own_deck = [by_iid[x] for x in own_deck]
    sigma = {}

    def put(old, new):
        """The zone's new contents, with every card that stays where it was
        (so it maps to itself) and the newcomers in the places vacated;
        sigma maps each card that left to the card now in its place."""
        stays = {id(c) for c in new} & {id(c) for c in old}
        incoming = [c for c in new if id(c) not in stays]
        out = []
        k = 0
        for c in old:
            if id(c) in stays:
                out.append(c)
            else:
                out.append(incoming[k])
                sigma[c] = incoming[k]
                k += 1
        return out + incoming[k:]

    if opponent is not None:
        p = game.players[3 - viewer]
        hand, archive, deck, under = opponent
        old = _opponent_zones(game, viewer)
        hand, archive, under = put(old[0], hand), put(old[1], archive), put(old[3], under)
        put(old[2], deck)  # (the deck keeps its sampled order: known ends are already in place)
        p.hand._cards = list(hand)
        p.archive._cards = list(archive)
        p.deck._cards = collections.deque(deck)
        for (host, k), c in zip(_hidden_under(game, viewer), under):
            host.under_cards[k] = c
    if own_deck is not None:
        p = game.players[viewer]
        old = p.deck.cards()
        sigma.update({a: b for a, b in zip(old, own_deck) if a is not b})
        p.deck._cards = collections.deque(own_deck)
    return sigma


# ------------------------------------------------- history in a world ----


def _fold(entries) -> Dict[int, Zone]:
    """Each card's zone, by the journal; None if a card leaves a zone the
    journal doesn't have it in."""
    where: Dict[int, Zone] = {}
    for e in entries:
        iid, frm, to = e[2], e[5], e[6]
        if iid is None or frm == to:
            continue
        if frm != LIMBO and frm[0] != "setup" and where.get(iid, LIMBO) != frm:
            return None
        where[iid] = to
    return where


def _zones_now(game) -> Dict[int, Zone]:
    where: Dict[int, Zone] = {}
    for pid, p in game.players.items():
        for kind in ("deck", "hand", "discard", "archive", "purged"):
            for c in getattr(p, kind).cards():
                where[c.instance_id] = (kind, pid)
        for c in p.play_area.creatures:
            where[c.instance_id] = ("battleline", pid)
        for c in p.play_area.artifacts:
            where[c.instance_id] = ("artifacts", pid)
    for host in game._cards_by_id.values():
        for u in getattr(host.type_object, "upgrades", None) or ():
            where[u.instance_id] = ("attached", host.instance_id)
        for u in host.under_cards:
            where[u.instance_id] = ("under", host.instance_id)
    return where


def relabel_history(world, viewer: int, sigma_iids: Dict[int, int]) -> bool:
    """O3's `relabel`: the world's journal with the cards of the entries
    `viewer` didn't see relabelled by sigma (true card -> the card in its
    place), so the opponent's private past matches the world. Accepted --
    and put in place -- only if the relabelled journal, folded, puts every
    card where the world has it (so no relabelled entry contradicts the
    public timeline, which `viewer` saw and sigma left alone). Returns
    whether it was."""
    if not sigma_iids:
        return True
    hidden = {x[1] for x in world.projected(viewer) if x[0] == "zone" and x[3] is None}
    j = world.journal
    entries = j.entries
    new = []
    for e in entries:
        if e[0] in hidden and e[2] is not None:
            e = e[:2] + (sigma_iids.get(e[2], e[2]),) + e[3:]
        new.append(e)
    folded = _fold(new)
    if folded is None:
        return False
    now = _zones_now(world)
    for iid, z in now.items():
        if folded.get(iid, LIMBO) != z:
            return False
    saved = (list(j._entries), list(world.log.events), list(j.records()))
    j._entries[:] = new
    # the opponent's private log fields (their draws) and the parts of their
    # decisions `viewer` didn't see, likewise
    from .log import LogEvent

    events = world.log.events
    for i, ev in enumerate(events):
        if ev.private and any(viewer not in who for who in ev.private.values()):
            data = dict(ev.data)
            for f, who in ev.private.items():
                if viewer not in who and f in data:
                    v = data[f]
                    data[f] = [sigma_iids.get(x, x) for x in v] if isinstance(v, list) else sigma_iids.get(v, v)
            events[i] = LogEvent(ev.kind, data, visible_to=ev.visible_to, private=ev.private)
    seen_forms = [x[8] for x in world.projected(viewer) if x[0] == "decision"]
    records = j.records()
    for i, (rec, shown) in enumerate(zip(records, seen_forms)):
        if rec[2] == viewer:
            continue
        chosen = tuple((f[0], sigma_iids.get(f[1], f[1])) if len(g) == 3 and g[1] is None else f
                       for f, g in zip(rec[8], shown))
        records[i] = rec[:8] + (chosen, rec[9])
    # and the opponent, reading that history, must know nothing false about
    # the world
    from .knowledge import Tracker
    from .projection import Projector

    o = 3 - viewer
    decklists = {pid: [c.instance_id for c in p.all_cards] for pid, p in world.players.items()}
    t = Tracker(o, decklists)
    t.update(Projector(o).update(j, world.log.events))
    if any(now.get(iid, LIMBO) not in m for iid, m in t.mask.items()):
        j._entries[:], world.log.events[:], j.decisions[:] = saved
        _reindex(world.log)
        return False
    _reindex(world.log)
    return True


def _reindex(log) -> None:
    by_kind = log.by_kind
    by_kind.clear()
    for e in log.events:
        by_kind[e.kind].append(e)

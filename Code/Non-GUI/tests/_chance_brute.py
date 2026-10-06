"""A brute-force enumeration of the generative process, for checking
keyforge/determinize.py's `ChanceFilter` (Agent Observation Plan O3).

A scenario is a list of events on the opponent's (player 2's) cards, seen
by player 1:

- ("deal", n): n cards dealt to the deck, then shuffled;
- ("draw",): the top card to the hand, unseen;
- ("play", k): the k-th card of the true hand is played (seen) to the discard;
- ("archive",): a uniformly random hand card goes to the archive, unseen;
- ("reveal_top",): the top card is revealed (seen), stays;
- ("reshuffle",): the discard is shuffled into the deck (seen), then the deck is shuffled;
- ("put_on_top", k): the k-th discard card goes on top of the deck (seen);
- ("return", k): the k-th discard card returns to the hand (seen).

`items(scenario, rng)` plays it in one true world and returns the viewer's
projection items (and the true final zones); `posterior(scenario, items)`
enumerates every world -- every deck order at each shuffle, every choice of
the random archive -- keeps those in which what the viewer saw happens, and
returns P(card in hand / archive / deck) and P(next draw).
"""

from __future__ import annotations

import itertools
import random
from fractions import Fraction

O = 2
HAND, ARCHIVE, DECK, DISCARD = ("hand", O), ("archive", O), ("deck", O), ("discard", O)


def _item(seq, iid, frm, to, op, position=None):
    return ("zone", seq, 0, iid, O, O, frm, to, op, None, position, None, "public")


def items(scenario, rng: random.Random):
    deck, hand, archive, discard = [], [], [], []
    out = []
    seq = 0
    for ev in scenario:
        kind = ev[0]
        if kind == "deal":
            for x in range(ev[1]):
                out.append(_item(seq, None, ("setup", O), DECK, "deal")); seq += 1
                deck.append(x)
            rng.shuffle(deck)
            out.append(_item(seq, None, DECK, DECK, "shuffle")); seq += 1
        elif kind == "draw":
            hand.append(deck.pop(0))
            out.append(_item(seq, None, DECK, HAND, "draw_top/add", "top")); seq += 1
        elif kind == "play":
            x = hand.pop(ev[1] % len(hand))
            discard.append(x)
            out.append(_item(seq, x, HAND, DISCARD, "remove/push")); seq += 1
        elif kind == "archive":
            x = hand.pop(rng.randrange(len(hand)))
            archive.append(x)
            out.append(_item(seq, None, HAND, ARCHIVE, "remove/add")); seq += 1
        elif kind == "reveal_top":
            out.append(_item(seq, deck[0], DECK, DECK, "reveal", "top")); seq += 1
        elif kind == "reshuffle":
            for x in discard:
                out.append(_item(seq, x, DISCARD, DECK, "take_all/shuffle_in")); seq += 1
            deck.extend(discard)
            discard.clear()
            rng.shuffle(deck)
            out.append(_item(seq, None, DECK, DECK, "shuffle")); seq += 1
        elif kind == "put_on_top":
            x = discard.pop(ev[1] % len(discard))
            deck.insert(0, x)
            out.append(_item(seq, x, DISCARD, DECK, "remove/put_on_top", "top")); seq += 1
        elif kind == "return":
            x = discard.pop(ev[1] % len(discard))
            hand.append(x)
            out.append(_item(seq, x, DISCARD, HAND, "remove/add")); seq += 1
        else:
            raise ValueError(kind)
    truth = {x: HAND for x in hand}
    truth.update({x: ARCHIVE for x in archive})
    truth.update({x: DECK for x in deck})
    return out, truth


def posterior(scenario, observed):
    """{card: (P(hand), P(archive), P(deck), P(next draw))} over the
    hidden pool, exactly (Fractions)."""
    shown = [x for x in observed if x[3] is not None]  # in order
    totals = {}
    weight_sum = Fraction(0)

    def run(i, deck, hand, archive, discard, k_shown, w):
        nonlocal weight_sum
        if i == len(scenario):
            weight_sum += w
            for x in hand:
                totals.setdefault(x, [0, 0, 0, 0])[0] += w
            for x in archive:
                totals.setdefault(x, [0, 0, 0, 0])[1] += w
            for x in deck:
                totals.setdefault(x, [0, 0, 0, 0])[2] += w
            if deck:
                totals.setdefault(deck[0], [0, 0, 0, 0])[3] += w
            return
        ev = scenario[i]
        kind = ev[0]

        def expect(x):
            return k_shown < len(shown) and shown[k_shown][3] == x

        if kind == "deal":
            cards = list(range(ev[1]))
            perms = list(itertools.permutations(cards))
            for p in perms:
                run(i + 1, list(p), hand, archive, discard, k_shown, w / len(perms))
        elif kind == "draw":
            run(i + 1, deck[1:], hand + [deck[0]], archive, discard, k_shown, w)
        elif kind == "play":
            x = shown[k_shown][3]
            if x in hand:
                h = list(hand)
                h.remove(x)
                run(i + 1, deck, h, archive, discard + [x], k_shown + 1, w)
        elif kind == "archive":
            for j in range(len(hand)):
                h = list(hand)
                x = h.pop(j)
                run(i + 1, deck, h, archive + [x], discard, k_shown, w / len(hand))
        elif kind == "reveal_top":
            if deck and expect(deck[0]):
                run(i + 1, deck, hand, archive, discard, k_shown + 1, w)
        elif kind == "reshuffle":
            ok = all(expect_at(shown, k_shown + j, x) for j, x in enumerate(discard))
            if not ok:
                return
            full = deck + discard
            perms = list(itertools.permutations(full))
            for p in perms:
                run(i + 1, list(p), hand, archive, [], k_shown + len(discard), w / len(perms))
        elif kind == "put_on_top":
            x = shown[k_shown][3]
            if x in discard:
                d = list(discard)
                d.remove(x)
                run(i + 1, [x] + deck, hand, archive, d, k_shown + 1, w)
        elif kind == "return":
            x = shown[k_shown][3]
            if x in discard:
                d = list(discard)
                d.remove(x)
                run(i + 1, deck, hand + [x], archive, d, k_shown + 1, w)

    run(0, [], [], [], [], 0, Fraction(1))
    return {x: tuple(v / weight_sum for v in vals) for x, vals in totals.items()}


def expect_at(shown, k, x):
    return k < len(shown) and shown[k][3] == x

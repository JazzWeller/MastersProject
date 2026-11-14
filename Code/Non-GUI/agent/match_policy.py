"""A network agent's two between-game decisions in an Adaptive match: whether
to go first or second, and how many chains to bid for the deck that won both
games (keyforge/match.py has the format).

Neither decision has a board to search, so both are read off the value head
at the next game's opening position (the first mulligan, both hands dealt,
any starting chains already shed), averaged over `deals` deals that are the
same for every option compared (common random numbers):

- First or second: the side with the higher mean value.
- Bidding (`bid="rollout"`, the default): game 3 is played out by the
  network's own policy on both sides (no search; `rollouts` games per chain
  count, half with each player first, the same deals at every count), giving
  P(c), the pilot's chance with the proven deck and c chains. The policy is
  the same on both sides, so my chance facing it is 1 - P(c), and the
  sensible strategy in an ascending auction is to raise while P(c) > 1/2:
  bid up to the indifference point c*, found by bisection over 0..24.
- `bid="value"` reads P_pilot(c) and P_face(c) off the value head at game 3's
  opening position instead, raising while P_pilot(c) > P_face(c) (game 2's
  loser chooses who goes first, so each takes that chooser's best option;
  the curves are made monotone against noise). Measured on real-v1 it badly
  underprices chains (piloting a 13-SAS-better deck with 24 chains still
  rated 41%: the value head almost never saw starting chains), so its bids
  run to the cap and the bid winner then loses game 3 -- kept as that
  diagnostic, not used.

Only for network v1 (`agent.features.encode`); a v2 player makes these two
decisions as HeuristicBot does (`ml.arena`).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from keyforge.config import GameConfig
from keyforge.enums import DecisionKind
from keyforge.game import Game
from keyforge.infoset import build_infoset
from keyforge.keyed_random import derive_rng
from keyforge.match import MAX_BID

from .agents.requests import HEAD_VALUE, Request
from .features import encode

MATCH_KINDS = (DecisionKind.BID_CHAINS, DecisionKind.CHOOSE_FIRST_PLAYER)


class _Cap:
    def __init__(self, game, viewer):
        self._game = game
        self.viewer = viewer

    def infoset(self):
        return build_infoset(self._game, self.viewer)


class NetMatchPolicy:
    def __init__(self, client, deals: int = 8, seed: int = 0, bid: str = "rollout", rollouts: int = 64,
                 max_turns: int = 200):
        if bid not in ("rollout", "value"):
            raise ValueError(f"unknown bid method {bid!r}")
        self.client = client
        self.deals = deals
        self.seed = seed
        self.bid_method = bid
        self.rollouts = rollouts
        self.max_turns = max_turns
        self._curves: Dict[tuple, Tuple[List[float], List[float]]] = {}
        self._cstar: Dict[tuple, int] = {}
        self.last_scores: Dict[int, float] = {}  # the rollout scores behind the last c* (records and tests)
        self.last: Optional[dict] = None  # the numbers behind the last decision (records and tests)

    # ---------------------------------------------------------------- values ----

    def _seeds(self, view, tag: str) -> List[int]:
        base = tuple(g.config.seed for g in view.games)
        return [derive_rng(self.seed, None, f"match-policy|{tag}|{base}", k).getrandbits(31) for k in range(self.deals)]

    def _values(self, setups, viewer: int, seeds) -> List[float]:
        """My mean win probability at the opening position of each setup (decks by pid, first player, chains)."""
        reqs = []
        for decks, first, chains in setups:
            for s in seeds:
                g = Game(GameConfig(decks=(decks[1], decks[2]), first_player=first, seed=s,
                                    starting_chains=chains or None))
                reqs.append(Request(enc=encode(build_infoset(g, viewer)), head=HEAD_VALUE))
        out = self.client.predict_many(reqs)
        n = len(seeds)
        return [sum((float(v) + 1.0) / 2.0 for _p, v in out[i * n:(i + 1) * n]) / n for i in range(len(setups))]

    # ------------------------------------------------------------- decisions ----

    def decide(self, view, decision):
        if decision.kind == DecisionKind.CHOOSE_FIRST_PLAYER:
            return self._first(view, decision)
        if decision.kind == DecisionKind.BID_CHAINS:
            return self._bid(view, decision)
        raise ValueError(f"not a match-level decision: {decision.kind}")

    def _next_game(self, view) -> Tuple[Dict[int, object], Dict[int, int]]:
        """The next game's decks by pid and starting chains."""
        g1 = view.games[0]
        if len(view.games) == 1:
            return {1: g1.seat_decks[2], 2: g1.seat_decks[1]}, {}
        bid = view.bid
        proven = g1.seat_decks[g1.winner]
        other = g1.seat_decks[3 - g1.winner]
        return {bid.winner: proven, 3 - bid.winner: other}, {bid.winner: bid.amount, 3 - bid.winner: 0}

    def _first(self, view, decision):
        me = decision.player
        decks, chains = self._next_game(view)
        seeds = self._seeds(view, "first")
        v_first, v_second = self._values([(decks, me, chains), (decks, 3 - me, chains)], me, seeds)
        self.last = {"kind": "first", "first": v_first, "second": v_second}
        return "first" if v_first >= v_second else "second"

    def curves(self, view, me: int) -> Tuple[List[float], List[float]]:
        """(P_pilot, P_face) for c = 0..MAX_BID, from my side."""
        key = (tuple(g.config.seed for g in view.games), me)
        if key not in self._curves:
            g1, g2 = view.games[0], view.games[1]
            proven = g1.seat_decks[g1.winner]
            other = g1.seat_decks[3 - g1.winner]
            chooser = 3 - g2.winner
            seeds = self._seeds(view, "bid")
            setups = []
            for c in range(MAX_BID + 1):
                for pilot in (me, 3 - me):
                    for first in (1, 2):
                        setups.append(({pilot: proven, 3 - pilot: other}, first, {pilot: c} if c else {}))
            vals = self._values(setups, me, seeds)
            pick = max if chooser == me else min
            p_pilot, p_face = [], []
            for c in range(MAX_BID + 1):
                a1, a2, b1, b2 = vals[4 * c:4 * c + 4]
                p_pilot.append(pick(a1, a2))
                p_face.append(pick(b1, b2))
            for c in range(1, MAX_BID + 1):  # monotone: more chains never help their holder
                p_pilot[c] = min(p_pilot[c], p_pilot[c - 1])
                p_face[c] = max(p_face[c], p_face[c - 1])
            self._curves[key] = (p_pilot, p_face)
        return self._curves[key]

    # -------------------------------------------------------------- rollouts ----

    def _rollout(self, view, c: int) -> float:
        """The pilot's mean score with the proven deck and `c` chains, game 3 played out by the policy on both sides."""
        from .agents.net_agent import NetAgent
        from .search.core import legal_actions

        g1 = view.games[0]
        proven, other = g1.seat_decks[g1.winner], g1.seat_decks[3 - g1.winner]
        base = tuple(g.config.seed for g in view.games)
        # one deal per pair of games, played once with each side first; the same deals at every c
        seeds = [derive_rng(self.seed, None, f"match-policy|rollout|{base}", k).getrandbits(31)
                 for k in range((self.rollouts + 1) // 2)]
        games = [Game(GameConfig(decks=(proven, other), first_player=1 + k % 2, seed=seeds[k // 2],
                                 starting_chains={1: c} if c else None, max_turns=self.max_turns))
                 for k in range(self.rollouts)]
        agent = NetAgent(self.client, mode="policy", seed=self.seed)
        live = list(games)
        while live:
            for g in live:
                while not g.is_over:
                    acts = legal_actions(g.pending_decision, 1024)
                    if acts is None or len(acts) != 1:
                        break
                    g.submit(acts[0][1])
            live = [g for g in live if not g.is_over]
            if not live:
                break
            reqs = [(None, g.pending_decision, None, _Cap(g, g.pending_decision.player)) for g in live]
            for g, choice in zip(live, agent.decide_many(reqs)):
                g.submit(choice)
        return sum((g.outcome_for(1) + 1) / 2 for g in games) / len(games)

    def indifference(self, view) -> int:
        """c*: the most chains at which piloting the proven deck still scores over 1/2 (-1: not even at 0)."""
        key = tuple(g.config.seed for g in view.games)
        if key not in self._cstar:
            scores = {}

            def wins(c):
                if c not in scores:
                    scores[c] = self._rollout(view, c)
                return scores[c] > 0.5

            if not wins(0):
                cstar = -1
            elif wins(MAX_BID):
                cstar = MAX_BID
            else:
                lo, hi = 0, MAX_BID  # wins(lo), not wins(hi)
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    lo, hi = (mid, hi) if wins(mid) else (lo, mid)
                cstar = lo
            self._cstar[key] = cstar
            self.last_scores = dict(sorted(scores.items()))
        return self._cstar[key]

    def _bid(self, view, decision):
        me = decision.player
        raises = sorted(o for o in decision.options if isinstance(o, int))
        if self.bid_method == "rollout":
            cstar = self.indifference(view)
            r = raises[0] if raises else None
            take = r is not None and r <= cstar
            self.last = {"kind": "bid", "raise": r, "indifference": cstar, "take": take}
            return r if take else "pass"
        p_pilot, p_face = self.curves(view, me)
        r = raises[0] if raises else None
        take = r is not None and p_pilot[r] > p_face[r]
        self.last = {"kind": "bid", "raise": r, "pilot": p_pilot[r] if r is not None else None,
                     "face": p_face[r] if r is not None else None, "take": take}
        return r if take else "pass"

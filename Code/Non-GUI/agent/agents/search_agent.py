"""The search agent (Agent Training Plan, Milestones M4-M6): the M4 core,
parameterized by regime (within-turn / full-game) and leaf estimator
(heuristic / student / belief+oracle), as an ordinary `Controller`.

Needs the search capability (`fork_determinized`); handed only an
observation-level capability (or none), it falls back to its fixed policy
(HeuristicBot by default) rather than failing -- so it passes the
conformance suite at every privilege level. A forced decision (one legal
submission) is answered without searching.

`last_result` keeps the most recent `SearchResult` (visit counts, root
value) -- the policy target a self-play actor records (M7).
"""

from __future__ import annotations

import time
from typing import Optional

from bots.base import Controller
from bots.heuristic_bot import HeuristicBot
from keyforge.enums import DecisionKind, Resample

from ..search.core import Search, SearchSettings, action_key, drive, legal_actions
from ..search.full_game import FullGame
from ..search.leaf import make_evaluator
from ..search.policies import make_policy
from ..search.within_turn import WithinTurn


def make_search(
    regime: str = "within_turn", *, leaf: str = "heuristic", rollout: str = "heuristic", client=None,
    settings: Optional[SearchSettings] = None, seed: Optional[int] = None, belief_samples: int = 8,
    branch_opponent: bool = False,
) -> Search:
    policy = make_policy(rollout, seed=seed, client=client)
    if regime == "within_turn":
        reg = WithinTurn(policy, branch_opponent=branch_opponent)
    elif regime == "full_game":
        reg = FullGame(policy)
    else:
        raise ValueError(f"unknown search regime {regime!r}")
    evaluator = make_evaluator(leaf, samples=belief_samples, seed=seed)
    return Search(reg, evaluator, policy, settings or SearchSettings(), seed=seed)


class SearchAgent(Controller):
    needs_view = True  # only for the fallback policy

    def __init__(
        self, regime: str = "within_turn", *, leaf: str = "heuristic", rollout: str = "heuristic", client=None,
        settings: Optional[SearchSettings] = None, seed: Optional[int] = None, belief_samples: int = 8,
        branch_opponent: bool = False,
    ):
        if leaf != "heuristic" and client is None:
            raise ValueError(f"leaf estimator {leaf!r} needs an inference client")
        self.client = client
        self.search = make_search(
            regime, leaf=leaf, rollout=rollout, client=client, settings=settings, seed=seed,
            belief_samples=belief_samples, branch_opponent=branch_opponent,
        )
        self.fallback = HeuristicBot(seed=seed)
        self.seat: Optional[int] = None
        self.last_result = None
        # Running totals over every search this agent has made (M10
        # telemetry and the parallel evaluator's per-search averages).
        self.search_stats = {"decisions_searched": 0, "search_seconds": 0.0, "simulations": 0, "evaluations": 0,
                             "forks": 0, "fork_seconds": 0.0}

    def on_game_start(self, seat: int, config: dict) -> None:
        self.seat = seat
        self.search.root = None
        self.search._reuse_from = None
        self.last_result = None

    def observe(self, event) -> None:
        if isinstance(event, tuple) and len(event) == 3:
            _kind, player, _choice = event
            if self.seat is not None and player != self.seat:
                self.search.note_foreign_decision()

    def decide(self, view, decision, budget=None, capability=None):
        self.last_result = None
        if capability is None or not hasattr(capability, "fork_determinized"):
            return self.fallback.decide(view, decision)
        actions = legal_actions(decision, self.search.settings.enumerate_cap)
        if actions is not None and len(actions) == 1:
            return actions[0][1]
        t0 = time.perf_counter()
        result = drive(self.search.search_gen(capability, decision, budget), self.client)
        self.last_result = result
        st = self.search_stats
        st["decisions_searched"] += 1
        st["search_seconds"] += time.perf_counter() - t0
        for k in ("simulations", "evaluations", "forks", "fork_seconds"):
            st[k] += result.stats.get(k, 0)
        choice = result.choices[result.chosen]
        self.search.note_own_action(result.keys[result.chosen], capability.infoset().turn_number)
        return choice

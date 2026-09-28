"""Head-to-head play between network-backed agents, batched (Agent
Training Plan, M7 checkpoint gating and the M9 evaluation matrix).

A `Player` is a search agent (any regime / leaf estimator / simulation
count) on some network, a search-free network agent (policy or DMC), or a
plain registered bot. `play_paired` runs the paired-seed protocol
(sim/paired.py: both seat orders x both deck assignments, the same seed for
all four) with many games in flight: every round, all search decisions go
through `run_searches_grouped` -- one batched call per network -- and all
network-policy decisions through one `decide_many` per network.

With `sprt=(elo0, elo1)` it stops as soon as the sequential test decides
(chess-engine practice: clear-cut comparisons finish on a fraction of the
fixed budget) -- used for promotion gating.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from agent.agents.net_agent import NetAgent
from agent.search.core import Search, SearchSettings, legal_actions, run_searches_grouped
from agent.search.full_game import FullGame
from agent.search.leaf import make_evaluator
from agent.search.policies import make_policy
from agent.search.within_turn import WithinTurn
from bots.inference_client import InProcessInferenceClient
from bots.registry import make_agent
from keyforge.config import GameConfig
from keyforge.enums import Resample
from keyforge.game import Game
from keyforge.infoset import build_infoset
from sim.paired import PairedReport, arrangements
from sim.rate import sprt as sprt_test


@dataclass
class Player:
    name: str
    kind: str  # "search" | "net" | "bot"
    model: Any = None  # ml.infer_server.TorchModel (search/net)
    regime: str = "within_turn"
    leaf: str = "student"
    simulations: int = 100
    resample: str = "all"
    mode: str = "policy"  # net: policy | q
    bot: str = "heuristic"
    _client: Any = field(default=None, repr=False)

    def client(self):
        if self._client is None and self.model is not None:
            self._client = InProcessInferenceClient(self.model)
        return self._client


class _Cap:
    def __init__(self, game, viewer):
        self._game = game
        self.viewer = viewer

    def fork_determinized(self, rng, resample=Resample.ALL, *, backend="replay"):
        return self._game.fork_determinized(self.viewer, rng, resample=resample, backend=backend)

    def infoset(self):
        return build_infoset(self._game, self.viewer)


def _make_search(p: Player, seed: int) -> Search:
    policy = make_policy("heuristic", seed=seed)
    regime = WithinTurn(policy) if p.regime == "within_turn" else FullGame(policy)
    settings = SearchSettings(simulations=p.simulations, resample=Resample(p.resample), reuse=False)
    leaf = p.leaf if p.model is not None else "heuristic"
    return Search(regime, make_evaluator(leaf, seed=seed), policy, settings, seed=seed)


def play_paired(
    a: Player, b: Player, seeds, deck_x="fignor", deck_y="igor", *, max_turns: int = 200, concurrency: int = 32,
    sprt: Optional[Tuple[float, float]] = None, min_games: int = 40, log=None,
) -> PairedReport:
    arr = arrangements(list(seeds), deck_x, deck_y)
    rep = PairedReport(games=0, a_wins=0, b_wins=0, draws=0, forfeits=0)
    net_agents = {id(p): NetAgent(p.client(), mode=p.mode, seed=7) for p in (a, b) if p.kind == "net"}
    queue = list(range(len(arr)))
    active: List[dict] = []
    t0 = time.time()

    def start(i: int) -> dict:
        ar = arr[i]
        game = Game(GameConfig(decks=ar.decks, seed=ar.seed, max_turns=max_turns))
        seats = {ar.a_seat: a, 3 - ar.a_seat: b}
        slot = {"i": i, "game": game, "seats": seats, "searches": {}, "bots": {}}
        for seat, p in seats.items():
            if p.kind == "search":
                slot["searches"][seat] = _make_search(p, ar.seed * 2 + seat)
            elif p.kind == "bot":
                slot["bots"][seat] = make_agent(p.bot, seed=ar.seed + seat)
        return slot

    def advance(slot) -> None:
        g = slot["game"]
        while not g.is_over:
            d = g.pending_decision
            acts = legal_actions(d, 1024)
            if acts is not None and len(acts) == 1:
                g.submit(acts[0][1])
                continue
            bot = slot["bots"].get(d.player)
            if bot is not None:
                g.submit(bot.decide(g.view_for(d.player), d))
                continue
            return

    while queue or active:
        while queue and len(active) < concurrency:
            slot = start(queue.pop(0))
            advance(slot)
            active.append(slot)
        for slot in [s for s in active if s["game"].is_over]:
            active.remove(slot)
            g = slot["game"]
            a_seat = arr[slot["i"]].a_seat
            winner = (g.result or {}).get("winner")
            rep.games += 1
            rep.a_wins += winner == a_seat
            rep.b_wins += winner is not None and winner != a_seat
            rep.draws += winner is None
            rep.turns.append(g.turn_number)
            seat = rep.by_seat.setdefault(a_seat, [0, 0])
            seat[0] += winner == a_seat
            seat[1] += 1
        if sprt is not None and rep.games >= min_games and rep.sprt(*sprt) is not None:
            break
        if not active:
            continue
        pairs, meta = [], []
        net_batches: Dict[int, list] = {}
        for slot in active:
            g = slot["game"]
            d = g.pending_decision
            p = slot["seats"][d.player]
            if p.kind == "search":
                pairs.append((slot["searches"][d.player].search_gen(_Cap(g, d.player), d), p.client()))
                meta.append(slot)
            else:
                net_batches.setdefault(id(p), []).append(slot)
        if pairs:
            for slot, res in zip(meta, run_searches_grouped(pairs)):
                slot["game"].submit(res.choices[res.chosen])
                advance(slot)
        for pid, slots in net_batches.items():
            agent = net_agents[pid]
            reqs = [(None, s["game"].pending_decision, None, _Cap(s["game"], s["game"].pending_decision.player)) for s in slots]
            for s, choice in zip(slots, agent.decide_many(reqs)):
                s["game"].submit(choice)
                advance(s)
        if log is not None and rep.games and rep.games % 50 == 0:
            log(f"[arena] {a.name} vs {b.name}: {rep.games} games, score {rep.score:.3f} ({time.time() - t0:.0f}s)")
    return rep

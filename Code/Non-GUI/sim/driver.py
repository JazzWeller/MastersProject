"""In-process concurrent game driver (Agent Interface Plan, Milestone G).

Runs K games (or matches) concurrently in one process: each round, every
still-active slot's pending decision is grouped by which agent is
deciding, and each distinct agent's `decide_many` is called ONCE per round
with its whole batch for that round. The engine is already externally
driven (`pending_decision`/`submit`), so this needs no engine change, and
it covers `Match` exactly like `Game` through the same protocol.

Fault isolation: an agent that raises, returns an illegal choice, or blows
its wall-clock budget forfeits that game -- recorded with the full replay
record for debugging -- and the run continues with everything else.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from bots.base import BatchController, Budget, Controller, SyncBatchAdapter
from keyforge.capabilities import make_capability
from keyforge.config import GameConfig
from keyforge.enums import DecisionKind, PrivilegeLevel
from keyforge.game import Game
from keyforge.match import Match, MatchConfig

_NOT_FORCED = object()


class BudgetExceeded(Exception):
    """Raised by `SyncBatchAdapter`'s own wall-clock check (see
    `TimedSyncBatchAdapter`) to report which request in a batch blew its
    budget, without losing the results already computed for earlier ones."""

    def __init__(self, index: int):
        super().__init__(f"request {index} exceeded its wall-clock budget")
        self.index = index


class TimedSyncBatchAdapter(SyncBatchAdapter):
    """`SyncBatchAdapter`, but enforcing `Budget.wall_clock_seconds` per
    request -- the only budget dimension a driver can enforce on an agent
    that isn't cooperating, since `Budget.simulations` requires the agent
    to report its own count."""

    def decide_many(self, requests):
        results = []
        for i, (view, decision, budget, capability) in enumerate(requests):
            start = time.perf_counter()
            choice = self.inner.decide(view, decision, budget, capability)
            if budget is not None and budget.wall_clock_seconds is not None:
                if time.perf_counter() - start > budget.wall_clock_seconds:
                    raise BudgetExceeded(i)
            results.append(choice)
        return results


def derive_agent_seed(run_seed: int, game_index: int, seat: int) -> int:
    """Each agent's per-game seed derives from `(run seed, game index,
    seat)`, so any single game out of a parallel run reproduces
    standalone."""
    digest = hashlib.sha256(f"{run_seed!r}|{game_index}|{seat}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _forced_choice(decision):
    """The engine already auto-resolves CHOOSE_CARDS/ORDER_EFFECTS/etc.
    when there's only one legal *combination* (see Game.choose_cards and
    friends) -- what's left, and what this covers, is a Decision that was
    actually created with exactly one option (most commonly CHOOSE_ACTION
    with only EndTurn legal): 11.7% of all decisions, per the plan's own
    baseline. CHOOSE_CARDS/ORDER_EFFECTS are excluded even with exactly one
    option -- a CHOOSE_CARDS decision submits a LIST, and `min_n < max_n`
    (e.g. "purge 0-2 cards" over a single legal card) still has more than
    one legal submission (`[]` or `[card]`) despite there being one option."""
    if decision.kind in (DecisionKind.CHOOSE_CARDS, DecisionKind.ORDER_EFFECTS):
        return _NOT_FORCED
    if len(decision.options) == 1:
        return decision.options[0]
    return _NOT_FORCED


@dataclass
class GameResult:
    index: int
    winner: Optional[int]
    reason: str
    turns: Optional[int]
    choice_record: list
    config: Any
    forfeit: Optional[Dict[str, Any]] = None


@dataclass
class _Slot:
    index: int
    obj: Union[Game, Match]
    agents: Dict[int, BatchController]
    privilege: Dict[int, PrivilegeLevel]
    seat_of_agent: Dict[int, int]  # id(agent) -> the seat it was registered for (for on_game_start/observe bookkeeping)
    notified_start: set = field(default_factory=set)


def _is_match(obj) -> bool:
    return isinstance(obj, Match)


def _live_game(obj) -> Optional[Game]:
    """The `Game` a decision would fork/observe against, or `None` for a
    between-games match-level decision (BID_CHAINS / CHOOSE_FIRST_PLAYER)."""
    if _is_match(obj):
        return obj.current_game if (obj.current_game is not None and not obj.current_game.is_over) else None
    return obj


def _capability_for(obj, viewer: int, level: PrivilegeLevel):
    live = _live_game(obj)
    if live is None:
        return None
    return make_capability(level, live, viewer)


def _result_for(index: int, obj, forfeit=None) -> GameResult:
    result = getattr(obj, "result", None) or {}
    winner = result.get("winner")
    reason = result.get("reason", "")
    if forfeit is not None:
        # The engine's own `.result` is unset -- the driver abandoned this
        # game rather than the engine ending it -- so the forfeit itself is
        # what decides the winner: whoever didn't forfeit.
        loser = forfeit["seat"]
        winner = 3 - loser if loser in (1, 2) else winner
        reason = f"forfeit: {forfeit['reason']}"
    turns = obj.turn_number if not _is_match(obj) else None
    return GameResult(
        index=index,
        winner=winner,
        reason=reason,
        turns=turns,
        choice_record=list(obj.choice_record),
        config=obj.config,
        forfeit=forfeit,
    )


def run_games(
    n_games: int,
    make_config: Callable[[int], Union[GameConfig, MatchConfig]],
    make_agents: Callable[[int], Dict[int, Controller]],
    *,
    concurrency: int = 8,
    budget: Optional[Budget] = None,
    privilege: Optional[Dict[int, PrivilegeLevel]] = None,
    run_seed: int = 0,
    auto_resolve_forced: bool = True,
) -> List[GameResult]:
    """Plays `n_games` games (or matches -- `make_config` may return either
    a `GameConfig` or a `MatchConfig`), `concurrency` at a time.

    `make_agents(game_index)` returns `{seat: Controller}`; each is wrapped
    in a `TimedSyncBatchAdapter` unless it's already a `BatchController`.
    `privilege` (default: observation-only for every seat) maps seat ->
    `PrivilegeLevel`, applied to every game.
    """
    privilege = privilege or {}
    results: List[Optional[GameResult]] = [None] * n_games
    active: Dict[int, _Slot] = {}
    next_to_start = 0

    def start_slot(index: int) -> _Slot:
        config = make_config(index)
        obj = Match(config) if isinstance(config, MatchConfig) else Game(config)
        raw_agents = make_agents(index)
        agents = {seat: (a if isinstance(a, BatchController) else TimedSyncBatchAdapter(a)) for seat, a in raw_agents.items()}
        slot_privilege = {seat: privilege.get(seat, PrivilegeLevel.OBSERVATION) for seat in agents}
        seat_of_agent = {id(a): seat for seat, a in agents.items()}
        slot = _Slot(index=index, obj=obj, agents=agents, privilege=slot_privilege, seat_of_agent=seat_of_agent)
        fmt = config.format if isinstance(config, MatchConfig) else "archon"
        for seat, agent in agents.items():
            if id(agent) in slot.notified_start:
                continue
            slot.notified_start.add(id(agent))
            agent.on_game_start(seat, {"format": fmt, "game_index": index})
        return slot

    def finish_slot(slot: _Slot, forfeit=None) -> None:
        outcome_of = {}
        if forfeit is not None:
            loser = forfeit["seat"]
            winner = 3 - loser if loser in (1, 2) else None
            outcome_of = {1: (1 if winner == 1 else (-1 if winner == 2 else 0)), 2: (1 if winner == 2 else (-1 if winner == 1 else 0))}
        elif getattr(slot.obj, "result", None) is not None:
            winner = slot.obj.result.get("winner")
            outcome_of = {1: (0 if winner is None else (1 if winner == 1 else -1)), 2: (0 if winner is None else (1 if winner == 2 else -1))}
        notified = set()
        for seat, agent in slot.agents.items():
            if id(agent) in notified:
                continue
            notified.add(id(agent))
            agent.on_game_end(outcome_of.get(seat, 0))
        results[slot.index] = _result_for(slot.index, slot.obj, forfeit=forfeit)

    def notify_observers(slot: _Slot, decision, choice, new_events) -> None:
        notified = set()
        for agent in slot.agents.values():
            if id(agent) in notified:
                continue
            notified.add(id(agent))
            agent.observe((decision.kind, decision.player, choice))
            for e in new_events:
                agent.observe(e)

    while next_to_start < n_games and len(active) < concurrency:
        active[next_to_start] = start_slot(next_to_start)
        next_to_start += 1

    while active:
        by_agent: Dict[int, List] = {}
        agent_by_id: Dict[int, BatchController] = {}
        finished_now = []
        for idx, slot in active.items():
            obj = slot.obj
            # Auto-resolve forced decisions (and any Match/Game internal
            # advancement they trigger) before asking any agent anything.
            while not obj.is_over:
                d = obj.pending_decision
                forced = _forced_choice(d) if auto_resolve_forced else _NOT_FORCED
                if forced is _NOT_FORCED:
                    break
                before = len(_live_game(obj).log.events) if _live_game(obj) is not None else 0
                # `_forced_choice` only returns non-_NOT_FORCED when there's
                # exactly one option and the kind isn't CHOOSE_CARDS/
                # ORDER_EFFECTS -- `forced` is always encoded as index 0
                # (Milestone L: submit_index skips validate + re-encoding).
                obj.submit_index(0)
                live = _live_game(obj)
                new_events = live.log.events[before:] if live is not None else []
                notify_observers(slot, d, forced, new_events)
            if obj.is_over:
                finished_now.append(idx)
                continue
            d = obj.pending_decision
            agent = slot.agents[d.player]
            # PlayerView/MatchView -- the existing Controller contract
            # ("PlayerView stays as it is for the GUI and existing bots").
            # An agent that wants the richer Observation gets one on demand
            # via `capability.observation()` instead. Building one is ~26%
            # of engine wall time (Milestone L), so skip it entirely for an
            # agent that has declared it never looks at `view`.
            view = obj.view_for(d.player) if agent.needs_view else None
            cap = _capability_for(obj, d.player, slot.privilege[d.player])
            by_agent.setdefault(id(agent), []).append((idx, view, d, budget, cap))
            agent_by_id[id(agent)] = agent

        for idx in finished_now:
            finish_slot(active.pop(idx))

        for agent_id, batch in by_agent.items():
            agent = agent_by_id[agent_id]
            requests = [(view, d, b, cap) for (_idx, view, d, b, cap) in batch]
            try:
                choices = agent.decide_many(requests)
            except BudgetExceeded as e:
                choices = None
                bad_idx, _, bad_d, _, _ = batch[e.index]
                finish_slot(active.pop(bad_idx), forfeit={"seat": bad_d.player, "reason": "budget exceeded"})
            except Exception as e:  # noqa: BLE001 -- fault isolation is the point
                choices = None
                # We don't know which slot in the batch raised; forfeit
                # them all rather than guess (or silently drop some).
                for bad_idx, _, bad_d, _, _ in batch:
                    if bad_idx in active:
                        finish_slot(active.pop(bad_idx), forfeit={"seat": bad_d.player, "reason": f"agent raised: {e!r}"})
                continue
            if choices is None:
                continue
            for (idx, _view, d, _b, _cap), choice in zip(batch, choices):
                slot = active.get(idx)
                if slot is None:  # a BudgetExceeded already forfeited a later item this round
                    continue
                if not d.validate(choice):
                    finish_slot(active.pop(idx), forfeit={"seat": d.player, "reason": f"illegal choice {choice!r}"})
                    continue
                before = len(_live_game(slot.obj).log.events) if _live_game(slot.obj) is not None else 0
                slot.obj.submit(choice)
                live = _live_game(slot.obj)
                new_events = live.log.events[before:] if live is not None else []
                notify_observers(slot, d, choice, new_events)
                if slot.obj.is_over:
                    finish_slot(active.pop(idx))

        while next_to_start < n_games and len(active) < concurrency:
            active[next_to_start] = start_slot(next_to_start)
            next_to_start += 1

    return results  # type: ignore[return-value]

"""Controller interface used by bots, the text UI, the GUI, and sim/driver.py
(Agent Interface Plan, Milestone G).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple


@dataclass(frozen=True)
class Budget:
    """Compute allowed for one decision -- set by the harness, not the
    agent, so a compute-normalized comparison between agents is a harness
    setting rather than a per-agent flag. Both are advisory: nothing in the
    engine enforces them, since only the agent's own search loop knows how
    to spend either one. `None` in either field means "no limit"."""

    simulations: Optional[int] = None
    wall_clock_seconds: Optional[float] = None


class Controller(ABC):
    #: Whether a harness must build a `PlayerView`/`Observation` before
    #: calling `decide()` (Milestone L: "lazy observations" -- building one
    #: for every decision is ~26% of engine wall time, and is wasted work
    #: for an agent that never looks at it, e.g. `RandomBot`). Default True
    #: so every existing agent keeps getting a view unless it opts out.
    needs_view: bool = True

    @abstractmethod
    def decide(self, view, decision, budget: Optional[Budget] = None, capability: Optional[Any] = None):
        """Return a legal choice for `decision` given the (redacted) `view`.
        `budget`/`capability` default to `None` so every existing
        `Controller` (and every existing 2-arg call site) is unaffected --
        an agent that doesn't search can ignore both."""
        raise NotImplementedError

    def on_game_start(self, seat: int, config: dict) -> None:
        """Called once, before the first decision. `config` is a small,
        public dict (format, deck labels, ...) -- never the seed, which an
        agent needing its own determinism should derive independently
        (`sim/driver.py`'s own per-agent seeding), not read off the game."""
        return None

    def observe(self, event) -> None:
        """Called for every public event and decision, including the
        opponent's. A belief filter needs incremental updates on every
        opponent move; MCTS subtree reuse needs to know which edge was
        actually taken (`HeuristicBot`'s old `_last_mode` is already
        evidence that agents carry state across decisions). Default is a
        no-op, so an agent with no belief state needs no change."""
        return None

    def on_game_end(self, outcome: int) -> None:
        return None


class BatchController(Controller):
    """A `Controller` that can decide a whole batch of pending decisions
    (from different games) in one call -- for a policy/value network,
    batching is the difference between one GPU call and one per game.
    """

    @abstractmethod
    def decide_many(self, requests: List[Tuple[Any, Any, Optional[Budget], Optional[Any]]]) -> List[Any]:
        """`requests` is a list of `(view, decision, budget, capability)`
        tuples, matching `decide`'s own parameters; returns one choice per
        request, in order."""
        raise NotImplementedError


class SyncBatchAdapter(BatchController):
    """Wraps a synchronous `Controller` (`RandomBot`, `HeuristicBot`, ...)
    as a `BatchController` of batch size one each, so `sim/driver.py` only
    has to know about `BatchController` -- every existing agent runs
    through it completely unchanged."""

    def __init__(self, inner: Controller):
        self.inner = inner
        self.needs_view = inner.needs_view

    def decide(self, view, decision, budget: Optional[Budget] = None, capability: Optional[Any] = None):
        return self.inner.decide(view, decision, budget, capability)

    def decide_many(self, requests):
        return [self.inner.decide(view, decision, budget, capability) for (view, decision, budget, capability) in requests]

    def on_game_start(self, seat: int, config: dict) -> None:
        self.inner.on_game_start(seat, config)

    def observe(self, event) -> None:
        self.inner.observe(event)

    def on_game_end(self, outcome: int) -> None:
        self.inner.on_game_end(outcome)

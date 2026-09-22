"""Name -> agent constructor, plus its registered privilege level: the
single source of agents for `sim/simulate.py`, `sim/driver.py`, and the
GUI's opponent choice (Agent Interface Plan, Milestone I) -- replacing the
hardcoded `RandomBot`/`HeuristicBot` construction that used to live at each
of those call sites separately. A learned agent registers as
`name@checkpoint`; `make_agent` splits on the first `@` and hands the
checkpoint half to the registered factory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

from keyforge.enums import PrivilegeLevel

from .base import Controller
from .heuristic_bot import HeuristicBot
from .random_bot import RandomBot


@dataclass(frozen=True)
class AgentEntry:
    name: str
    make: Callable[..., Controller]  # (seed=None, checkpoint=None, **kwargs) -> Controller
    privilege: PrivilegeLevel = PrivilegeLevel.OBSERVATION
    description: str = ""


_REGISTRY: Dict[str, AgentEntry] = {}


def register(
    name: str,
    make: Callable[..., Controller],
    *,
    privilege: PrivilegeLevel = PrivilegeLevel.OBSERVATION,
    description: str = "",
) -> None:
    if name in _REGISTRY:
        raise ValueError(f"duplicate agent registration for {name!r}")
    _REGISTRY[name] = AgentEntry(name=name, make=make, privilege=privilege, description=description)


def make_agent(name: str, seed: Optional[int] = None, **kwargs) -> Controller:
    """`name` may carry a checkpoint suffix (`agent@checkpoint`); the part
    after `@`, if any, is passed to the factory as `checkpoint=`."""
    base_name, _, checkpoint = name.partition("@")
    if base_name not in _REGISTRY:
        raise KeyError(f"no agent registered as {base_name!r} (have: {sorted(_REGISTRY)})")
    entry = _REGISTRY[base_name]
    if checkpoint:
        kwargs.setdefault("checkpoint", checkpoint)
    return entry.make(seed=seed, **kwargs)


def privilege_of(name: str) -> PrivilegeLevel:
    base_name = name.partition("@")[0]
    return _REGISTRY[base_name].privilege


def available_agents() -> Dict[str, AgentEntry]:
    return dict(_REGISTRY)


def _make_random(seed=None, **_kwargs) -> Controller:
    return RandomBot(seed=seed)


def _make_heuristic(seed=None, bid_ceiling=4, **_kwargs) -> Controller:
    return HeuristicBot(seed=seed, bid_ceiling=bid_ceiling)


register("random", _make_random, description="Uniform random legal choice -- the weakest baseline.")
register("heuristic", _make_heuristic, description="Rules-aware heuristic opponent (fight/aember judgment, no search).")

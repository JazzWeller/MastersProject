"""Registers the Agent Training Plan's agents in `bots.registry` (interface
plan Milestone I), so the harness, `sim/simulate.py`, the diagnostics and
the GUI can name them:

- `search-within-turn`, `search-full-game` -- the M4/M5 search with the
  no-network heuristic evaluator (SEARCH privilege). `simulations=` sets
  the per-decision budget (default 200, the G2 operating point).
- `search-within-turn-privileged`, `search-full-game-privileged` -- the
  same, handed the exact fork (the hidden-information diagnostic's upper
  bound; PRIVILEGED).

Network agents (`net@<checkpoint>`, `search-*@<checkpoint>`) need torch in
the process that owns the model; `ml.registry_entries` adds those, and is
imported explicitly by a torch-side entry point, never by this module.

Idempotent: importing twice registers once.
"""

from __future__ import annotations

from bots import registry
from keyforge.enums import PrivilegeLevel, Resample

from ..search.core import SearchSettings
from .search_agent import SearchAgent

_RESAMPLE = {r.value: r for r in Resample}


def _search_factory(regime: str, exact: bool = False):
    def make(seed=None, simulations: int = 200, resample: str = "all", rollout: str = "heuristic", **_kw):
        settings = SearchSettings(simulations=int(simulations), resample=None if exact else _RESAMPLE[resample])
        return SearchAgent(regime, leaf="heuristic", rollout=rollout, settings=settings, seed=seed)

    return make


def register_all() -> None:
    have = registry.available_agents()
    for regime in ("within_turn", "full_game"):
        name = "search-" + regime.replace("_", "-")
        if name not in have:
            registry.register(name, _search_factory(regime), privilege=PrivilegeLevel.SEARCH,
                              description=f"M5 {regime} search, heuristic evaluator (no network).")
        if name + "-privileged" not in have:
            registry.register(name + "-privileged", _search_factory(regime, exact=True), privilege=PrivilegeLevel.PRIVILEGED,
                              description=f"M5 {regime} search on the exact fork (diagnostic upper bound).")


register_all()

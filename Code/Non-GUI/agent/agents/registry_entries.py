"""Registers the Agent Training Plan's agents in `bots.registry` (interface
plan Milestone I), so the harness, `sim/simulate.py`, the diagnostics and
the GUI can name them:

- `search-within-turn`, `search-full-game` -- the M4/M5 search with the
  no-network heuristic evaluator (SEARCH privilege). `simulations=` sets
  the per-decision budget (default 200, the G2 operating point).
- `search-within-turn-privileged`, `search-full-game-privileged` -- the
  same, handed the exact fork (the hidden-information diagnostic's upper
  bound; PRIVILEGED).

- `search-net@<port>[/<simulations>]` -- the within-turn search with a
  trained v1 network's leaf (`student`), asking `ml.infer_server` on
  127.0.0.1:<port> (or `host:port`) over TCP: the process that plays needs
  no torch (the GUI on Windows, the server in WSL). Searched as the Tier 2
  ladder rated it: chance_exact worlds, no tree reuse, 100 simulations
  unless given.

Idempotent: importing twice registers once.
"""

from __future__ import annotations

from bots import registry
from keyforge.enums import PrivilegeLevel, Resample

from ..search.core import SearchSettings
from .search_agent import SearchAgent

_RESAMPLE = {r.value: r for r in Resample}


def _search_factory(regime: str, exact: bool = False):
    def make(seed=None, simulations: int = 200, resample: str = "all", rollout: str = "heuristic", quiet_leaves: bool = False, **_kw):
        settings = SearchSettings(simulations=int(simulations), resample=None if exact else _RESAMPLE[resample])
        return SearchAgent(regime, leaf="heuristic", rollout=rollout, settings=settings, seed=seed, quiet_leaves=quiet_leaves)

    return make


def _network_search(seed=None, checkpoint: str = "6150", simulations: int = 100, **_kw):
    from bots.inference_client import RemoteInferenceClient

    where, _, sims = checkpoint.partition("/")
    host, _, port = where.rpartition(":")
    client = RemoteInferenceClient((host or "127.0.0.1", int(port)), b"keyforge")
    settings = SearchSettings(simulations=int(sims or simulations), resample=Resample.ALL, determinization="chance_exact",
                              reuse=False)
    return SearchAgent("within_turn", leaf="student", client=client, settings=settings, seed=seed)


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
    if "search-net" not in have:
        registry.register("search-net", _network_search, privilege=PrivilegeLevel.SEARCH,
                          description="within-turn search on a trained v1 network served by ml.infer_server "
                                      "(search-net@<port>[/<simulations>]).")


register_all()

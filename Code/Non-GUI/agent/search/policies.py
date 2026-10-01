"""Default and rollout policies for search (Agent Training Plan, M4/M5): a
`policy(world, decision) -> choice` callable used to play out the rest of a
turn (regime A's leaf) and to resolve decisions a regime doesn't branch on.

Every policy reads only the view of the player it's choosing for, *in the
search's own determinized world* -- never the true game.
"""

from __future__ import annotations

from typing import Callable

from bots.heuristic_bot import HeuristicBot
from bots.random_bot import RandomBot

Policy = Callable[[object, object], object]


def heuristic_policy(seed=None) -> Policy:
    """`HeuristicBot` -- the default: ~30k decisions/sec, and it finishes a
    turn in ~4 decisions."""
    bot = HeuristicBot(seed=seed)
    return lambda game, d: bot.decide(game.view_for(d.player), d)


def random_policy(seed=None) -> Policy:
    bot = RandomBot(seed=seed)
    return lambda game, d: bot.decide(None, d)


def network_policy(client, seed=None) -> Policy:
    """Network-greedy: one (unbatched) policy evaluation per decision --
    the accurate, expensive rollout option."""
    from keyforge.infoset import build_infoset

    from ..agents.net_agent import NetAgent

    agent = NetAgent(client, seed=seed)

    class _Cap:
        def __init__(self, game, pid):
            self.game, self.pid = game, pid

        def infoset(self):
            return build_infoset(self.game, self.pid)

    return lambda game, d: agent.decide(None, d, None, _Cap(game, d.player))


def make_policy(name: str, *, seed=None, client=None) -> Policy:
    if name == "heuristic":
        return heuristic_policy(seed)
    if name == "random":
        return random_policy(seed)
    if name == "network":
        if client is None:
            raise ValueError("the network rollout policy needs an inference client")
        return network_policy(client, seed)
    raise ValueError(f"unknown policy {name!r}")

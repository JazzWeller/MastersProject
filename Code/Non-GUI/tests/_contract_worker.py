#!/usr/bin/env python3
"""Worker for test_agent_contract.py's cross-process reproducibility check:
plays one game with the named agent at the given seed and prints its
choice record + result as JSON."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bots.heuristic_bot import HeuristicBot
from bots.random_bot import RandomBot
from keyforge.config import GameConfig
from keyforge.game import Game

_AGENTS = {
    "random": lambda seed: RandomBot(seed=seed),
    "heuristic": lambda seed: HeuristicBot(seed=seed),
}


def main():
    name, seed_str = sys.argv[1], sys.argv[2]
    seed = int(seed_str)
    factory = _AGENTS[name]
    game = Game(GameConfig(decks=("fignor", "igor"), seed=seed, max_turns=150))
    bots = {1: factory(seed), 2: factory(seed + 500)}
    while not game.is_over:
        d = game.pending_decision
        choice = bots[d.player].decide(game.view_for(d.player), d)
        game.submit(choice)
    print(json.dumps({"choice_record": game.choice_record, "result": game.result}))


if __name__ == "__main__":
    main()

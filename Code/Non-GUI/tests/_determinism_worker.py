#!/usr/bin/env python3
"""Worker for test_determinism.py: plays a fixed batch of seeded games and
prints one combined digest. Run as a subprocess with different
`PYTHONHASHSEED` values -- if a card effect ever introduces a
`for x in some_set:` whose order depends on hash randomization, the digest
changes and the parent test fails.
"""

import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bots.random_bot import RandomBot
from keyforge.config import GameConfig
from keyforge.game import Game

N_GAMES = 30


def main():
    h = hashlib.sha256()
    for seed in range(N_GAMES):
        config = GameConfig(decks=("fignor", "igor"), seed=seed, max_turns=80)
        game = Game(config)
        bots = {1: RandomBot(seed=seed), 2: RandomBot(seed=seed + 1000)}
        while not game.is_over:
            d = game.pending_decision
            choice = bots[d.player].decide(game.view_for(d.player), d)
            game.submit(choice)
        h.update(repr(game.choice_record).encode("utf-8"))
        h.update(repr(game.result).encode("utf-8"))
        h.update(str(game.turn_number).encode("utf-8"))
        h.update(game.state_hash().encode("utf-8"))
    print(h.hexdigest())


if __name__ == "__main__":
    main()

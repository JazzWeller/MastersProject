#!/usr/bin/env python3
"""Worker for test_agent_observation_o5.py's determinism test: encodes (v2)
every decision of a fixed batch of seeded games, both viewers, and prints
one digest. Run as a subprocess under different `PYTHONHASHSEED` values --
the digest must never change."""

import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.features_v2 import encode_v2  # noqa: E402
from tools.o1_acceptance import POOLS, bots_for, game_config  # noqa: E402
from keyforge.game import Game  # noqa: E402


def digest(n_games: int = 2) -> str:
    h = hashlib.sha256()
    for pool in POOLS:
        for i in range(n_games):
            config = game_config(pool, i)
            game = Game(config)
            bots = bots_for(i, config.seed)
            while not game.is_over:
                d = game.pending_decision
                for viewer in (1, 2):
                    h.update(encode_v2(game, viewer).to_bytes())
                game.submit(bots[d.player].decide(game.view_for(d.player), d))
    return h.hexdigest()


if __name__ == "__main__":
    print(digest(int(sys.argv[1]) if len(sys.argv) > 1 else 2))

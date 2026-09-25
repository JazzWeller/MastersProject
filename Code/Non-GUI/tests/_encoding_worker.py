#!/usr/bin/env python3
"""Worker for test_agent_training_m1.py's determinism test: encodes every
decision of a fixed batch of seeded games and prints one digest. Run as a
subprocess under different `PYTHONHASHSEED` values (and, by hand, under
other interpreters/platforms) -- the digest must never change."""

import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.features import encode
from bots.heuristic_bot import HeuristicBot
from keyforge.config import GameConfig
from keyforge.game import Game
from keyforge.infoset import build_infoset


def digest(n_games: int = 6) -> str:
    h = hashlib.sha256()
    for seed in range(n_games):
        game = Game(GameConfig(decks=("fignor", "igor"), seed=seed, max_turns=60))
        bots = {1: HeuristicBot(seed=seed), 2: HeuristicBot(seed=seed + 1)}
        while not game.is_over:
            d = game.pending_decision
            for viewer in (1, 2):
                h.update(encode(build_infoset(game, viewer)).to_bytes())
            game.submit(bots[d.player].decide(game.view_for(d.player), d))
    return h.hexdigest()


if __name__ == "__main__":
    print(digest(int(sys.argv[1]) if len(sys.argv) > 1 else 6))

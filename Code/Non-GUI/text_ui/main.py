"""Play a game from the command line: any mix of human/bot seats."""

from __future__ import annotations

import argparse

from bots.random_bot import RandomBot
from keyforge.config import GameConfig
from keyforge.game import Game

from .human_controller import HumanController
from .render import render_board


def build_controller(kind, seat, game):
    if kind == "human":
        return HumanController(game)
    return RandomBot(seed=None)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1", choices=["human", "random"], default="human")
    parser.add_argument("--p2", choices=["human", "random"], default="random")
    parser.add_argument("--p1-deck", default="fignor")
    parser.add_argument("--p2-deck", default="igor")
    parser.add_argument("--first", choices=["p1", "p2"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    args = parser.parse_args(argv)

    first_player = None
    if args.first == "p1":
        first_player = 1
    elif args.first == "p2":
        first_player = 2

    config = GameConfig(
        decks=(args.p1_deck, args.p2_deck),
        first_player=first_player,
        seed=args.seed,
        max_turns=args.max_turns,
    )
    game = Game(config)
    controllers = {1: build_controller(args.p1, 1, game), 2: build_controller(args.p2, 2, game)}

    while not game.is_over:
        d = game.pending_decision
        choice = controllers[d.player].decide(game.view_for(d.player), d)
        game.submit(choice)

    print(render_board(game.view_for(1)))
    print("\nResult:", game.result)


if __name__ == "__main__":
    main()

"""Play a game or match from the command line: any mix of human/bot seats."""

from __future__ import annotations

import argparse
import random

from bots.random_bot import RandomBot
from keyforge.cards.decks import random_deck
from keyforge.match import Match, MatchConfig

from .human_controller import HumanController
from .render import render_board


def build_controller(kind, seat, match):
    if kind == "human":
        return HumanController(match)
    return RandomBot(seed=None)


def _resolve_deck_arg(value: str, rng: random.Random, label: str):
    """`value` is a preset name, a path to a deck JSON file, or the literal
    string "random" for a freshly generated random legal deck."""
    if value == "random":
        return random_deck(rng, name=label)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1", choices=["human", "random"], default="human")
    parser.add_argument("--p2", choices=["human", "random"], default="random")
    parser.add_argument("--p1-deck", default="fignor", help="preset name, path to a deck JSON file, or 'random'")
    parser.add_argument("--p2-deck", default="igor", help="preset name, path to a deck JSON file, or 'random'")
    parser.add_argument("--format", choices=["archon", "reversal", "adaptive"], default="archon")
    parser.add_argument("--first", choices=["p1", "p2"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    args = parser.parse_args(argv)

    first_player = None
    if args.first == "p1":
        first_player = 1
    elif args.first == "p2":
        first_player = 2

    deck_rng = random.Random(args.seed)
    config = MatchConfig(
        format=args.format,
        decks=(
            _resolve_deck_arg(args.p1_deck, deck_rng, "Random P1"),
            _resolve_deck_arg(args.p2_deck, deck_rng, "Random P2"),
        ),
        first_player=first_player,
        seed=args.seed,
        max_turns=args.max_turns,
    )
    match = Match(config)
    controllers = {1: build_controller(args.p1, 1, match), 2: build_controller(args.p2, 2, match)}

    while not match.is_over:
        d = match.pending_decision
        choice = controllers[d.player].decide(match.view_for(d.player), d)
        match.submit(choice)

    if match.current_game is not None:
        print(render_board(match.current_game.view_for(1)))
    print("\nMatch result:", match.result)
    for i, g in enumerate(match.games, 1):
        print(f"  Game {i}: winner={g.winner} turns={g.turns} decks={g.seat_decks} chains={g.final_chains}")


if __name__ == "__main__":
    main()

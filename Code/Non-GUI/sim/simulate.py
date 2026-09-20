"""Headless batch runs: N games (or N matches), win-rate summary, invariant checks."""

from __future__ import annotations

import argparse
import math
import random

from bots.random_bot import RandomBot
from keyforge.cards.decks import random_deck
from keyforge.config import GameConfig
from keyforge.enums import DecisionKind
from keyforge.game import Game
from keyforge.match import Match, MatchConfig

# Card-count invariants only need to hold at the boundaries between top-level
# actions: mid-resolution, a card can be legitimately "in transit" (e.g.
# removed from the deck top by Wild Wormhole but not yet placed, while a
# nested CHOOSE_FLANK decision is pending).
_BOUNDARY_KINDS = (DecisionKind.CHOOSE_ACTION, DecisionKind.CHOOSE_HOUSE, DecisionKind.TAKE_ARCHIVE)


def check_invariants(game: Game):
    # Tally every reachable card by its OWNER, not by whichever zone/side it is
    # currently sitting in -- upgrades and controller-changed cards can end up
    # attached to or controlled by the other player.
    counts = {1: 0, 2: 0}
    for p in game.players.values():
        for zone in (p.deck.cards(), p.hand.cards(), p.discard.cards(), p.archive.cards(), p.purged.cards()):
            for c in zone:
                counts[c.owner] += 1
        for c in p.play_area.creatures:
            counts[c.owner] += 1
            for upg in c.type_object.upgrades:
                counts[upg.owner] += 1
            for under in c.under_cards:
                counts[under.owner] += 1
        for c in p.play_area.artifacts:
            counts[c.owner] += 1
            for under in c.under_cards:
                counts[under.owner] += 1
    for pid in (1, 2):
        assert counts[pid] == 36, f"player {pid} has {counts[pid]} cards, expected 36"
        p = game.players[pid]
        assert p.aember >= 0, f"player {pid} has negative aember"
        assert 0 <= p.chains <= 24, f"player {pid} chains out of range: {p.chains}"


def run_one(p1_deck, p2_deck, first, seed, max_turns, check_invariants_flag):
    config = GameConfig(decks=(p1_deck, p2_deck), first_player=first, seed=seed, max_turns=max_turns)
    game = Game(config)
    controllers = {1: RandomBot(seed=seed), 2: RandomBot(seed=(seed or 0) + 1 if seed is not None else None)}
    choices = []
    while not game.is_over:
        d = game.pending_decision
        choice = controllers[d.player].decide(game.view_for(d.player), d)
        choices.append(choice)
        game.submit(choice)
        if check_invariants_flag and (game.is_over or game.pending_decision.kind in _BOUNDARY_KINDS):
            check_invariants(game)
    return game.result, game.turn_number


def run_one_match(p1_deck, p2_deck, fmt, first, seed, max_turns, check_invariants_flag):
    config = MatchConfig(format=fmt, decks=(p1_deck, p2_deck), first_player=first, seed=seed, max_turns=max_turns)
    match = Match(config)
    controllers = {1: RandomBot(seed=seed), 2: RandomBot(seed=(seed or 0) + 1 if seed is not None else None)}
    while not match.is_over:
        d = match.pending_decision
        choice = controllers[d.player].decide(match.view_for(d.player), d)
        match.submit(choice)
        if check_invariants_flag and match.current_game is not None:
            g = match.current_game
            if g.is_over or (g.pending_decision is not None and g.pending_decision.kind in _BOUNDARY_KINDS):
                check_invariants(g)
    return match


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--matches", type=int, default=None, help="run N matches (--format) instead of N single games")
    parser.add_argument("--format", choices=["archon", "reversal", "adaptive"], default="archon")
    parser.add_argument("--p1", choices=["random"], default="random")
    parser.add_argument("--p2", choices=["random"], default="random")
    parser.add_argument("--p1-deck", default="fignor")
    parser.add_argument("--p2-deck", default="igor")
    parser.add_argument("--random-decks", action="store_true", help="give each game (or match) a fresh random legal deck per player, instead of --p1-deck/--p2-deck")
    parser.add_argument("--first", choices=["p1", "p2"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--check-invariants", action="store_true")
    args = parser.parse_args(argv)

    first_player = {"p1": 1, "p2": 2}.get(args.first)
    deck_rng = random.Random(args.seed)

    if args.matches is not None:
        _run_matches(args, first_player, deck_rng)
        return

    wins = {1: 0, 2: 0, None: 0}
    total_turns = 0
    for i in range(args.games):
        seed = (args.seed + i) if args.seed is not None else None
        p1_deck, p2_deck = _pick_decks(args, deck_rng)
        result, turns = run_one(p1_deck, p2_deck, first_player, seed, args.max_turns, args.check_invariants)
        wins[result["winner"]] += 1
        total_turns += turns

    print(f"Games: {args.games}")
    print(f"P1 wins: {wins[1]} ({wins[1]/args.games:.1%})")
    print(f"P2 wins: {wins[2]} ({wins[2]/args.games:.1%})")
    print(f"Draws (turn limit): {wins[None]} ({wins[None]/args.games:.1%})")
    print(f"Average turns: {total_turns/args.games:.1f}")


def _pick_decks(args, rng):
    if args.random_decks:
        return random_deck(rng, name="Random P1"), random_deck(rng, name="Random P2")
    return args.p1_deck, args.p2_deck


def _run_matches(args, first_player, deck_rng):
    wins = {1: 0, 2: 0, None: 0}
    reached_game3 = 0
    bids = []
    for i in range(args.matches):
        seed = (args.seed + i) if args.seed is not None else None
        p1_deck, p2_deck = _pick_decks(args, deck_rng)
        match = run_one_match(p1_deck, p2_deck, args.format, first_player, seed, args.max_turns, args.check_invariants)
        wins[match.result["winner"]] += 1
        if len(match.games) >= 3:
            reached_game3 += 1
        if match.bid is not None:
            bids.append(match.bid.amount)

    n = args.matches
    print(f"Matches ({args.format}): {n}")
    print(f"P1 wins: {wins[1]} ({wins[1]/n:.1%})")
    print(f"P2 wins: {wins[2]} ({wins[2]/n:.1%})")
    print(f"Undecided: {wins[None]} ({wins[None]/n:.1%})")
    if args.format == "adaptive":
        print(f"Reached game 3 (split 1-1): {reached_game3} ({reached_game3/n:.1%})")
        if bids:
            print(f"Average winning bid: {sum(bids)/len(bids):.1f} chains (over {len(bids)} bids)")


if __name__ == "__main__":
    main()

"""Headless batch runs: N games (or N matches), win-rate summary, invariant checks."""

from __future__ import annotations

import argparse
import math
import random

from bots.registry import available_agents, make_agent
from keyforge.cards.card_data import CARD_DEFS
from keyforge.cards.decks import random_deck
from keyforge.config import GameConfig
from keyforge.enums import BOUNDARY_KINDS as _BOUNDARY_KINDS
from keyforge.game import Game
from keyforge.match import Match, MatchConfig

# Card-count invariants only need to hold at the boundaries between top-level
# actions: mid-resolution, a card can be legitimately "in transit" (e.g.
# removed from the deck top by Wild Wormhole but not yet placed, while a
# nested CHOOSE_FLANK decision is pending). `_BOUNDARY_KINDS` name kept for
# every existing importer (tests/test_agent_contract.py); `enums.
# BOUNDARY_KINDS` is the one canonical definition now.


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

    # NOTE: armor_used_this_turn <= get_armor(card) is NOT a sound
    # after-the-fact invariant -- conditional armor (Shoulder Armor's
    # flank-only bonus, Red-Hot Armor's negation) can legitimately make
    # get_armor() drop *below* armor already spent earlier in the same
    # turn once the condition stops holding. The real guarantee (a hit
    # never absorbs more than was available *at that moment*) is
    # structural in steps.deal_damage and is covered directly by
    # test_steps.TestDealDamage instead of a fuzz invariant here.

    # Every Æmber currently sitting captured/placed on an in-play card must
    # trace back to a "capture" or "place_aember" log entry -- it can be
    # spent down by other means (Word of Returning), never manufactured.
    total_captured_ever = sum(
        e.data["amount"] for e in game.log.events if e.kind in ("capture", "place_aember")
    )
    currently_held = sum(
        c.aember_captured
        for p in game.players.values()
        for c in list(p.play_area.creatures) + list(p.play_area.artifacts)
    )
    assert currently_held <= total_captured_ever, (
        f"{currently_held} Æmber currently captured/placed on cards, but only "
        f"{total_captured_ever} was ever captured or placed"
    )


_USAGE_LOG_KIND_TO_CARD_KEY = {
    "play_card": "card",
    "reap": "card",
    "use_action": "card",
    "use_omni": "card",
    "fight": "attacker",
}


def collect_card_usage(game: Game, seen: set) -> None:
    """Records every card name that was played, reaped, fought with, or had
    its Action/Omni used at some point in `game`, for a coverage report
    across a whole batch of games (Code/PHASE_3_PLAN.md Milestone F: every
    one of the 370 cards should turn up at least once over a long run)."""
    for e in game.log.events:
        key = _USAGE_LOG_KIND_TO_CARD_KEY.get(e.kind)
        if key is not None and key in e.data:
            seen.add(e.data[key])


def run_one(p1_deck, p2_deck, first, seed, max_turns, check_invariants_flag, usage_tracker=None, p1_agent="random", p2_agent="random"):
    config = GameConfig(decks=(p1_deck, p2_deck), first_player=first, seed=seed, max_turns=max_turns)
    game = Game(config)
    controllers = {
        1: make_agent(p1_agent, seed=seed),
        2: make_agent(p2_agent, seed=(seed or 0) + 1 if seed is not None else None),
    }
    choices = []
    while not game.is_over:
        d = game.pending_decision
        agent = controllers[d.player]
        view = game.view_for(d.player) if agent.needs_view else None
        choice = agent.decide(view, d)
        choices.append(choice)
        game.submit(choice)
        if check_invariants_flag and (game.is_over or game.pending_decision.kind in _BOUNDARY_KINDS):
            check_invariants(game)
    if usage_tracker is not None:
        collect_card_usage(game, usage_tracker)
    return game.result, game.turn_number


def run_one_match(p1_deck, p2_deck, fmt, first, seed, max_turns, check_invariants_flag, p1_agent="random", p2_agent="random"):
    config = MatchConfig(format=fmt, decks=(p1_deck, p2_deck), first_player=first, seed=seed, max_turns=max_turns)
    match = Match(config)
    controllers = {
        1: make_agent(p1_agent, seed=seed),
        2: make_agent(p2_agent, seed=(seed or 0) + 1 if seed is not None else None),
    }
    while not match.is_over:
        d = match.pending_decision
        agent = controllers[d.player]
        view = match.view_for(d.player) if agent.needs_view else None
        choice = agent.decide(view, d)
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
    parser.add_argument("--p1", default="random", help=f"agent name from bots.registry (available: {sorted(available_agents())})")
    parser.add_argument("--p2", default="random", help=f"agent name from bots.registry (available: {sorted(available_agents())})")
    parser.add_argument("--p1-deck", default="fignor")
    parser.add_argument("--p2-deck", default="igor")
    parser.add_argument("--random-decks", action="store_true", help="give each game (or match) a fresh random legal deck per player, instead of --p1-deck/--p2-deck")
    parser.add_argument("--first", choices=["p1", "p2"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--check-invariants", action="store_true")
    parser.add_argument(
        "--coverage", action="store_true",
        help="report which of the full card pool was never played/reaped/fought/used across the run (implies --random-decks)",
    )
    args = parser.parse_args(argv)
    if args.coverage:
        args.random_decks = True

    first_player = {"p1": 1, "p2": 2}.get(args.first)
    deck_rng = random.Random(args.seed)

    if args.matches is not None:
        _run_matches(args, first_player, deck_rng)
        return

    wins = {1: 0, 2: 0, None: 0}
    total_turns = 0
    usage_tracker = set() if args.coverage else None
    for i in range(args.games):
        seed = (args.seed + i) if args.seed is not None else None
        p1_deck, p2_deck = _pick_decks(args, deck_rng)
        result, turns = run_one(
            p1_deck, p2_deck, first_player, seed, args.max_turns, args.check_invariants, usage_tracker,
            p1_agent=args.p1, p2_agent=args.p2,
        )
        wins[result["winner"]] += 1
        total_turns += turns

    print(f"Games: {args.games}")
    print(f"P1 wins: {wins[1]} ({wins[1]/args.games:.1%})")
    print(f"P2 wins: {wins[2]} ({wins[2]/args.games:.1%})")
    print(f"Draws (turn limit): {wins[None]} ({wins[None]/args.games:.1%})")
    print(f"Average turns: {total_turns/args.games:.1f}")

    if usage_tracker is not None:
        _report_coverage(usage_tracker)


def _report_coverage(seen: set) -> None:
    all_names = set(CARD_DEFS)
    missing = sorted(all_names - seen)
    print(f"\nCard coverage: {len(all_names) - len(missing)}/{len(all_names)} played/reaped/fought/used at least once.")
    if missing:
        print(f"Never touched ({len(missing)}):")
        for name in missing:
            print(f"  {name}")


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
        match = run_one_match(
            p1_deck, p2_deck, args.format, first_player, seed, args.max_turns, args.check_invariants,
            p1_agent=args.p1, p2_agent=args.p2,
        )
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

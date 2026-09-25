#!/usr/bin/env python3
"""The M1 no-leak acceptance check at full scale (Agent Training Plan,
Milestone M1): over N seeded games, at every decision and for both
viewers, every entity of the encoded observation is placed only where
that viewer is entitled to know it is.

Run from Code/Non-GUI: `python -m tools.check_encoder --games 1000`
(the unit suite runs the same `check_entitlement` on a smaller sample).
"""

from __future__ import annotations

import argparse
import random
import time
from typing import Dict, List, Optional, Tuple

from keyforge.infoset import KNOWN_TO_ME, KNOWN_TO_THEM, ZONE_NAMES, InfoSet, build_infoset

_LOCATED_PUBLIC = {"discard", "purged", "creature", "artifact", "upgrade_attached"}


def _true_locations(game) -> Dict[int, Tuple[str, int, Optional[object]]]:
    """iid -> (zone, side pid, host card or None), the ground truth."""
    where: Dict[int, Tuple[str, int, Optional[object]]] = {}
    for pid, p in game.players.items():
        for zone, cards in (
            ("hand", p.hand.cards()), ("deck", p.deck.cards()), ("archive", p.archive.cards()),
            ("discard", p.discard.cards()), ("purged", p.purged.cards()),
        ):
            for c in cards:
                where[c.instance_id] = (zone, pid, None)
        for c in p.play_area.creatures:
            where[c.instance_id] = ("creature", pid, None)
            for u in c.type_object.upgrades:
                where[u.instance_id] = ("upgrade_attached", u.controller, c)
            for u in c.under_cards:
                where[u.instance_id] = ("under", pid, c)
        for c in p.play_area.artifacts:
            where[c.instance_id] = ("artifact", pid, None)
            for u in c.under_cards:
                where[u.instance_id] = ("under", pid, c)
    return where


def check_entitlement(game, viewer: int, info: InfoSet) -> List[str]:
    """Every way `info` could place a card somewhere `viewer` isn't
    entitled to know it is -- or somewhere it isn't. Empty means clean."""
    problems = []
    opp = 3 - viewer
    opp_hand_revealed = viewer in game.players[opp].hand_revealed_to
    truth = _true_locations(game)
    for i, iid in enumerate(info.entity_iids):
        zone_name = ZONE_NAMES[info.zones[i]]
        true = truth.get(iid)  # None: in no zone (limbo)
        card = game.card_by_id(iid)
        side_name, _, rest = zone_name.partition("_")
        flags = info.flags[i]

        def bad(msg):
            problems.append(f"{card.name}#{iid} for viewer {viewer}: {msg} (encoded {zone_name}, truly {true and true[:2]})")

        if zone_name == "opp_unseen":
            ok = true is not None and (
                (true[0] in ("deck", "archive") and true[1] == opp)
                or (true[0] == "hand" and true[1] == opp and not opp_hand_revealed)
                or (true[0] == "under" and card.owner != viewer)
            )
            if not ok:
                bad("marked unseen but it isn't hidden from the viewer")
            if flags & (KNOWN_TO_ME | KNOWN_TO_THEM) or i in info.inplay:
                bad("an unseen card carries location-dependent fields")
            continue
        if zone_name == "my_deck_unordered":
            if true != ("deck", viewer, None):
                bad("marked in my deck but isn't")
            continue
        if rest == "limbo":
            if true is not None:
                bad("marked in limbo but is in a zone")
            continue
        if zone_name == "none":
            bad("'none' is only for between-games match decisions")
            continue
        want_pid = viewer if side_name == "my" else opp
        if true is None or true[0] != rest or true[1] != want_pid:
            bad("placed in the wrong zone")
            continue
        # Located in a specific zone: that must be something the viewer can see.
        if rest == "hand" and want_pid == opp and not opp_hand_revealed:
            bad("opponent's hand card located without a reveal")
        if rest == "archive" and want_pid == opp:
            bad("opponent's archive card located")
        if rest == "under" and card.owner != viewer:
            bad("a facedown card located by someone who didn't place it")
        if not flags & KNOWN_TO_ME:
            bad("located card not flagged known_to_me")
    # Conversely: every card in a zone hidden from the viewer must be unseen.
    for iid, (zone, pid, _host) in truth.items():
        i = info.index_of_iid.get(iid)
        if i is None:
            continue
        hidden = pid == opp and (zone in ("deck", "archive") or (zone == "hand" and not opp_hand_revealed))
        if hidden and ZONE_NAMES[info.zones[i]] != "opp_unseen":
            problems.append(f"iid {iid} is hidden in the opponent's {zone} but encoded {ZONE_NAMES[info.zones[i]]}")
    return problems


def run(n_games: int, seed: int = 0, random_decks_every: int = 2) -> Tuple[int, int, List[str]]:
    from bots.heuristic_bot import HeuristicBot
    from bots.random_bot import RandomBot
    from keyforge.cards.decks import random_deck
    from keyforge.config import GameConfig
    from keyforge.game import Game

    deck_rng = random.Random(seed)
    decisions = 0
    problems: List[str] = []
    for g in range(n_games):
        s = seed * 1_000_003 + g
        if random_decks_every and g % random_decks_every:
            decks = (random_deck(deck_rng, "R1"), random_deck(deck_rng, "R2"))
            bots = {1: RandomBot(seed=s), 2: RandomBot(seed=s + 1)}
        else:
            decks = ("fignor", "igor")
            bots = {1: HeuristicBot(seed=s), 2: HeuristicBot(seed=s + 1)}
        game = Game(GameConfig(decks=decks, seed=s, max_turns=60))
        while not game.is_over:
            d = game.pending_decision
            for viewer in (1, 2):
                found = check_entitlement(game, viewer, build_infoset(game, viewer))
                if found:
                    problems.extend(f"game {g} turn {game.turn_number}: {p}" for p in found)
            decisions += 1
            game.submit(bots[d.player].decide(game.view_for(d.player), d))
    return n_games, decisions, problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    t0 = time.perf_counter()
    games, decisions, problems = run(args.games, args.seed)
    dt = time.perf_counter() - t0
    print(f"{games} games, {decisions} decisions x 2 viewers checked in {dt:.1f}s")
    if problems:
        print(f"{len(problems)} PROBLEMS, first 20:")
        for p in problems[:20]:
            print("  " + p)
        raise SystemExit(1)
    print("no leaks: every entity placed only where its viewer is entitled to know it is")


if __name__ == "__main__":
    main()

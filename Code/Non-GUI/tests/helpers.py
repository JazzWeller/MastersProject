"""Shared test helpers: build a game, place cards directly, and drive
generator-based effects with a scripted sequence of decision answers."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keyforge.cards.card import Card
from keyforge.cards.card_data import get_card_def
from keyforge.config import GameConfig
from keyforge.enums import DecisionKind
from keyforge.game import Game


def default_chooser(decision):
    if decision.kind in (DecisionKind.CHOOSE_CARDS,):
        return list(decision.options[: decision.min_n])
    if decision.kind == DecisionKind.ORDER_EFFECTS:
        return list(decision.options)
    return decision.options[0]


def drive(gen, answers=None):
    """Run a generator-based effect to completion. `answers` is a list of
    exact choices to feed to successive Decisions in order; once exhausted,
    remaining decisions (if any) fall back to a safe default."""
    answers = list(answers) if answers else []
    try:
        d = next(gen)
    except StopIteration:
        return
    while True:
        choice = answers.pop(0) if answers else default_chooser(d)
        try:
            d = gen.send(choice)
        except StopIteration:
            return


def new_game(p1_deck="igor", p2_deck="igor", seed=1, max_turns=50, mulligans=(False, False)):
    """A game advanced to player 1's first CHOOSE_ACTION decision."""
    game = Game(GameConfig(decks=(p1_deck, p2_deck), first_player=1, seed=seed, max_turns=max_turns))
    game.submit(mulligans[0])
    game.submit(mulligans[1])
    while game.pending_decision.kind != DecisionKind.CHOOSE_ACTION:
        d = game.pending_decision
        if d.kind == DecisionKind.TAKE_ARCHIVE:
            game.submit(False)
        else:
            game.submit(default_chooser(d))
    return game


def make_card(name: str, owner: int, controller: int = None) -> Card:
    card = Card(get_card_def(name), owner)
    card.controller = controller if controller is not None else owner
    return card


def put_creature(game, pid, name, exhausted=False, can_be_used=True, flank=None):
    card = make_card(name, pid)
    game.players[pid].play_area.add_creature(card, flank)
    game._cards_by_id[card.instance_id] = card
    cdef = card.card_def
    if cdef.register_passive:
        cdef.register_passive(game, card)
    card.Exhausted = exhausted
    card.CanBeUsed = can_be_used
    return card


def put_artifact(game, pid, name, exhausted=False, can_be_used=True):
    card = make_card(name, pid)
    game.players[pid].play_area.add_artifact(card)
    game._cards_by_id[card.instance_id] = card
    cdef = card.card_def
    if cdef.register_passive:
        cdef.register_passive(game, card)
    card.Exhausted = exhausted
    card.CanBeUsed = can_be_used
    return card


def hand_card(game, pid, name):
    card = make_card(name, pid)
    game.players[pid].hand.add(card)
    game._cards_by_id[card.instance_id] = card
    return card


def run_hook(game, hook, card, answers=None):
    """Call a (game, card) -> generator effect hook and drive it to completion."""
    drive(hook(game, card), answers)

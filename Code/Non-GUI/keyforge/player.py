"""Player state and get_* functions."""

from __future__ import annotations

from typing import Dict, Optional

from .effects.effect_object import ActiveEffectList
from .zones import Archive, Deck, DiscardPile, Hand, PurgedZone, PlayArea


class Player:
    def __init__(self, player_id: int):
        self.id = player_id
        self.deck = Deck()
        self.discard = DiscardPile()
        self.hand = Hand()
        self.archive = Archive()
        self.purged = PurgedZone()
        self.play_area = PlayArea()

        self.aember = 0
        self.keys = 0
        self.chains = 0

        # base values for get_* variables
        self.CanKeyForge = True
        self.CanPlayActions = True
        self.CanPlayCreatures = True
        self.CardDrawModifier = 0
        self.CardPlayedLimit = None  # None = unlimited
        self.CardsPlayed: Dict[str, int] = {}
        self.DrawUpToLimit = 6
        self.KeyForgeCost = 6
        self.NonLogosCardsPlayable = 0
        self.HouseSelection = None  # forced house, or None

        self.selected_house = None
        self.cards_played_or_discarded_this_turn = 0
        self.hand_plays_this_turn = 0
        self.used_this_turn: Dict[str, int] = {}  # card name -> uses (rule of 6)
        self.archive_choice_made_this_turn = False
        self.all_cards: list = []  # the full 36-card pool this player owns, set at setup

    # --- get_* accessors: apply active DurationEffects on top of the base value ---

    def _apply(self, game, variable: str, base):
        value = base
        for effect in game.active_effects.duration_effects_for(variable, self.id):
            value = effect.apply(value, game)
        return value

    def get_can_key_forge(self, game) -> bool:
        return self._apply(game, "CanKeyForge", self.CanKeyForge)

    def get_can_play_actions(self, game) -> bool:
        return self._apply(game, "CanPlayActions", self.CanPlayActions)

    def get_can_play_creatures(self, game) -> bool:
        return self._apply(game, "CanPlayCreatures", self.CanPlayCreatures)

    def get_card_draw_modifier(self, game) -> int:
        return self._apply(game, "CardDrawModifier", self.CardDrawModifier)

    def get_card_played_limit(self, game) -> Optional[int]:
        return self._apply(game, "CardPlayedLimit", self.CardPlayedLimit)

    def get_draw_up_to_limit(self, game) -> int:
        return max(0, self._apply(game, "DrawUpToLimit", self.DrawUpToLimit))

    def get_key_forge_cost(self, game) -> int:
        return max(0, self._apply(game, "KeyForgeCost", self.KeyForgeCost))

    def get_non_logos_cards_playable(self, game) -> int:
        return max(0, self._apply(game, "NonLogosCardsPlayable", self.NonLogosCardsPlayable))

    def get_house_selection(self, game):
        return self._apply(game, "HouseSelection", self.HouseSelection)

    def reset_turn_counters(self):
        self.CardsPlayed = {}
        self.used_this_turn = {}
        self.cards_played_or_discarded_this_turn = 0
        self.hand_plays_this_turn = 0
        self.NonLogosCardsPlayable = 0
        self.selected_house = None
        self.HouseSelection = None
        self.archive_choice_made_this_turn = False

    def uses_of(self, card_name: str) -> int:
        return self.used_this_turn.get(card_name, 0)

    def record_use(self, card_name: str) -> None:
        self.used_this_turn[card_name] = self.used_this_turn.get(card_name, 0) + 1

    def record_play(self, card_name: str) -> None:
        self.CardsPlayed[card_name] = self.CardsPlayed.get(card_name, 0) + 1

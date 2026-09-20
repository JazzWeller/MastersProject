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
        self.ReapGainBecomesSteal = False  # Dimension Door: reap gain is stolen from the opponent instead
        self.CanFight = True  # Foggify: cannot use creatures to fight
        self.CannotUseCards = False  # Skippy Timehog: cannot reap/fight/action/omni (playing/discarding still allowed)
        self.CanPlayCards = True  # Treasure Map: cannot play any card for the rest of the turn
        self.FirstCreatureEntersReady = False  # Speed Sigil
        self.CannotBeDealtDamage = False  # Shield of Justice, Potion of Invulnerability: all your creatures, this turn
        self.CanOnlyFight = False  # Horseman of War: this turn, friendly creatures can only be used to fight
        self.CannotBeStolenFrom = False  # The Vaultkeeper
        self.next_entry_ready = False  # Soft Landing: the next creature/artifact played this turn enters ready
        self.next_mars_creature_ready = False  # Blypyp: the next Mars creature played this turn enters ready
        self.creatures_played_this_turn = 0

        self.selected_house = None
        self.cards_played_or_discarded_this_turn = 0
        self.hand_plays_this_turn = 0
        self.used_this_turn: Dict[str, int] = {}  # card name -> uses (rule of 6)
        self.archive_choice_made_this_turn = False
        self.all_cards: list = []  # the full 36-card pool this player owns, set at setup

        # Psychic Bug, Imperial Traitor, A Fair Game: pids this player's
        # hand is currently revealed to (in addition to themself, who can
        # always see it). Cleared at the end of every turn.
        self.hand_revealed_to: set = set()

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

    def get_reap_gain_becomes_steal(self, game) -> bool:
        return self._apply(game, "ReapGainBecomesSteal", self.ReapGainBecomesSteal)

    def get_can_fight(self, game) -> bool:
        return self._apply(game, "CanFight", self.CanFight)

    def get_cannot_use_cards(self, game) -> bool:
        return self._apply(game, "CannotUseCards", self.CannotUseCards)

    def get_can_play_cards(self, game) -> bool:
        return self._apply(game, "CanPlayCards", self.CanPlayCards)

    def get_first_creature_enters_ready(self, game) -> bool:
        return self._apply(game, "FirstCreatureEntersReady", self.FirstCreatureEntersReady)

    def get_cannot_be_dealt_damage(self, game) -> bool:
        return self._apply(game, "CannotBeDealtDamage", self.CannotBeDealtDamage)

    def get_can_only_fight(self, game) -> bool:
        return self._apply(game, "CanOnlyFight", self.CanOnlyFight)

    def get_cannot_be_stolen_from(self, game) -> bool:
        return self._apply(game, "CannotBeStolenFrom", self.CannotBeStolenFrom)

    def get_use_permitted_extra(self, game) -> frozenset:
        """Houses (or the wildcard `True`, meaning every house) this
        player's creatures may be fully used as this turn -- reap, action
        and fight, not just fight (Sigil of Brotherhood). See also
        `get_fight_permitted_extra`, which only ever grants fighting."""
        extra = set()
        for e in game.active_effects.duration_effects_for("UsePermittedHouse", self.id):
            if e.is_active(game):
                extra.add(e.value)
        return frozenset(extra)

    def get_artifact_use_toll(self, game):
        """(amount, receiver_pid), or None -- Tentacus: pay to use an artifact."""
        for e in game.active_effects.duration_effects_for("ArtifactUseToll", self.id):
            if e.is_active(game):
                return e.value
        return None

    def get_artifact_play_toll(self, game):
        """(amount, receiver_pid), or None -- Customs Office: pay to play an artifact."""
        for e in game.active_effects.duration_effects_for("ArtifactPlayToll", self.id):
            if e.is_active(game):
                return e.value
        return None

    def get_fight_permitted_extra(self, game) -> frozenset:
        """Houses this player's creatures may additionally fight as this
        turn, on top of the active house and versatile (Brothers in
        Battle: a chosen house; Follow the Leader: the sentinel `True`,
        meaning every house). Reap/Action/Omni are unaffected -- these
        cards grant fighting permission only."""
        extra = set()
        for e in game.active_effects.duration_effects_for("FightPermittedHouse", self.id):
            if e.is_active(game):
                extra.add(e.value)
        return frozenset(extra)

    def get_cannot_choose_houses(self, game) -> frozenset:
        """Houses this player currently cannot choose as their active house
        (Restringuntus). Unlike the scalar get_* accessors, several sources
        can each ban a different house at once, so this unions every active
        effect's value rather than folding them through one base value."""
        banned = set()
        for e in game.active_effects.duration_effects_for("CannotChooseHouse", self.id):
            if e.is_active(game):
                banned.add(e.value)
        return frozenset(banned)

    def reset_turn_counters(self):
        self.CardsPlayed = {}
        self.used_this_turn = {}
        self.cards_played_or_discarded_this_turn = 0
        self.hand_plays_this_turn = 0
        self.NonLogosCardsPlayable = 0
        self.selected_house = None
        self.HouseSelection = None
        self.archive_choice_made_this_turn = False
        self.creatures_played_this_turn = 0
        self.next_entry_ready = False
        self.next_mars_creature_ready = False

    def uses_of(self, card_name: str) -> int:
        return self.used_this_turn.get(card_name, 0)

    def record_use(self, card_name: str) -> None:
        self.used_this_turn[card_name] = self.used_this_turn.get(card_name, 0) + 1

    def record_play(self, card_name: str) -> None:
        self.CardsPlayed[card_name] = self.CardsPlayed.get(card_name, 0) + 1

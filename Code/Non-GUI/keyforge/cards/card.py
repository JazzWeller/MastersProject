"""Card and TypeObject definitions."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from ..enums import CardType, House

_instance_counter = itertools.count(1)


class TypeObject:
    """Base for the per-type behaviour object attached to a Card."""

    def __init__(self):
        self.card: Optional["Card"] = None


class ActionType(TypeObject):
    pass


class ArtifactType(TypeObject):
    pass


class CreatureType(TypeObject):
    def __init__(self, power: int, armor: int = 0):
        super().__init__()
        self.base_power = power
        self.base_armor = armor
        self.damage = 0
        self.armor_used_this_turn = 0
        self.upgrades: List["Card"] = []


class UpgradeType(TypeObject):
    def __init__(self):
        super().__init__()
        self.host: Optional["Card"] = None


@dataclass
class CardDef:
    """Static definition of a printed card, shared by every physical copy."""

    id: int
    name: str
    house: House
    type: CardType
    aember_on_play: int = 0
    aember_captured: int = 0
    tags: tuple = ()
    image: Optional[str] = None
    power: int = 0
    armor: int = 0
    on_play: Optional[Callable] = None          # generator effect function(game, card) or None
    on_reap: Optional[Callable] = None
    on_fight: Optional[Callable] = None
    on_action: Optional[Callable] = None
    on_omni: Optional[Callable] = None
    on_destroyed: Optional[Callable] = None
    register_passive: Optional[Callable] = None  # (game, card) -> None, called on enter play
    unregister_passive: Optional[Callable] = None  # (game, card) -> None, called on leave play


class Card:
    """A single physical instance of a card."""

    def __init__(self, card_def: CardDef, owner: int):
        self.instance_id = next(_instance_counter)
        self.id = card_def.id
        self.name = card_def.name
        self.house = card_def.house
        self.type = card_def.type
        self.tags = card_def.tags
        self.aember_on_play = card_def.aember_on_play
        self.aember_captured = card_def.aember_captured
        self.image = card_def.image
        self.card_def = card_def
        self.owner = owner
        self.controller = owner

        if card_def.type == CardType.CREATURE:
            self.type_object: TypeObject = CreatureType(card_def.power, card_def.armor)
        elif card_def.type == CardType.ARTIFACT:
            self.type_object = ArtifactType()
        elif card_def.type == CardType.UPGRADE:
            self.type_object = UpgradeType()
        else:
            self.type_object = ActionType()
        self.type_object.card = self

        # per-card variables (Elusive/Skirmish are granted by passive registration,
        # not derived from flavor tags)
        self.Elusive = False
        self.IgnoreElusive = False
        self.Skirmish = False
        self.CanBeUsed = False
        self.Exhausted = True
        self.destroyed = False
        self.fought_this_turn = False  # was Target of a fight (for Elusive)
        self.flank: Optional[str] = None
        # extra abilities granted by upgrades: event name -> list of (game, card) -> generator
        self.extra_triggers: dict = {"after_reap": [], "after_fight": [], "destroyed": []}
        # overrides where a destroyed card goes instead of the discard pile (Bad Penny)
        self.destined_zone: Optional[str] = None

    def __repr__(self):
        return f"<Card {self.name} #{self.instance_id}>"

    def reset_on_leave_play(self):
        if isinstance(self.type_object, CreatureType):
            self.type_object.damage = 0
            self.type_object.armor_used_this_turn = 0
            self.type_object.upgrades = []
        self.Exhausted = True
        self.destroyed = False
        self.fought_this_turn = False
        self.CanBeUsed = False
        self.flank = None
        self.destined_zone = None

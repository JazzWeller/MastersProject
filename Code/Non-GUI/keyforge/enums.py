"""Enumerations shared across the engine."""

from enum import Enum, auto


class House(Enum):
    DIS = "Dis"
    LOGOS = "Logos"
    SHADOWS = "Shadows"


class CardType(Enum):
    ACTION = "Action"
    ARTIFACT = "Artifact"
    CREATURE = "Creature"
    UPGRADE = "Upgrade"


class Trigger(Enum):
    PLAY = auto()
    AFTER_REAP = auto()
    AFTER_FIGHT = auto()
    BEFORE_FIGHT = auto()
    FIGHTING = auto()
    DESTROYED = auto()
    OMNI = auto()
    ACTION = auto()


class Zone(Enum):
    DECK = auto()
    HAND = auto()
    DISCARD = auto()
    ARCHIVE = auto()
    PURGED = auto()
    PLAY_CREATURE = auto()
    PLAY_ARTIFACT = auto()


class Flank(Enum):
    LEFT = auto()
    CENTER = auto()
    RIGHT = auto()


class DecisionKind(Enum):
    MULLIGAN = auto()
    CHOOSE_HOUSE = auto()
    TAKE_ARCHIVE = auto()
    CHOOSE_ACTION = auto()
    CHOOSE_CARDS = auto()
    CHOOSE_FLANK = auto()
    CHOOSE_HOUSE_FOR_EFFECT = auto()
    ORDER_EFFECTS = auto()
    YES_NO = auto()

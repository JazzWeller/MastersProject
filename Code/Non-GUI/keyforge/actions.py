"""CHOOSE_ACTION option types."""

from __future__ import annotations

from dataclasses import dataclass

from .cards.card import Card


@dataclass(frozen=True)
class PlayCard:
    card: Card


@dataclass(frozen=True)
class DiscardCard:
    card: Card


@dataclass(frozen=True)
class UseAction:
    card: Card


@dataclass(frozen=True)
class UseOmni:
    card: Card


@dataclass(frozen=True)
class Reap:
    card: Card


@dataclass(frozen=True)
class Fight:
    card: Card


@dataclass(frozen=True)
class EndTurn:
    pass

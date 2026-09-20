"""Read-only player view with hidden information removed."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .cards.card import Card


@dataclass
class PlayerPublicState:
    id: int
    aember: int
    keys: int
    chains: int
    hand_count: int
    hand: Optional[List[Card]]  # None if hidden (opponent's hand)
    archive_count: int
    archive: Optional[List[Card]]  # None if hidden
    deck_count: int
    discard: List[Card]
    purged: List[Card]
    creatures: List[Card]
    artifacts: List[Card]
    selected_house: object
    decklist: List[Card]  # the full 36-card list, unordered, always visible


@dataclass
class ActiveEffectSummary:
    source_name: str
    source_iid: int
    player_affected: int
    variable: str
    op: str
    value: object
    remaining_duration: int  # effect_object.INFINITE (-1) means infinite


@dataclass
class PlayerView:
    viewer: int
    active_player: int
    turn_number: int
    players: dict  # pid -> PlayerPublicState
    log_tail: list
    active_effects: List[ActiveEffectSummary] = field(default_factory=list)

    def me(self) -> PlayerPublicState:
        return self.players[self.viewer]

    def opponent(self) -> PlayerPublicState:
        return self.players[3 - self.viewer]


def build_view(game, viewer: int) -> PlayerView:
    states = {}
    for pid, player in game.players.items():
        visible_hand = player.hand.cards() if (pid == viewer or viewer in player.hand_revealed_to) else None
        visible_archive = player.archive.cards() if pid == viewer else None
        states[pid] = PlayerPublicState(
            id=pid,
            aember=player.aember,
            keys=player.keys,
            chains=player.chains,
            hand_count=len(player.hand),
            hand=visible_hand,
            archive_count=len(player.archive),
            archive=visible_archive,
            deck_count=len(player.deck),
            discard=player.discard.cards(),
            purged=player.purged.cards(),
            creatures=list(player.play_area.creatures),
            artifacts=list(player.play_area.artifacts),
            selected_house=player.selected_house,
            decklist=list(player.all_cards),
        )
    effects = [
        ActiveEffectSummary(
            source_name=e.source_card.name if e.source_card is not None else "?",
            source_iid=e.source_card.instance_id if e.source_card is not None else -1,
            player_affected=e.player_affected,
            variable=e.variable,
            op=e.op,
            value=e.value,
            remaining_duration=e.remaining_duration,
        )
        for e in game.active_effects.duration_effects
    ]
    return PlayerView(
        viewer=viewer,
        active_player=game.active_player_id,
        turn_number=game.turn_number,
        players=states,
        log_tail=game.log.tail(20),
        active_effects=effects,
    )

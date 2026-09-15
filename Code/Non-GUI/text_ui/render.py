"""Text rendering helpers for the text UI."""

from __future__ import annotations


def card_str(card) -> str:
    extra = ""
    if hasattr(card.type_object, "damage"):
        extra = f" [{card.type_object.damage}/{card.type_object.base_power}]"
    return f"{card.name}{extra}"


def render_status(view) -> str:
    lines = []
    for pid in (1, 2):
        p = view.players[pid]
        marker = "*" if pid == view.active_player else " "
        lines.append(
            f"{marker}P{pid}: keys={p.keys} aember={p.aember} chains={p.chains} "
            f"hand={p.hand_count} deck={p.deck_count} archive={p.archive_count}"
        )
    return "\n".join(lines)


def render_board(view) -> str:
    lines = [render_status(view), ""]
    for pid in (1, 2):
        p = view.players[pid]
        lines.append(f"P{pid} creatures: {', '.join(card_str(c) for c in p.creatures) or '(none)'}")
        lines.append(f"P{pid} artifacts: {', '.join(card_str(c) for c in p.artifacts) or '(none)'}")
    return "\n".join(lines)


def render_hand(view) -> str:
    hand = view.me().hand or []
    return "\n".join(f"{i}: {card_str(c)} ({c.house.value})" for i, c in enumerate(hand))


def render_option(option) -> str:
    return repr(option)

"""Info commands that don't consume a decision."""

from __future__ import annotations

from .render import card_str, render_board, render_hand

INFO_COMMANDS = {"board", "hand", "archive", "discard", "purged", "decklist", "log", "help", "effects"}


def is_info_command(text: str) -> bool:
    return text.split()[0] in INFO_COMMANDS if text.strip() else False


def run_info_command(text: str, view, match) -> str:
    # `match` may be a Match (between games, its `current_game` is the live
    # one) or a plain Game (older callers) -- either way this resolves to
    # the engine object that actually has `.log` / `.active_effects`.
    game = getattr(match, "current_game", match)
    parts = text.split()
    cmd = parts[0]
    if cmd in ("board", "hand", "archive", "discard", "purged", "decklist") and not hasattr(view, "players"):
        return "(no game in progress -- between games in the match)"
    if cmd == "board":
        return render_board(view)
    if cmd == "hand":
        return render_hand(view)
    if cmd == "archive":
        arch = view.me().archive or []
        return "\n".join(card_str(c) for c in arch) or "(empty)"
    if cmd == "discard":
        pid = 2 if len(parts) > 1 and parts[1] == "opp" else view.viewer
        return "\n".join(card_str(c) for c in view.players[pid].discard) or "(empty)"
    if cmd == "purged":
        pid = 2 if len(parts) > 1 and parts[1] == "opp" else view.viewer
        return "\n".join(card_str(c) for c in view.players[pid].purged) or "(empty)"
    if cmd == "decklist":
        pid = 2 if len(parts) > 1 and parts[1] == "opp" else view.viewer
        names = sorted(c.name for c in view.players[pid].decklist)
        return "\n".join(names)
    if cmd == "log":
        if game is None:
            return "(no game in progress -- between games in the match)"
        n = int(parts[1]) if len(parts) > 1 else 10
        return "\n".join(str(e) for e in game.log.tail(n))
    if cmd == "help":
        return "Commands: board, hand, archive, discard [me|opp], purged [me|opp], decklist [me|opp], log [n], help, quit"
    if cmd == "effects":
        if game is None:
            return "(no game in progress -- between games in the match)"
        return "\n".join(
            f"{e.source_card.name}: {e.variable} {e.op} {e.value} (remaining={e.remaining_duration})"
            for e in game.active_effects.duration_effects
        ) or "(none active)"
    return f"Unknown command: {cmd}"

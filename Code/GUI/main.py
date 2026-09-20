#!/usr/bin/env python3
"""Entry point: `python main.py`. Launches straight into the menu; optional
flags jump straight into a match (handy for quick manual testing)."""

from __future__ import annotations

import argparse
import os
import sys

_GUI_DIR = os.path.dirname(os.path.abspath(__file__))
_NON_GUI_DIR = os.path.join(os.path.dirname(_GUI_DIR), "Non-GUI")
for _p in (_GUI_DIR, _NON_GUI_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="KeyForge GUI (Archon / Reversal / Adaptive)")
    parser.add_argument("--p1", choices=["human", "bot"], default=None)
    parser.add_argument("--p2", choices=["human", "bot"], default=None)
    parser.add_argument("--p1-deck", default="fignor", help="a bundled preset name or a path to a saved deck JSON file")
    parser.add_argument("--p2-deck", default="igor", help="a bundled preset name or a path to a saved deck JSON file")
    parser.add_argument("--format", choices=["archon", "reversal", "adaptive"], default="archon")
    parser.add_argument("--first", choices=["p1", "p2"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    args = parser.parse_args(argv)

    from gui.app import App
    from gui.scenes.menu_scene import MenuScene

    app = App()

    if args.p1 or args.p2:
        from gui.engine_bridge import MatchSettings

        first_player = {"p1": 1, "p2": 2}.get(args.first)
        settings = MatchSettings(
            p1_deck=args.p1_deck,
            p2_deck=args.p2_deck,
            p1_seat=args.p1 or "human",
            p2_seat=args.p2 or "bot",
            format=args.format,
            first_player=first_player,
            seed=args.seed,
            max_turns=args.max_turns,
        )
        if args.format == "archon":
            from gui.scenes.game_scene import GameScene

            app.push(GameScene(settings))
        else:
            from gui.scenes.match_scene import MatchScene

            app.push(MatchScene(settings))
    else:
        app.push(MenuScene())

    app.run()


if __name__ == "__main__":
    main()

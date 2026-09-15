"""A Controller that reads choices from stdin."""

from __future__ import annotations

from keyforge.enums import DecisionKind

from bots.base import Controller

from .commands import is_info_command, run_info_command
from .render import render_option


class HumanController(Controller):
    def __init__(self, game):
        self.game = game

    def decide(self, view, decision):
        while True:
            print(f"\n-- Player {decision.player}: {decision.prompt} --")
            for i, opt in enumerate(decision.options):
                print(f"  [{i}] {render_option(opt)}")
            text = input("> ").strip()
            if not text:
                continue
            if is_info_command(text):
                print(run_info_command(text, view, self.game))
                continue
            if text == "quit":
                raise SystemExit(0)
            if decision.kind in (DecisionKind.CHOOSE_CARDS, DecisionKind.ORDER_EFFECTS):
                try:
                    indices = [int(x) for x in text.replace(",", " ").split()]
                    choice = [decision.options[i] for i in indices]
                except (ValueError, IndexError):
                    print("Invalid selection.")
                    continue
            else:
                try:
                    choice = decision.options[int(text)]
                except (ValueError, IndexError):
                    print("Invalid selection.")
                    continue
            if decision.validate(choice):
                return choice
            print("That choice isn't legal right now.")

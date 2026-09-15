"""Test setup: make both `gui` and the Non-GUI `keyforge`/`bots` packages
importable, and force a headless (dummy) SDL video/audio driver so tests
never need a real display."""

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

_GUI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NON_GUI_DIR = os.path.join(os.path.dirname(_GUI_DIR), "Non-GUI")
for p in (_GUI_DIR, _NON_GUI_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import pygame  # noqa: E402

pygame.init()
pygame.display.set_mode((640, 360))

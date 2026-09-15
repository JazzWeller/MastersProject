"""Theme tokens, sizes, paths, and animation timings for the GUI.

Nothing here talks to pygame directly (no Surface/Font objects) so it can be
imported by headless tests without a display.
"""

from __future__ import annotations

import os

# --------------------------------------------------------------- paths ----

GUI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.dirname(GUI_DIR)
PROJECT_DIR = os.path.dirname(CODE_DIR)
NON_GUI_DIR = os.path.join(CODE_DIR, "Non-GUI")
CARD_ART_DIR = os.path.join(PROJECT_DIR, "Phase 1", "Cards")
ASSETS_DIR = os.path.join(GUI_DIR, "assets")
FONTS_DIR = os.path.join(ASSETS_DIR, "fonts")
SOUNDS_DIR = os.path.join(ASSETS_DIR, "sounds")

CINZEL_PATH = os.path.join(FONTS_DIR, "Cinzel-Bold.ttf")
INTER_PATH = os.path.join(FONTS_DIR, "Inter-Regular.ttf")

# --------------------------------------------------------- logical canvas ----

CANVAS_W, CANVAS_H = 1600, 900
BOARD_W = 1300           # left region: board + hands + hud
SIDE_X = BOARD_W         # right region: zoom + log
SIDE_W = CANVAS_W - BOARD_W
FPS = 60

# ------------------------------------------------------------- card sizes ----

CARD_ART_W, CARD_ART_H = 300, 420           # native art resolution
BOARD_CARD_W, BOARD_CARD_H = 105, 147       # creatures/artifacts on the board
HAND_CARD_W, HAND_CARD_H = 120, 168         # cards in a hand fan
ZOOM_CARD_W, ZOOM_CARD_H = 300, 420         # hover zoom panel
PILE_CARD_W, PILE_CARD_H = 90, 126          # deck/discard/archive/purged, in the browser modal
PILE_ICON_W, PILE_ICON_H = 56, 78           # compact pile stack shown in the left column

# ------------------------------------------------------------- board bands ----
# The board is split into a left column (piles), a center play area (hands,
# rows, HUD, prompt, action bar), and a right sidebar (zoom + log).

LEFT_COL_X, LEFT_COL_W = 10, 140
PLAY_X, PLAY_W = 160, 1130   # PLAY_X + PLAY_W == BOARD_W - 10
PILE_SLOT_H, PILE_GAP = 70, 4

BAND_OPP_HAND = (0, 84)
BAND_OPP_HUD = (88, 138)
BAND_OPP_ARTIFACTS = (144, 216)
BAND_OPP_CREATURES = (221, 386)
BAND_PROMPT = (391, 429)
BAND_YOUR_CREATURES = (434, 599)
BAND_YOUR_ARTIFACTS = (604, 676)
BAND_YOUR_HUD = (681, 731)
BAND_YOUR_HAND = (736, 850)
BAND_ACTION_BAR = (855, 892)

BAND_PILES_OPP = (22, 300)
BAND_PILES_YOU = (600, 878)

SIDE_ZOOM = (10, 440)
SIDE_LOG = (450, 892)

# --------------------------------------------------------------- palette ----


def _hex(h: str):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


BG_DEEP = _hex("0E0B16")
FELT_TOP = _hex("1A1526")
FELT_BOTTOM = _hex("241C33")
PANEL = _hex("171225")
PANEL_LIGHT = _hex("221B33")

AEMBER = _hex("F2A93B")
AEMBER_GLOW = _hex("FFD27A")
KEY_GOLD = _hex("E8C15A")
KEY_EMPTY = _hex("4A4360")

TEXT = _hex("EDE6F5")
TEXT_DIM = _hex("9C92AE")
TEXT_FAINT = _hex("6B6480")

DANGER = _hex("E5484D")
HEAL = _hex("46C18F")
PURGE = _hex("8B5CF6")

HOUSE_COLORS = {
    "Dis": _hex("D0246E"),
    "Logos": _hex("2F9BD6"),
    "Shadows": _hex("4E9C6B"),
}

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

CARD_BACK_BASE = _hex("241C4A")
CARD_BACK_EDGE = _hex("120C28")
CARD_BACK_FILIGREE = KEY_GOLD

GLOW_LEGAL = AEMBER
GLOW_SELECTED = WHITE
GLOW_TARGET = PURGE

# ---------------------------------------------------------- animation timings (ms) ----

T_DEAL = 300
T_SHUFFLE = 900
T_RESHUFFLE = 500
T_BANNER = 900
T_ARCHIVE_TAKE = 400
T_PLAY_MOVE = 450
T_EXHAUST_SETTLE = 200
T_PLAY_ACTION_FLY = 350
T_PLAY_ACTION_HOLD = 700
T_DISCARD = 350
T_REAP = 500
T_FIGHT = 700
T_DAMAGE = 400
T_HEAL = 350
T_DESTROY = 650
T_PURGE = 600
T_MOVE_ZONE = 350
T_AEMBER_GEM = 500
T_KEY_FORGE = 1200
T_CHAIN = 400
T_CHIP = 400
T_SETTLE = 300
T_GAME_OVER = 1500
T_BOT_THINK = 500

MAX_AEMBER_GEMS = 8

# ------------------------------------------------------------- misc sizes ----

BUTTON_H = 44
HAND_FAN_MAX_SPREAD = 900
HAND_FAN_LIFT = 26
LOG_LINES_VISIBLE = 14

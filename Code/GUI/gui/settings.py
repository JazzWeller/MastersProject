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
BOARD_CARD_W, BOARD_CARD_H = 120, 168       # creatures/artifacts on the board
HAND_CARD_W, HAND_CARD_H = 130, 182         # the viewer's own hand fan
OPP_HAND_CARD_W, OPP_HAND_CARD_H = 64, 90   # the opponent's hand: a compact flat row
UPGRADE_TAB_W, UPGRADE_TAB_H = 44, 62       # an upgrade, tucked inside its host's corner
ZOOM_CARD_W, ZOOM_CARD_H = 300, 420         # hover zoom panel
PILE_CARD_W, PILE_CARD_H = 90, 126          # deck/discard/archive/purged, in the browser modal
PILE_ICON_W, PILE_ICON_H = 50, 70           # compact pile stack shown in the left column
BROWSER_CARD_W, BROWSER_CARD_H = 140, 196   # pile browser / decklist grid cells
INSPECT_MAX_W, INSPECT_MAX_H = 520, 728     # full-size card inspector (B1)
MULLIGAN_CARD_W, MULLIGAN_CARD_H = 190, 266 # mulligan hand review screen

# ------------------------------------------------------------- board bands ----
# The board is split into a left column (piles), a center play area (hands,
# rows, HUD, prompt, action bar), and a right sidebar (zoom + log).

LEFT_COL_X, LEFT_COL_W = 10, 140
PLAY_X, PLAY_W = 160, 1130   # PLAY_X + PLAY_W == BOARD_W - 10
PILE_SLOT_H, PILE_GAP = 70, 16  # gap holds each pile's label

# Every band fully contains its own cards -- no row may overlap another (see
# Code/PLAYTEST_FIX_PLAN.md 0.6, and test_snapshot_layout.py, which checks
# each card's rotated bounding box against its band for 1-12 cards).
#
# Creatures and artifacts share one "board" band per player, split into two
# lanes side by side (creatures left, artifacts right, divider between), so
# there are 8 bands instead of 10 and every card can be larger.
BAND_OPP_HAND = (6, 100)          # 64x90 card backs, flat
BAND_OPP_HUD = (108, 148)
BAND_OPP_BOARD = (156, 340)       # 120x168 cards + 8px padding
BAND_PROMPT = (348, 384)
BAND_YOUR_BOARD = (392, 576)
BAND_YOUR_HUD = (584, 624)
BAND_YOUR_HAND = (632, 852)       # 130x182 fan, see HAND_FAN_* below
BAND_ACTION_BAR = (858, 894)

CREATURE_LANE_W = 718
LANE_DIVIDER_W = 12
ARTIFACT_LANE_W = PLAY_W - CREATURE_LANE_W - LANE_DIVIDER_W   # 400: three artifacts side by side
LANE_PAD = 8

BAND_PILES_OPP = (4, 348)
BAND_PILES_YOU = (548, 892)

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
NOTE = _hex("F0A35E")  # an effect that could not be carried out in full

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

# Distinct tint per pile kind (D4) so "which stack is the archive" isn't a
# memory test -- four identical grey rectangles, before this.
PILE_TINT_DECK = CARD_BACK_EDGE
PILE_TINT_DISCARD = _hex("3A2020")
PILE_TINT_ARCHIVE = _hex("3A3018")
PILE_TINT_PURGED = _hex("281A3A")

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
T_SHORTFALL = 900        # pause so an "it did nothing, because..." popup can be read
T_SHORTFALL_TEXT = 2200  # how long that popup stays up
T_PURGE = 600
T_MOVE_ZONE = 350
T_AEMBER_GEM = 500
T_KEY_FORGE = 2200
T_CHAIN = 400
T_CHIP = 400
T_SETTLE = 300
T_GAME_OVER = 1500
T_BOT_THINK = 500

MAX_AEMBER_GEMS = 8

# ------------------------------------------------------------- misc sizes ----

BUTTON_H = 44
HAND_FAN_MAX_SPREAD = 900
HAND_FAN_LIFT = 8           # px the outermost card is lifted, arcing toward the board
HAND_FAN_MAX_ROT = 5        # degrees, clamp on the outermost card's tilt
HAND_FAN_ROT_SLOPE = 2.0    # degrees of tilt per card-index away from center, before the clamp
MIN_FONT = 12               # no label anywhere is drawn smaller than this
LOG_LINES_VISIBLE = 14

"""The per-player HUD strip: house badge, name, aember, keys, chains, and
status chips for active lasting effects.
"""

from __future__ import annotations

from typing import List

import pygame

from .. import settings as S
from .widgets import Chip, draw_panel


class PlayerHUDState:
    """The small bit of per-player HUD state that animates independently of
    the true value (an æmber counter that visibly counts up/down)."""

    def __init__(self, initial_aember: int = 0):
        self.displayed_aember = float(initial_aember)


def _effect_chip_text(effect: dict) -> str:
    var = effect["variable"]
    op = effect["op"]
    val = effect["value"]
    dur = effect["remaining_duration"]
    dur_txt = "" if dur == -1 else f" ({dur})"
    labels = {
        "CanKeyForge": "Can't forge a key" if val is False else "Can forge a key",
        "CanPlayActions": "Can't play Actions" if val is False else "Can play Actions",
        "CanPlayCreatures": "Can't play Creatures" if val is False else "Can play Creatures",
        "HouseSelection": f"House forced: {val}",
        "KeyForgeCost": f"Key cost {op}{val}",
        "DrawUpToLimit": f"Draw limit {op}{val}",
        "CardDrawModifier": f"Draw {op}{val}",
        "CardPlayedLimit": f"Play limit = {val}",
        "NonLogosCardsPlayable": f"Non-Logos plays +{val}",
    }
    base = labels.get(var, f"{var} {op}{val}")
    return f"{base}{dur_txt}"


def draw_hud(
    surface: pygame.Surface,
    assets,
    rect: pygame.Rect,
    player_snapshot,
    hud_state: PlayerHUDState,
    is_active_player: bool,
    label: str,
    effects: List[dict],
) -> None:
    draw_panel(surface, rect, alpha=190, border=(S.AEMBER if is_active_player else None))

    x = rect.left + 12
    cy = rect.centery

    house = player_snapshot.selected_house
    color = S.HOUSE_COLORS.get(house, S.TEXT_FAINT)
    r = 14
    pygame.draw.circle(surface, (*color, 90), (x + r, cy), r)
    pygame.draw.circle(surface, color, (x + r, cy), r, width=2)
    x += 2 * r + 10

    name_font = assets.font("inter", 15, bold=True)
    name_txt = f"{label}" + (f" · {house}" if house else "")
    img = name_font.render(name_txt, True, S.TEXT)
    surface.blit(img, (x, cy - img.get_height() // 2))
    x += img.get_width() + 18

    gem = assets.aember_gem(16)
    surface.blit(gem, (x, cy - 8))
    x += 20
    aember_font = assets.font("inter", 16, bold=True)
    aember_txt = f"{round(hud_state.displayed_aember)}"
    img = aember_font.render(aember_txt, True, S.AEMBER)
    surface.blit(img, (x, cy - img.get_height() // 2))
    x += img.get_width() + 18

    for i in range(3):
        key_img = assets.key_icon(18, forged=(i < player_snapshot.keys))
        surface.blit(key_img, (x, cy - 9))
        x += 20
    x += 6

    if player_snapshot.chains > 0:
        chain_img = assets.chain_icon(18)
        surface.blit(chain_img, (x, cy - 9))
        x += 22
        chain_font = assets.font("inter", 14, bold=True)
        img = chain_font.render(str(player_snapshot.chains), True, S.TEXT_DIM)
        surface.blit(img, (x, cy - img.get_height() // 2))
        x += img.get_width() + 12

    # status chips, right-aligned within remaining space
    if effects:
        chip_x = x
        max_x = rect.right - 10
        for eff in effects:
            chip = Chip(_effect_chip_text(eff), color=S.HOUSE_COLORS.get(eff.get("_house"), S.PURGE))
            size_rect = chip.size(assets, 12)
            if chip_x + size_rect.width > max_x:
                break
            chip.draw(surface, assets, chip_x, cy - size_rect.height // 2, 12)
            chip_x += size_rect.width + 6

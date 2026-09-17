"""The per-player HUD strip: house badge, name, aember, keys, key cost,
chains, and status chips for active lasting effects.

`draw_hud` returns where it drew each element, so animations can target
the real key slot and the scene can show tooltips over keys and chips.
"""

from __future__ import annotations

from typing import Dict, List

import pygame

from .. import settings as S
from .widgets import Chip, draw_panel


class PlayerHUDState:
    """The small bit of per-player HUD state that animates independently of
    the true value (an æmber counter that visibly counts up/down)."""

    def __init__(self, initial_aember: int = 0):
        self.displayed_aember = float(initial_aember)


def effect_chip_text(effect: dict) -> str:
    var = effect["variable"]
    op = effect["op"]
    val = effect["value"]
    dur = effect["remaining_duration"]
    dur_txt = "" if dur == -1 else f" ({dur} turn{'s' if dur != 1 else ''} left)"
    labels = {
        "CanKeyForge": "Can't forge a key" if val is False else "Can forge a key",
        "CanPlayActions": "Can't play actions" if val is False else "Can play actions",
        "CanPlayCreatures": "Can't play creatures" if val is False else "Can play creatures",
        "HouseSelection": f"Must choose {getattr(val, 'value', val)}",
        "KeyForgeCost": f"Key cost {op}{val}",
        "DrawUpToLimit": f"Hand refill {op}{val}",
        "CardDrawModifier": f"Draw {op}{val}",
        "CardPlayedLimit": f"Max {val} plays per turn",
        "NonLogosCardsPlayable": f"May play {val} non-Logos",
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
    tag: str = "",
    forge_turns: List[int] = (),
) -> Dict[str, object]:
    rects: Dict[str, object] = {"keys": [], "chips": []}
    draw_panel(surface, rect, alpha=190, border=(S.AEMBER if is_active_player else None))

    x = rect.left + 12
    cy = rect.centery

    house = player_snapshot.selected_house
    color = S.HOUSE_COLORS.get(house, S.TEXT_FAINT)
    r = 13
    pygame.draw.circle(surface, color, (x + r, cy), r, width=0 if house else 2)
    x += 2 * r + 10

    name_font = assets.font("inter", 15, bold=True)
    name_txt = f"{label}" + (f" · {house}" if house else "")
    img = name_font.render(name_txt, True, S.TEXT)
    surface.blit(img, (x, cy - img.get_height() // 2))
    x += img.get_width() + 10

    if tag:
        tag_font = assets.font("inter", 12, bold=True)
        t = tag_font.render(tag, True, S.BLACK)
        tr = t.get_rect().inflate(12, 4)
        tr.midleft = (x, cy)
        pygame.draw.rect(surface, S.AEMBER_GLOW, tr, border_radius=tr.height // 2)
        surface.blit(t, t.get_rect(center=tr.center))
        rects["tag"] = tr
        x = tr.right + 12
    else:
        x += 8

    gem = assets.aember_gem(18)
    surface.blit(gem, (x, cy - 9))
    x += 22
    aember_font = assets.font("inter", 18, bold=True)
    img = aember_font.render(f"{round(hud_state.displayed_aember)}", True, S.AEMBER)
    aember_rect = img.get_rect(midleft=(x, cy))
    surface.blit(img, aember_rect)
    rects["aember"] = aember_rect.union(pygame.Rect(x - 22, cy - 9, 18, 18))
    x = aember_rect.right + 16

    for i in range(3):
        key_img = assets.key_icon(24, forged=(i < player_snapshot.keys))
        kr = pygame.Rect(x, cy - 12, 24, 24)
        surface.blit(key_img, kr)
        tip = f"Key {i + 1}: forged on turn {forge_turns[i]}" if i < len(forge_turns) else f"Key {i + 1}: not forged"
        rects["keys"].append((kr, tip))
        x += 27
    x += 8

    cost_font = assets.font("inter", 13)
    if player_snapshot.can_forge:
        cost_color = S.TEXT_DIM if player_snapshot.key_cost == 6 else S.AEMBER
        cost_txt = f"Key cost {player_snapshot.key_cost}"
    else:
        cost_color, cost_txt = S.DANGER, "Can't forge a key"
    img = cost_font.render(cost_txt, True, cost_color)
    cost_rect = img.get_rect(midleft=(x, cy))
    surface.blit(img, cost_rect)
    rects["cost"] = cost_rect
    x = cost_rect.right + 14

    if player_snapshot.chains > 0:
        chain_img = assets.chain_icon(18)
        surface.blit(chain_img, (x, cy - 9))
        x += 22
        chain_font = assets.font("inter", 14, bold=True)
        img = chain_font.render(str(player_snapshot.chains), True, S.TEXT_DIM)
        surface.blit(img, (x, cy - img.get_height() // 2))
        rects["chains"] = pygame.Rect(x - 22, cy - 10, img.get_width() + 22, 20)
        x += img.get_width() + 12

    # Status chips; whatever doesn't fit collapses into a "+N" chip whose
    # tooltip lists the rest (they used to silently stop drawing).
    if effects:
        max_x = rect.right - 10
        font_size = 12
        plus_w = Chip("+99", color=S.TEXT_DIM).size(assets, font_size).width + 6
        chip_x = x
        for i, eff in enumerate(effects):
            text = effect_chip_text(eff)
            chip = Chip(text, color=S.HOUSE_COLORS.get(eff.get("_house"), S.PURGE))
            size = chip.size(assets, font_size)
            remaining = len(effects) - i - 1
            room = max_x - (plus_w if remaining else 0)
            if chip_x + size.width > room:
                rest = effects[i:]
                more = Chip(f"+{len(rest)}", color=S.TEXT_DIM)
                cr = more.draw(surface, assets, chip_x, cy - size.height // 2, font_size)
                rects["chips"].append((cr, [f"{effect_chip_text(e)} — {e['source_name']}" for e in rest]))
                break
            cr = chip.draw(surface, assets, chip_x, cy - size.height // 2, font_size)
            rects["chips"].append((cr, [f"{text} — from {eff['source_name']}"]))
            chip_x += size.width + 6
    return rects

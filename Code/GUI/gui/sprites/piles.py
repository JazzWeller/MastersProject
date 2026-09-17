"""Draws the backdrop for one deck/discard/archive/purged pile: an outline
when empty, or a plain card-back-shaped backing plate when not (the actual
top card, face up or down as visibility allows, is a real CardSprite drawn
in the normal sprite pass, positioned exactly here by `Board.slot`) — plus
a count badge and a small label. Clicking a pile (handled by the scene, via
`rect.collidepoint`) opens a CardGridBrowser modal.
"""

from __future__ import annotations

import pygame

from .. import settings as S

_KIND_TINT = {
    "deck": S.PILE_TINT_DECK,
    "discard": S.PILE_TINT_DISCARD,
    "archive": S.PILE_TINT_ARCHIVE,
    "purged": S.PILE_TINT_PURGED,
}
_KIND_LABEL_COLOR = {
    "deck": S.TEXT_FAINT,
    "discard": S.DANGER,
    "archive": S.AEMBER,
    "purged": S.PURGE,
}


def draw_pile(
    surface: pygame.Surface,
    assets,
    rect: pygame.Rect,
    kind: str,
    count: int,
    hovered: bool = False,
    browsable: bool = True,
) -> None:
    tint = _KIND_TINT.get(kind, S.CARD_BACK_EDGE)
    if count == 0:
        pygame.draw.rect(surface, S.TEXT_FAINT, rect, width=1, border_radius=6)
    else:
        pygame.draw.rect(surface, tint, rect, border_radius=6)
        pygame.draw.rect(surface, _KIND_LABEL_COLOR.get(kind, S.TEXT_FAINT), rect, width=1, border_radius=6)
        if hovered and browsable:
            pygame.draw.rect(surface, S.GLOW_LEGAL, rect, width=2, border_radius=6)

    label_font = assets.font("inter", 10)
    lab = label_font.render(kind.capitalize(), True, _KIND_LABEL_COLOR.get(kind, S.TEXT_FAINT))
    surface.blit(lab, (rect.centerx - lab.get_width() // 2, rect.top - 13))

    if count > 0 and not browsable:
        # A small padlock: this pile exists and has cards, but you can't
        # look through it (the opponent's archive, or any deck).
        lock_r = pygame.Rect(0, 0, 14, 10)
        lock_r.center = (rect.centerx, rect.centery + 2)
        pygame.draw.rect(surface, S.TEXT_DIM, lock_r, border_radius=2)
        pygame.draw.arc(
            surface, S.TEXT_DIM,
            pygame.Rect(lock_r.centerx - 5, lock_r.top - 9, 10, 12), 0, 3.2, width=2,
        )

    if count > 0:
        count_font = assets.font("inter", 12, bold=True)
        txt = count_font.render(str(count), True, S.WHITE)
        bubble = pygame.Rect(0, 0, txt.get_width() + 10, txt.get_height() + 6)
        bubble.bottomright = (rect.right - 2, rect.bottom - 2)
        pygame.draw.rect(surface, S.PANEL, bubble, border_radius=bubble.height // 2)
        pygame.draw.rect(surface, S.TEXT_DIM, bubble, width=1, border_radius=bubble.height // 2)
        surface.blit(txt, txt.get_rect(center=bubble.center))

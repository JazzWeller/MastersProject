"""App: window, 60 FPS clock, a fixed 1600x900 logical canvas letterboxed
into whatever window size the user has, a scene stack, and the handful of
global hotkeys (F11 fullscreen, F3 debug info)."""

from __future__ import annotations

from typing import List, Optional

import pygame

from . import settings as S
from .assets import AssetCache

_MOUSE_EVENTS = (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP)


class Scene:
    """Base class: override what you need."""

    app: "App" = None  # set by App.push

    def on_enter(self) -> None:
        pass

    def on_exit(self) -> None:
        pass

    def handle_event(self, event: pygame.event.Event) -> None:
        pass

    def update(self, dt_ms: float) -> None:
        pass

    def draw(self, surface: pygame.Surface) -> None:
        pass

    def draw_crisp(self, window_surface: pygame.Surface) -> None:
        """Optional second drawing pass, called after the logical canvas has
        already been scaled and blitted to the real window. Most scenes
        don't need this -- it exists so a scene showing one large card (the
        full-size inspector) can re-render *that* card straight from its
        native art at the window's actual pixel size, instead of inheriting
        whatever blur the canvas's own scale-to-window step added on top of
        the canvas's own scale-from-art step. See `App._canvas_to_window_rect`
        and UX_FIX_PLAN.md B2."""
        pass

    @property
    def mouse(self) -> tuple:
        """Current mouse position in *canvas* coordinates (1600x900 logical
        space), not raw window pixels. Every hover/click test that polls the
        mouse instead of reading it off a translated event must go through
        this -- see `App._to_canvas` / `App._translate`."""
        return self.app.mouse_canvas


def _default_window_size() -> tuple:
    """The largest 16:9 box (the canvas's own aspect) that comfortably fits
    the desktop, capped at the canvas's native 1600x900 -- so the common
    case is an exact 1:1 window with no resampling blur at all, rather than
    always opening at a fixed 1280x720 that's neither the canvas size nor
    the desktop size. See UX_FIX_PLAN.md C1."""
    try:
        sizes = pygame.display.get_desktop_sizes()
        dw, dh = sizes[0] if sizes else (1280, 720)
    except Exception:
        dw, dh = 1280, 720
    avail_w, avail_h = max(1, dw - 120), max(1, dh - 160)
    scale = min(1.0, avail_w / S.CANVAS_W, avail_h / S.CANVAS_H)
    return (max(960, round(S.CANVAS_W * scale)), max(540, round(S.CANVAS_H * scale)))


class App:
    def __init__(self, window_size=None, title="KeyForge · Phase 1.1 Archon"):
        pygame.init()
        try:
            pygame.mixer.init()
        except pygame.error:
            pass
        if window_size is None:
            window_size = _default_window_size()
        self.window = pygame.display.set_mode(window_size, pygame.RESIZABLE)
        pygame.display.set_caption(title)
        self.canvas = pygame.Surface((S.CANVAS_W, S.CANVAS_H)).convert()
        self.clock = pygame.time.Clock()
        self.assets = AssetCache()
        self.scenes: List[Scene] = []
        self.running = True
        self._fullscreen = False
        self._pre_fullscreen_size = window_size
        self._dest_rect = pygame.Rect(0, 0, *window_size)
        self.show_fps = False
        self.mouse_canvas = (0.0, 0.0)
        self._history = None

    @property
    def history(self):
        """The game-history database, opened on first use. None if it can't
        be opened (read-only disk, etc.) -- games still play, unrecorded."""
        if self._history is None:
            try:
                from .history import GameHistory

                self._history = GameHistory()
            except Exception as exc:  # pragma: no cover - environment dependent
                print(f"Game history disabled: {exc}")
                self._history = False
        return self._history or None

    # ------------------------------------------------------------ scenes ----

    def push(self, scene: Scene) -> None:
        scene.app = self
        self.scenes.append(scene)
        scene.on_enter()

    def pop(self) -> None:
        if self.scenes:
            scene = self.scenes.pop()
            scene.on_exit()

    def replace(self, scene: Scene) -> None:
        self.pop()
        self.push(scene)

    def quit(self) -> None:
        self.running = False

    # -------------------------------------------------------- coordinates ----

    def _recompute_dest_rect(self) -> None:
        ww, wh = self.window.get_size()
        scale = min(ww / S.CANVAS_W, wh / S.CANVAS_H)
        w, h = int(S.CANVAS_W * scale), int(S.CANVAS_H * scale)
        x, y = (ww - w) // 2, (wh - h) // 2
        self._dest_rect = pygame.Rect(x, y, w, h)

    def _to_canvas(self, pos) -> tuple:
        r = self._dest_rect
        if r.width == 0 or r.height == 0:
            return (0, 0)
        sx = (pos[0] - r.x) / r.width * S.CANVAS_W
        sy = (pos[1] - r.y) / r.height * S.CANVAS_H
        return (sx, sy)

    def canvas_to_window_rect(self, rect: pygame.Rect) -> pygame.Rect:
        """The window-pixel rect a canvas-space rect currently maps to, for
        `Scene.draw_crisp`."""
        r = self._dest_rect
        if S.CANVAS_W == 0 or r.width == 0:
            return pygame.Rect(0, 0, 0, 0)
        scale = r.width / S.CANVAS_W
        return pygame.Rect(
            int(r.x + rect.x * scale), int(r.y + rect.y * scale),
            max(1, round(rect.width * scale)), max(1, round(rect.height * scale)),
        )

    def _translate(self, event: pygame.event.Event) -> pygame.event.Event:
        if event.type in _MOUSE_EVENTS:
            data = dict(event.__dict__)
            data["pos"] = self._to_canvas(event.pos)
            self.mouse_canvas = data["pos"]
            return pygame.event.Event(event.type, data)
        return event

    def toggle_fullscreen(self) -> None:
        self._fullscreen = not self._fullscreen
        if self._fullscreen:
            self._pre_fullscreen_size = self.window.get_size()
            self.window = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.window = pygame.display.set_mode(self._pre_fullscreen_size, pygame.RESIZABLE)

    def draw_scenes(self) -> None:
        """Draw the top scene, and the scene beneath it first if the top one
        is an overlay (game over draws over the final board)."""
        self.canvas.fill(S.BG_DEEP)
        if not self.scenes:
            return
        top = self.scenes[-1]
        if getattr(top, "overlay", False) and len(self.scenes) > 1:
            self.scenes[-2].draw(self.canvas)
        top.draw(self.canvas)

    # ------------------------------------------------------------- present ----

    def _present(self) -> None:
        self.window.fill((0, 0, 0))
        self._recompute_dest_rect()
        scaled = pygame.transform.smoothscale(self.canvas, self._dest_rect.size)
        self.window.blit(scaled, self._dest_rect)
        if self.scenes:
            self.scenes[-1].draw_crisp(self.window)
        if self.show_fps:
            font = self.assets.font("inter", 14, bold=True)
            img = font.render(f"{self.clock.get_fps():.0f} FPS", True, (0, 255, 120))
            self.window.blit(img, (8, 8))
        pygame.display.flip()

    # ----------------------------------------------------------------- run ----

    def run(self) -> None:
        self._recompute_dest_rect()
        while self.running and self.scenes:
            dt_ms = self.clock.tick(S.FPS)
            self.mouse_canvas = self._to_canvas(pygame.mouse.get_pos())
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    continue
                if event.type == pygame.VIDEORESIZE and not self._fullscreen:
                    self.window = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                    continue
                if event.type == pygame.KEYDOWN and event.key == pygame.K_F11:
                    self.toggle_fullscreen()
                    continue
                if event.type == pygame.KEYDOWN and event.key == pygame.K_F3:
                    self.show_fps = not self.show_fps
                    continue
                if not self.scenes:
                    continue
                self.scenes[-1].handle_event(self._translate(event))

            if not self.scenes:
                break
            scene = self.scenes[-1]
            scene.update(dt_ms)
            self.draw_scenes()
            self._present()

        pygame.quit()

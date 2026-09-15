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


class App:
    def __init__(self, window_size=(1280, 720), title="KeyForge · Phase 1.1 Archon"):
        pygame.init()
        try:
            pygame.mixer.init()
        except pygame.error:
            pass
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

    def _translate(self, event: pygame.event.Event) -> pygame.event.Event:
        if event.type in _MOUSE_EVENTS:
            data = dict(event.__dict__)
            data["pos"] = self._to_canvas(event.pos)
            return pygame.event.Event(event.type, data)
        return event

    def toggle_fullscreen(self) -> None:
        self._fullscreen = not self._fullscreen
        if self._fullscreen:
            self._pre_fullscreen_size = self.window.get_size()
            self.window = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.window = pygame.display.set_mode(self._pre_fullscreen_size, pygame.RESIZABLE)

    # ------------------------------------------------------------- present ----

    def _present(self) -> None:
        self.window.fill((0, 0, 0))
        self._recompute_dest_rect()
        scaled = pygame.transform.smoothscale(self.canvas, self._dest_rect.size)
        self.window.blit(scaled, self._dest_rect)
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
            self.canvas.fill(S.BG_DEEP)
            scene.draw(self.canvas)
            self._present()

        pygame.quit()

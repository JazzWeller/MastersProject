"""Runs a FIFO queue of animation "beats" (each a Tween/Sequence/Parallel
tree), one at a time. Input to the board is blocked while `is_busy`.
"""

from __future__ import annotations

from typing import List, Optional

from .tween import Playable


class Animator:
    def __init__(self):
        self.queue: List[Playable] = []
        self.current: Optional[Playable] = None
        self.speed = 1.0
        self.enabled = True  # False = "animations off" -> everything resolves instantly

    def enqueue(self, beat: Optional[Playable]) -> None:
        if beat is None:
            return
        if not self.enabled:
            beat.finish()
            return
        self.queue.append(beat)

    def update(self, dt_ms: float) -> None:
        if not self.enabled:
            # Drain anything already queued instantly (e.g. the toggle was
            # flipped mid-animation).
            while self.current is not None or self.queue:
                self._pop_and_finish()
            return
        dt_ms *= self.speed
        if self.current is None:
            if not self.queue:
                return
            self.current = self.queue.pop(0)
        if self.current.tick(dt_ms):
            self.current = None

    def _pop_and_finish(self) -> None:
        if self.current is not None:
            self.current.finish()
            self.current = None
        elif self.queue:
            self.current = self.queue.pop(0)

    def skip(self) -> None:
        """Jump the in-flight beat and everything queued to its end state."""
        if self.current is not None:
            self.current.finish()
            self.current = None
        for beat in self.queue:
            beat.finish()
        self.queue.clear()

    def clear(self) -> None:
        """Drop everything without resolving it (used when abandoning a scene)."""
        self.current = None
        self.queue.clear()

    @property
    def is_busy(self) -> bool:
        return self.current is not None or bool(self.queue)

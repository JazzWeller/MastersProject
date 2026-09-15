"""A tiny tween tree: Tween, Sequence, Parallel, Delay, Call.

Every node is a `Playable`: `start()` is called once, then `update(dt_ms)`
each frame until it returns True (done). `finish()` jumps straight to the
end state, used to implement "skip animation" (Space).

Durations are in milliseconds to match `gui/settings.py`.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from .easing import ease_out_cubic


class Playable:
    def __init__(self):
        self._started = False
        self._done = False

    def start(self) -> None:
        self._started = True

    def update(self, dt_ms: float) -> bool:
        raise NotImplementedError

    def finish(self) -> None:
        """Jump to the end state instantly."""
        if not self._started:
            self.start()
        self.update(10 ** 9)

    def tick(self, dt_ms: float) -> bool:
        """Drive this node: start it lazily, then update. Returns done."""
        if self._done:
            return True
        if not self._started:
            self.start()
        self._done = self.update(dt_ms)
        return self._done


class Tween(Playable):
    """Animate a single numeric attribute of `obj` to `to` over `duration_ms`."""

    def __init__(
        self,
        obj: Any,
        attr: str,
        to,
        duration_ms: float,
        ease: Callable[[float], float] = ease_out_cubic,
        on_complete: Optional[Callable[[], None]] = None,
    ):
        """`to` may be a number, or a zero-arg callable evaluated at `start()`
        (so e.g. a shake or lunge can target "current position + delta"
        without knowing that position when the beat tree is built)."""
        super().__init__()
        self.obj = obj
        self.attr = attr
        self.to = to
        self.duration_ms = max(1.0, duration_ms)
        self.ease = ease
        self.on_complete = on_complete
        self.frm = 0.0
        self._resolved_to = 0.0
        self.elapsed = 0.0

    def start(self) -> None:
        super().start()
        self.frm = getattr(self.obj, self.attr)
        self._resolved_to = self.to() if callable(self.to) else self.to
        self.elapsed = 0.0

    def update(self, dt_ms: float) -> bool:
        self.elapsed += dt_ms
        t = min(1.0, self.elapsed / self.duration_ms)
        value = self.frm + (self._resolved_to - self.frm) * self.ease(t)
        setattr(self.obj, self.attr, value)
        if t >= 1.0:
            setattr(self.obj, self.attr, self._resolved_to)
            if self.on_complete:
                self.on_complete()
            return True
        return False


class Delay(Playable):
    def __init__(self, duration_ms: float):
        super().__init__()
        self.duration_ms = max(0.0, duration_ms)
        self.elapsed = 0.0

    def start(self) -> None:
        super().start()
        self.elapsed = 0.0

    def update(self, dt_ms: float) -> bool:
        self.elapsed += dt_ms
        return self.elapsed >= self.duration_ms


class Call(Playable):
    """Fire-and-forget: calls `fn()` once, then is immediately done."""

    def __init__(self, fn: Callable[[], None]):
        super().__init__()
        self.fn = fn
        self._fired = False

    def update(self, dt_ms: float) -> bool:
        if not self._fired:
            self._fired = True
            self.fn()
        return True


class Sequence(Playable):
    def __init__(self, *children: Playable):
        super().__init__()
        self.children: List[Playable] = list(children)
        self.i = 0

    def start(self) -> None:
        super().start()
        self.i = 0

    def update(self, dt_ms: float) -> bool:
        while self.i < len(self.children):
            done = self.children[self.i].tick(dt_ms)
            if not done:
                return False
            self.i += 1
            dt_ms = 0  # only the first child in a frame gets real dt; the rest start "free"
        return True

    def finish(self) -> None:
        if not self._started:
            self.start()
        while self.i < len(self.children):
            self.children[self.i].finish()
            self.i += 1
        self._done = True


class Parallel(Playable):
    def __init__(self, *children: Playable):
        super().__init__()
        self.children: List[Playable] = list(children)

    def update(self, dt_ms: float) -> bool:
        all_done = True
        for c in self.children:
            if not c.tick(dt_ms):
                all_done = False
        return all_done

    def finish(self) -> None:
        if not self._started:
            self.start()
        for c in self.children:
            c.finish()
        self._done = True


def empty() -> Playable:
    return Delay(0)

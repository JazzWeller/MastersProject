"""The explicit resolution stack (Agent Observation Plan, Part R).

Card effects and the rules kernel are written as Python generator functions:
`choice = yield Decision(...)` pauses mid-effect, `yield from f(...)` calls
another pausing function. That source stays the single source of truth. The
engine can run it two ways (`Game(config, execution=...)`):

- `native`: the generators themselves, as the engine always has.
- `compiled`: `tools/compile_engine.py` turns every pausing function into a
  *routine* (`keyforge/compiled/*.py`, checked in) that runs the same code
  between suspension points natively, and keeps everything a suspended
  generator would hide -- where it is (`pc`) and its locals -- in a `Frame`
  on this module's `Machine`. What is left to resolve is then plain data:
  it can be copied at any decision, serialized, hashed and shown to a
  network.

A routine is called as `routine(frame, sent)` and returns one of:

- `(SUSPEND, decision)`: a player has to decide; the choice comes back as
  `sent` when the machine is resumed;
- `(CALL, f, args, kwargs)`: `yield from f(*args, **kwargs)` -- the machine
  pushes a frame for `f` (a compiled routine if `f` has one, otherwise a
  `NativeFrame` around the generator `f` returns) and delivers its return
  value as `sent` when it finishes;
- `(RETURN, value)`: the routine is done.

Exceptions propagate down the stack; a frame inside a `try/finally` region
runs its finally block first, as a generator would.

`NativeFrame` is the adapter that lets code that isn't compiled run inside
the machine unchanged (Part R, R2): a stack holding one can't be copied
(`Machine.copyable`), and `Game.copy()` falls back to replay for it.
"""

from __future__ import annotations

import os
import types
from typing import Any, Callable, Dict, List, Optional, Tuple

CALL = 1
SUSPEND = 2
RETURN = 3

EXECUTION_MODES = ("native", "compiled")


def default_execution() -> str:
    """The execution mode a `Game` gets when none is named:
    `$KEYFORGE_EXECUTION`, else `native`."""
    mode = os.environ.get("KEYFORGE_EXECUTION", "native")
    if mode not in EXECUTION_MODES:
        raise ValueError(f"KEYFORGE_EXECUTION={mode!r}: expected one of {EXECUTION_MODES}")
    return mode


class _Unbound:
    """A local that hasn't been assigned yet. Correct code never reads one;
    the sentinel only fills the slot so a frame's locals have a fixed
    layout."""

    __slots__ = ()

    def __repr__(self):
        return "<unbound>"

    def __reduce__(self):  # pragma: no cover -- a singleton
        return "UNBOUND"


UNBOUND = _Unbound()


class Routine:
    """A compiled pausing function: its step function and how to bind a
    call's arguments into a new frame's locals.

    - `step(frame, sent)` runs the routine from `frame.pc`.
    - `binder(*args, **kwargs)` returns the initial locals list, built with
      the original function's own signature (so Python binds the call
      exactly as it would have), with the original's defaults installed.
    - `freevars`: `(slot, name)` for each free variable of a nested
      function: its slot holds the closure's own cell.
    """

    __slots__ = ("rid", "qualname", "module", "step", "binder", "dynamic_defaults", "freevars", "local_names",
                 "code", "line_of_pc", "ops_of_pc")

    def __init__(self, rid: str, qualname: str, module: str, step: Callable, binder: Callable, *, dynamic_defaults: bool,
                 freevars: Tuple[Tuple[int, str], ...], local_names: Tuple[str, ...], code, line_of_pc: Dict[int, int],
                 ops_of_pc: Optional[Dict[int, tuple]] = None):
        self.rid = rid
        self.qualname = qualname
        self.module = module
        self.step = step
        self.binder = binder
        self.dynamic_defaults = dynamic_defaults
        self.freevars = freevars
        self.local_names = local_names
        self.code = code
        self.line_of_pc = line_of_pc
        self.ops_of_pc = ops_of_pc or {}

    def __repr__(self):
        return f"<Routine {self.rid}>"


# code object of the ORIGINAL generator function -> its Routine. Filled by
# `load_compiled()`; the machine looks a callee up here by `__code__`.
ROUTINES: Dict[types.CodeType, Routine] = {}
ROUTINES_BY_ID: Dict[str, Routine] = {}


class Frame:
    """One routine's suspended activation: `pc` is where it resumes, `L` its
    locals in the routine's fixed slot order, `handlers` the finally blocks
    it is inside (innermost last), `exc` an exception in flight through one
    of them."""

    __slots__ = ("routine", "pc", "L", "handlers", "exc")

    def __init__(self, routine: Routine, L: list):
        self.routine = routine
        self.pc = 0
        self.L = L
        self.handlers: Optional[List[int]] = None
        self.exc: Optional[BaseException] = None

    def __repr__(self):
        return f"<Frame {self.routine.rid} pc={self.pc}>"


class NativeFrame:
    """A live generator (or any iterator) inside the machine: code that
    isn't compiled, run exactly as `yield from` would run it."""

    __slots__ = ("gen", "started")

    def __init__(self, gen):
        self.gen = gen
        self.started = False

    def step(self, sent):
        gen = self.gen
        try:
            if not self.started:
                self.started = True
                value = next(gen)
            elif sent is None:
                value = next(gen)
            else:
                value = gen.send(sent)  # as `yield from` does: an iterator without send() raises
        except StopIteration as stop:
            return RETURN, stop.value
        return SUSPEND, value

    def throw(self, exc):
        try:
            value = self.gen.throw(exc) if hasattr(self.gen, "throw") else None
        except StopIteration as stop:
            return RETURN, stop.value
        if value is None and not hasattr(self.gen, "throw"):
            raise exc
        return SUSPEND, value

    def __repr__(self):
        return f"<NativeFrame {getattr(self.gen, '__qualname__', type(self.gen).__name__)}>"


def _routine_for(f) -> Tuple[Optional[Routine], Any, tuple]:
    """`(routine, function, leading args)` for a callee: a bound method's
    `self` becomes the first argument."""
    if type(f) is types.MethodType:
        func = f.__func__
        code = getattr(func, "__code__", None)
        if code is not None:
            r = ROUTINES.get(code)
            if r is not None:
                return r, func, (f.__self__,)
        return None, f, ()
    code = getattr(f, "__code__", None)
    if code is not None:
        r = ROUTINES.get(code)
        if r is not None:
            return r, f, ()
    return None, f, ()


def new_frame(f, args: tuple, kwargs: Optional[dict]):
    """The frame `yield from f(*args, **kwargs)` runs in."""
    routine, func, lead = _routine_for(f)
    if routine is None:
        return NativeFrame(iter(f(*args, **kwargs) if kwargs else f(*args)))
    binder = routine.binder
    if routine.dynamic_defaults:
        binder.__defaults__ = func.__defaults__
        binder.__kwdefaults__ = func.__kwdefaults__
    if lead:
        args = lead + args
    L = binder(*args, **kwargs) if kwargs else binder(*args)
    if routine.freevars:
        closure = func.__closure__
        names = func.__code__.co_freevars
        for slot, name in routine.freevars:
            L[slot] = closure[names.index(name)]
    return Frame(routine, L)


class Machine:
    """The resolution stack of one game (or match): frames, bottom to top.
    `run(sent)` resumes the top frame with `sent` and runs until a frame
    suspends on a decision, which it returns; when the bottom frame
    returns, `done` is set and `result` holds its value."""

    __slots__ = ("stack", "done", "result")

    def __init__(self, root=None):
        self.stack: List[Any] = [] if root is None else [root]
        self.done = False
        self.result = None

    @classmethod
    def calling(cls, f, *args, **kwargs) -> "Machine":
        return cls(new_frame(f, args, kwargs or None))

    @property
    def copyable(self) -> bool:
        return not any(type(fr) is NativeFrame for fr in self.stack)

    def run(self, sent=None):
        stack = self.stack
        while stack:
            fr = stack[-1]
            try:
                if type(fr) is Frame:
                    out = fr.routine.step(fr, sent)
                else:
                    out = fr.step(sent)
            except BaseException as exc:  # noqa: BLE001 -- re-raised once every finally has run
                out = self._unwind(exc)
            op = out[0]
            if op == SUSPEND:
                return out[1]
            if op == CALL:
                stack.append(new_frame(out[1], out[2], out[3]))
                sent = None
                continue
            stack.pop()
            sent = out[1]
        self.done = True
        self.result = sent
        return None

    def _unwind(self, exc: BaseException):
        """Propagates `exc` from the top frame down: the first frame inside
        a finally region resumes in its finally block (which re-raises when
        it ends); a native generator gets it thrown in. Re-raises if no
        frame handles it."""
        stack = self.stack
        stack.pop()
        while stack:
            fr = stack[-1]
            if type(fr) is Frame:
                if fr.handlers:
                    fr.pc = fr.handlers.pop()
                    fr.exc = exc
                    try:
                        return fr.routine.step(fr, None)
                    except BaseException as again:  # noqa: BLE001
                        exc = again
                        stack.pop()
                        continue
                stack.pop()
                continue
            try:
                return fr.throw(exc)
            except BaseException as again:  # noqa: BLE001
                exc = again
                stack.pop()
        raise exc


class MachineDriver:
    """The `send`/`__next__` face of a `Machine`, so `Game`'s driver code
    is the same in both execution modes: `next()` primes it, `send(choice)`
    resumes it, and `StopIteration` ends it."""

    __slots__ = ("machine",)

    def __init__(self, machine: Machine):
        self.machine = machine

    def __iter__(self):
        return self

    def __next__(self):
        return self.send(None)

    def send(self, value):
        m = self.machine
        if m.done:
            raise StopIteration(m.result)
        decision = m.run(value)
        if m.done:
            raise StopIteration(m.result)
        return decision


# ------------------------------------------------- compiled-code support ----
# Names the generated routines reference. They are injected into the
# globals of each module that has compiled routines (prefixed, so they can't
# collide with that module's own names).


def make_function(code, globals_, defaults, kwdefaults, cells_by_name):
    """A nested function exactly as the original `def`/`lambda` would make
    it: the ORIGINAL code object (so its identity, name and qualname are the
    native ones), its defaults as evaluated at the def, and a closure of the
    enclosing routine's own cells."""
    closure = tuple(cells_by_name[n] for n in code.co_freevars) if code.co_freevars else None
    fn = types.FunctionType(code, globals_, code.co_name, defaults, closure)
    if kwdefaults:
        fn.__kwdefaults__ = kwdefaults
    fn.__qualname__ = code.co_qualname if hasattr(code, "co_qualname") else code.co_name
    return fn


def as_sequence(x):
    """What a compiled `for` loop iterates by index: a list, tuple or range
    itself (a list is read live, exactly as Python's own list iterator reads
    it), anything else materialized once."""
    t = type(x)
    if t is list or t is tuple or t is range:
        return x
    return list(x)


SUPPORT = {
    "_kf_U": UNBOUND,
    "_kf_cell": types.CellType,
    "_kf_fn": make_function,
    "_kf_seq": as_sequence,
}


_LOADED: List[str] = []


def load_compiled() -> None:
    """Imports every generated module under `keyforge/compiled/` once,
    registering its routines. Idempotent."""
    if _LOADED:
        return
    _LOADED.append("loading")
    from . import compiled

    compiled.load_all()
    _LOADED[0] = "loaded"

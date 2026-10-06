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

A routine is ONE function with two ways in:

- called with the original function's own arguments (Python binds them,
  with the original defaults), it runs a fresh call from the top;
- called as `step(_kfF=frame, _sent=value)` it resumes a suspended frame:
  its locals and `pc` come from the frame, and `value` is what it was
  waiting for.

Either way it returns the function's return value when it is done, or a
`Suspended` when a player has to decide. `yield from f(*args)` runs
**inline**: the routine calls `step_of(f)(*args)` on the Python stack.
Nothing reaches the machine unless something suspends: then each routine on
the way out that has no frame yet makes one (saving its locals and `pc`) and
adds it to `Suspended.frames`, top first, and the machine pushes them. A
call that finishes without suspending -- most of them -- costs one extra
Python call (`step_of`). The machine's stack at a decision is exactly the
chain of routines waiting on it.

Exceptions propagate as Python exceptions. A routine inside a `try/finally`
region runs its finally block as Python itself does; a frame already on the
machine's stack, waiting inside a try region, has the exception raised where
it waits (`Machine._unwind`), and a native generator gets it thrown in.

`NativeFrame` is the adapter that lets code that isn't compiled run inside
the machine unchanged (Part R, R2): a stack holding one can't be copied
(`Machine.copyable`).
"""

from __future__ import annotations

import contextlib
import os
import types
from typing import Any, Callable, Dict, List, Optional, Tuple

class Suspended:
    """What a routine returns instead of a value when a player has to
    decide: the decision, and the frames that suspended on the way out to
    the machine, top first (None for a frame the machine resumed itself)."""

    __slots__ = ("decision", "frames")

    def __init__(self, decision, frames=None):
        self.decision = decision
        self.frames = [] if frames is None else frames

    def __repr__(self):
        return f"<Suspended on {self.decision!r}>"

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
    """A compiled pausing function (see the module docstring for how its
    function `step` is called).

    - `local_names`: its frame's slots, in order.
    - `freevars`: a nested function's free variables; a fresh call gets its
      closure's cells (`_kf_closure`), in this order.
    - `exc_slot`, `exc_points`: for a routine with `try/finally`, the slot
      an exception is delivered through and the resume ids inside a try
      region (else None and empty).
    - `dynamic`: a nested function, whose defaults and closure come from the
      function object being called.
    - `line_of_pc`: the source line each `pc` resumes at.
    - `site_of_pc`: what the frame waits on at each `pc` -- the callee's
      source text, or "decision" (R7).
    - `ops_of_pc`: the rules operations still reachable from each `pc`, in
      source order, loops marked (R7; see tools/compile_engine.py's
      `_ops_of`).
    """

    __slots__ = ("rid", "qualname", "module", "step", "dynamic", "freevars", "local_names", "code", "line_of_pc",
                 "exc_slot", "exc_points", "ops_of_pc", "site_of_pc", "n_positional")

    def __init__(self, rid: str, qualname: str, module: str, step: Callable, *, dynamic: bool,
                 freevars: Tuple[str, ...], local_names: Tuple[str, ...], code, line_of_pc: Dict[int, int],
                 exc_slot: Optional[int], exc_points: frozenset, n_positional: int, ops_of_pc: Optional[Dict[int, tuple]] = None,
                 site_of_pc: Optional[Dict[int, str]] = None):
        self.rid = rid
        self.qualname = qualname
        self.module = module
        self.step = step
        self.dynamic = dynamic
        self.freevars = freevars
        self.local_names = local_names
        self.code = code
        self.line_of_pc = line_of_pc
        self.exc_slot = exc_slot
        self.exc_points = exc_points
        self.n_positional = n_positional
        self.ops_of_pc = ops_of_pc or {}
        self.site_of_pc = site_of_pc or {}

    def __repr__(self):
        return f"<Routine {self.rid}>"


# id() of the code object of the ORIGINAL generator function -> its Routine.
# Filled by `load_compiled()`; the machine looks a callee up here by
# `id(callee.__code__)`. By id, not by the code object itself: CPython
# doesn't cache a code object's hash, and hashing one walks its bytecode and
# constants -- that alone made every call several times dearer. The Routine
# holds its code object, so the id stays valid.
ROUTINES: Dict[int, Routine] = {}
ROUTINES_BY_ID: Dict[str, Routine] = {}


class Frame:
    """One routine's suspended activation: `pc` is where it resumes, `L` its
    locals in the routine's slot order."""

    __slots__ = ("routine", "pc", "L")

    def __init__(self, routine: Routine, L: list):
        self.routine = routine
        self.pc = 0
        self.L = L

    def __repr__(self):
        return f"<Frame {self.routine.rid} pc={self.pc}>"


class NativeFrame:
    """A live generator (or any iterator) inside the machine: code that
    isn't compiled, run exactly as `yield from` would run it."""

    __slots__ = ("gen",)

    def __init__(self, gen):
        self.gen = gen

    def step(self, sent):
        """The generator's return value, or a `Suspended` (with no frames:
        this frame is on the stack already)."""
        gen = self.gen
        try:
            # as `yield from` does: an iterator without send() can only be
            # sent None
            value = next(gen) if sent is None else gen.send(sent)
        except StopIteration as stop:
            return stop.value
        return Suspended(value, None)

    def throw(self, exc):
        if not hasattr(self.gen, "throw"):
            raise exc
        try:
            value = self.gen.throw(exc)
        except StopIteration as stop:
            return stop.value
        return Suspended(value, None)

    def __repr__(self):
        return f"<NativeFrame {getattr(self.gen, '__qualname__', type(self.gen).__name__)}>"


class _NativeCall:
    """`step_of(f)` for an `f` that isn't compiled: calls it, runs the
    generator (or iterable) it returns until it suspends or ends."""

    __slots__ = ("f",)

    def __init__(self, f):
        self.f = f

    def __call__(self, *args, **kwargs):
        gen = iter(self.f(*args, **kwargs))
        try:
            value = next(gen)
        except StopIteration as stop:
            return stop.value
        return Suspended(value, [NativeFrame(gen)])


_MethodType = types.MethodType
_FunctionType = types.FunctionType


# The original function object of every compiled module-level function and
# method -> its routine's step: what `step_of` returns for it (bound to
# `self` for a method). Keyed by the function itself (hashed by identity, so
# a lookup is cheap); filled by the loader.
STEPS: Dict[Any, Callable] = {}

# The attribute a nested function (a closure, e.g. a card hook made by a
# factory) caches its own step under: its routine bound to its closure.
STEP_ATTR = "__kf_step__"

_ENABLED = True


# Every module's globals the loader injected the support names into: what
# `disabled()` switches the direct Game calls off in.
INJECTED: List[dict] = []


class _NotAGame:
    """Stands in for `Game` in the direct-call check while disabled."""


@contextlib.contextmanager
def disabled():
    """For tests of the adapter: inside, no routine is registered, so every
    call runs as native code in a `NativeFrame`."""
    global _ENABLED
    load_compiled()
    saved_r, saved_s = dict(ROUTINES), dict(STEPS)
    saved_g = [g.get("_kfGame") for g in INJECTED]
    ROUTINES.clear()
    STEPS.clear()
    for g in INJECTED:
        g["_kfGame"] = _NotAGame
    _ENABLED = False
    try:
        yield
    finally:
        _ENABLED = True
        ROUTINES.update(saved_r)
        STEPS.update(saved_s)
        for g, v in zip(INJECTED, saved_g):
            g["_kfGame"] = v


def step_of(f):
    """What `yield from f(...)` calls, with `f`'s own arguments: `f`'s
    routine (bound to its `self` for a method), or a runner for code that
    isn't compiled."""
    if type(f) is _MethodType:
        s = STEPS.get(f.__func__)
        if s is not None:
            return _MethodType(s, f.__self__)
        return _NativeCall(f)
    s = STEPS.get(f)
    if s is not None:
        return s
    d = getattr(f, "__dict__", None)
    if d is not None and _ENABLED:
        s = d.get(STEP_ATTR)
        if s is not None:
            return s
        r = ROUTINES.get(id(getattr(f, "__code__", None)))
        if r is not None:
            s = _dynamic_step(r, f)
            d[STEP_ATTR] = s
            return s
    return _NativeCall(f)


def _dynamic_step(r: Routine, func):
    """A nested function's routine, with that function's own closure (and
    defaults, if it has any)."""
    import functools

    step = r.step
    if func.__defaults__ or func.__kwdefaults__:
        defaults = (UNBOUND,) * (r.n_positional - len(func.__defaults__ or ())) + tuple(func.__defaults__ or ()) + (None, None)
        step = _FunctionType(step.__code__, step.__globals__, step.__name__, defaults, step.__closure__)
        kw = dict(r.step.__kwdefaults__)
        kw.update(func.__kwdefaults__ or {})
        step.__kwdefaults__ = kw
    return functools.partial(step, _kf_closure=func.__closure__)


class Machine:
    """The resolution stack of one game (or match): frames, bottom to top.
    `run(sent)` resumes the top frame with `sent` and runs until something
    suspends on a decision, which it returns; when the bottom frame returns,
    `done` is set and `result` holds its value."""

    __slots__ = ("stack", "done", "result", "_start")

    def __init__(self):
        self.stack: List[Any] = []
        self.done = False
        self.result = None
        self._start = None

    @classmethod
    def calling(cls, f, *args, **kwargs) -> "Machine":
        """A machine whose first `run()` calls `f(*args, **kwargs)`."""
        m = cls()
        m._start = (f, args, kwargs)
        return m

    @property
    def copyable(self) -> bool:
        return self._start is None and not any(type(fr) is not Frame for fr in self.stack)

    def run(self, sent=None):
        stack = self.stack
        if self._start is not None:
            f, args, kwargs = self._start
            self._start = None
            out = step_of(f)(*args, **kwargs)
            if type(out) is Suspended:
                stack.extend(reversed(out.frames))
                return out.decision
            self.done = True
            self.result = out
            return None
        while stack:
            fr = stack[-1]
            try:
                out = fr.routine.step(_kfF=fr, _sent=sent) if type(fr) is Frame else fr.step(sent)
            except BaseException as exc:  # noqa: BLE001 -- re-raised once every finally has run
                out = self._unwind(exc)
            if type(out) is Suspended:
                if out.frames:
                    stack.extend(reversed(out.frames))
                return out.decision
            stack.pop()
            sent = out
        self.done = True
        self.result = sent
        return None

    def _unwind(self, exc: BaseException):
        """Propagates `exc` from the top frame (which has already run its own
        finally blocks) down the stack: a frame waiting inside a try region
        has it raised where it waits, which runs its finally block and
        re-raises; a native generator gets it thrown in, and may handle it.
        Re-raises once no frame is left."""
        stack = self.stack
        stack.pop()
        while stack:
            fr = stack[-1]
            try:
                if type(fr) is Frame:
                    r = fr.routine
                    if fr.pc not in r.exc_points:
                        stack.pop()  # not inside a try region: nothing to run
                        continue
                    fr.L[r.exc_slot] = exc
                    return r.step(_kfF=fr, _sent=None)
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
    "_kf_step": step_of,
    "_kf_S": Suspended,
    "_kf_Frame": Frame,
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

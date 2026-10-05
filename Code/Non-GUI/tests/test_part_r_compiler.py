"""Agent Observation Plan, Part R: the compiler and the machine (R2, R3).

- Every construct in the supported subset, compiled and run on
  `keyforge.vm.Machine`, behaves exactly as the native generator: the same
  values suspended on, the same side effects, the same return value.
- Randomized programs within the subset (tests/_random_programs.py), run
  natively and compiled with the same choices, give identical traces.
- Anything outside the subset is a `CompileError` naming its line.
- The checked-in `keyforge/compiled/` is up to date with the source.
- The adapter (R2): with no routine registered, every call runs as a
  `NativeFrame` inside the machine, and a game is still identical to native.
"""

import random
import unittest

from keyforge import vm
from tests._random_programs import random_module
from tools import compile_engine
from tools.compile_engine import CompileError, load_source_module


def choose(value, step):
    """The scripted 'player': a pure function of what was asked and when."""
    if isinstance(value, tuple) and value and value[0] == "listed":
        return None  # `yield from` a plain iterator: only None can be sent
    return (hash(repr(value)) % 13 + step * 3) % 11 if not isinstance(value, str) else step


def run_native(fn, *args):
    trace = []
    gen = fn(*args)
    step = 0
    try:
        value = next(gen)
        while True:
            c = choose(value, step)
            trace.append((value, c))
            step += 1
            value = gen.send(c)
    except StopIteration as stop:
        return trace, stop.value


def run_compiled(fn, *args):
    trace = []
    m = vm.Machine.calling(fn, *args)
    step = 0
    value = m.run(None)
    while not m.done:
        c = choose(value, step)
        trace.append((value, c))
        step += 1
        value = m.run(c)
    return trace, m.result


CONSTRUCTS = '''
from keyforge.decision import Decision

LOG = []


def ask(x):
    got = yield ("ask", x)
    return got


def native_helper(x):
    yield ("native", x)
    return x * 10


def returns_list(x):
    return [("listed", x), ("listed", x + 1)]


def kwonly(a, b=2, *, c=3, d):
    got = yield ("kw", a, b, c, d)
    return a + b + c + d + got


def shared_default(x, acc=[]):
    acc.append(x)
    yield ("acc", tuple(acc))
    return len(acc)


def straight(a):
    x = yield ("first", a)
    y = yield from ask(x + 1)
    return x, y


def branches(a):
    if a % 2:
        x = yield ("odd", a)
    elif a % 3:
        x = yield from ask(a)
    else:
        x = 0
    if (yield from ask(x)):
        LOG.append("truthy")
    else:
        LOG.append("falsy")
    return x


def loops(n):
    total = 0
    i = 0
    while i < n:
        i += 1
        if i == 2:
            continue
        got = yield ("while", i)
        total += got
        if total > 20:
            break
    seq = [1, 2]
    for x in seq:
        if len(seq) < 4:
            seq.append(x + 10)
        got = yield ("for", x)
        total += got
    for k in range(3):
        if k == 1:
            continue
        total += yield from ask(k)
    return total, seq


def finally_normal(a):
    LOG.append("before")
    try:
        x = yield ("in try", a)
        if x % 2:
            y = yield from ask(x)
        else:
            y = 0
    finally:
        LOG.append("finally")
    return x, y


def boom(x):
    yield ("boom?", x)
    raise ValueError("boom")


def finally_exception(a):
    try:
        yield from boom(a)
    finally:
        LOG.append("cleaned up")


def cells(a):
    v = a
    def get():
        return v
    f = lambda k: v + k
    v = yield ("cell", v)
    LOG.append((get(), f(1)))
    v = v + 100
    return get(), f(2)


def factory(n):
    def effect(x):
        got = yield ("factory", n, x)
        return got + n
    return effect


def nested_generator_inside(a):
    def inner(b):
        got = yield ("inner", a, b)
        return got + a
    r1 = yield from inner(1)
    a = a + 5
    r2 = yield from inner(2)
    return r1, r2


def mixed(a):
    r = yield from native_helper(a)
    s = yield from returns_list(a)
    t = yield from kwonly(1, d=4)
    u = yield from shared_default(a)
    w = yield from factory(3)(a)
    return r, s, t, u, w


def recursive(n):
    if n <= 0:
        return 0
        yield
    got = yield ("depth", n)
    rest = yield from recursive(n - 1)
    return got + rest


def stub(a):
    LOG.append(("stub", a))
    return a
    yield


def calls_stub(a):
    x = yield from stub(a)
    y = yield from stub(x + 1)
    return x + y


def decision_yield(p):
    choice = yield Decision(p, None, "pick", [1, 2, 3], 1, 1)
    return choice
'''


class TestConstructs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod, cls.routines = load_source_module("part_r_constructs", CONSTRUCTS)
        # `native_helper` and `returns_list` must stay native: they test
        # calls into code that isn't compiled
        for name in ("native_helper",):
            vm.ROUTINES.pop(getattr(cls.mod, name).__code__, None)

    def both(self, name, *args):
        fn = getattr(self.mod, name)
        shared = self.mod.shared_default.__defaults__[0]  # one list, shared by every call -- in both modes
        self.mod.LOG.clear()
        shared.clear()
        native = run_native(fn, *args)
        log_native = list(self.mod.LOG)
        self.mod.LOG.clear()
        shared.clear()
        compiled = run_compiled(fn, *args)
        log_compiled = list(self.mod.LOG)
        self.assertEqual(native, compiled, name)
        self.assertEqual(log_native, log_compiled, name)
        return compiled

    def test_every_construct_matches_native(self):
        for name, args in [
            ("straight", (3,)), ("branches", (1,)), ("branches", (4,)), ("branches", (6,)),
            ("loops", (5,)), ("loops", (1,)), ("finally_normal", (2,)), ("finally_normal", (3,)),
            ("cells", (7,)), ("nested_generator_inside", (2,)), ("mixed", (4,)), ("recursive", (4,)),
            ("calls_stub", (5,)), ("decision_yield", (1,)),
        ]:
            with self.subTest(name=name, args=args):
                self.both(name, *args)

    def test_compiled_routines_were_used(self):
        names = {r.qualname for r in self.routines}
        for n in ("straight", "loops", "cells", "factory.<locals>.effect", "nested_generator_inside.<locals>.inner"):
            self.assertIn(n, names)
        m = vm.Machine.calling(self.mod.loops, 3)
        m.run(None)
        self.assertTrue(all(type(f) is vm.Frame for f in m.stack))

    def test_an_exception_runs_finally_blocks_then_propagates(self):
        self.mod.LOG.clear()
        with self.assertRaises(ValueError):
            run_native(self.mod.finally_exception, 1)
        native = list(self.mod.LOG)
        self.mod.LOG.clear()
        with self.assertRaises(ValueError):
            run_compiled(self.mod.finally_exception, 1)
        self.assertEqual(native, ["cleaned up"])
        self.assertEqual(self.mod.LOG, native)

    def test_nested_functions_are_the_native_ones(self):
        """A closure made inside a routine has the original code object and
        qualname, and shares its cells with the frame."""
        m = vm.Machine.calling(self.mod.cells, 1)
        m.run(None)
        frame = m.stack[-1]
        fns = [v for v in frame.L if callable(v) and getattr(v, "__code__", None) is not None]
        native_codes = {c for c in self.mod.cells.__code__.co_consts if hasattr(c, "co_name")}
        self.assertEqual(len(fns), 2)
        self.assertTrue(all(f.__code__ in native_codes for f in fns))
        self.assertEqual(sorted(f.__qualname__ for f in fns), ["cells.<locals>.<lambda>", "cells.<locals>.get"])


class TestRandomPrograms(unittest.TestCase):
    def test_random_programs_match_native(self):
        rng = random.Random(1)
        for seed in range(300):
            src = random_module(seed)
            with self.subTest(seed=seed):
                mod, _ = load_source_module(f"part_r_random_{seed}", src)
                a, b = rng.randint(0, 9), rng.randint(0, 9)
                mod.CLOSURES.clear()
                native = run_native(mod.main, [], a, b)
                mod.CLOSURES.clear()
                compiled = run_compiled(mod.main, [], a, b)
                self.assertEqual(native, compiled, src)


UNSUPPORTED = {
    "with": "def f():\n    with x:\n        yield 1\n",
    "nonlocal": "def g():\n    v = 1\n    def f():\n        nonlocal v\n        yield v\n    return f\n",
    "yield in an expression": "def f():\n    x = 1 + (yield 2)\n",
    "yield from a non-call": "def f(g):\n    yield from g\n",
    "starred call": "def f(a):\n    yield from h(*a)\n",
    "while/else": "def f():\n    while True:\n        yield 1\n    else:\n        pass\n",
    "try/except": "def f():\n    try:\n        yield 1\n    except E:\n        pass\n",
    "return in try": "def f():\n    try:\n        yield 1\n        return 2\n    finally:\n        pass\n",
    "yield in finally": "def f():\n    try:\n        pass\n    finally:\n        yield 1\n",
    "walrus": "def f():\n    if (n := 3):\n        yield n\n",
    "lambda in comprehension": "def f(xs):\n    fs = [lambda: x for x in xs]\n    yield fs\n",
    "varargs": "def f(*a):\n    yield a\n",
}


class TestUnsupported(unittest.TestCase):
    def test_each_unsupported_construct_is_a_compile_error_with_its_line(self):
        for what, src in UNSUPPORTED.items():
            with self.subTest(what=what):
                with self.assertRaises(CompileError) as cm:
                    compile_engine.compile_source(src.encode(), "bad.py")
                self.assertRegex(str(cm.exception), r"bad\.py:\d+:")


class TestGeneratedModules(unittest.TestCase):
    def test_keyforge_compiled_is_up_to_date(self):
        for rel in compile_engine.MODULES:
            with self.subTest(module=rel):
                src, _ = compile_engine.compile_module(rel)
                with open(compile_engine.out_path(rel), encoding="utf-8") as f:
                    self.assertEqual(f.read(), src, f"run python -m tools.compile_engine ({rel} is stale)")

    def test_every_generator_function_in_the_engine_compiles(self):
        vm.load_compiled()
        from tools.ref_coverage import scan
        import os

        here = os.path.dirname(compile_engine.PACKAGE_DIR)
        n = len(scan(os.path.join(here, "keyforge")))
        compiled = {r.rid for r in vm.ROUTINES.values() if r.module.split(".")[0] in ("game", "match", "effects")}
        self.assertEqual(len(compiled), n)


class TestNativeFrameAdapter(unittest.TestCase):
    """R2: the machine with nothing compiled runs every call as a
    NativeFrame, and is still the same game."""

    def test_a_game_on_the_machine_with_no_routines_is_identical(self):
        from tools import diff_engines

        vm.load_compiled()
        saved = dict(vm.ROUTINES)
        try:
            vm.ROUTINES.clear()
            a, b = diff_engines.Engine("keyforge@compiled"), diff_engines.Engine("keyforge")
            for spec in diff_engines.golden_specs()[:6]:
                diff_engines.play(a, b, spec)
            from keyforge.config import GameConfig
            from keyforge.game import Game

            g = Game(GameConfig(seed=3), execution="compiled")
            self.assertTrue(all(type(f) is vm.NativeFrame for f in g._driver.machine.stack))
            self.assertFalse(g._driver.machine.copyable)
        finally:
            vm.ROUTINES.update(saved)


if __name__ == "__main__":
    unittest.main()

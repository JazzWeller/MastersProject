"""Random generator functions within the Part R compiler's subset (for
tests/test_part_r_compiler.py's property test).

`random_module(seed)` returns the source of a module with one entry point,
`main(ctx, a, b)`, plus helper generators it may `yield from`. Programs mix
suspensions (`yield ("ask", v)`, `yield from helper(...)`), assignments,
`if`/`while`/`for` with `break`/`continue`, `try/finally`, nested functions
and lambdas over variables that are reassigned later (cell semantics), and
early returns. Every loop is bounded, every value an int, and `ctx` (a list)
records side effects, so a run is a pure function of its choices.
"""

from __future__ import annotations

import random
from typing import List

VARS = ("v0", "v1", "v2", "v3")


class _Gen:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.lines: List[str] = []
        self.counter = 0
        self.closures = 0

    def fresh(self, stem: str) -> str:
        self.counter += 1
        return f"{stem}{self.counter}"

    def expr(self, depth: int = 0) -> str:
        r = self.rng
        if depth > 1 or r.random() < 0.4:
            return r.choice([*VARS, str(r.randint(0, 9))])
        op = r.choice(["+", "-", "*", "%"])
        a, b = self.expr(depth + 1), self.expr(depth + 1)
        if op == "%":
            return f"({a} % ({b} % 7 + 1))"
        return f"({a} {op} {b})"

    def cond(self) -> str:
        return f"{self.expr()} % {self.rng.randint(2, 4)} == {self.rng.randint(0, 1)}"

    def emit(self, ind: int, text: str):
        self.lines.append("    " * ind + text)

    def block(self, ind: int, depth: int, in_loop: bool, in_try: bool, n: int):
        for _ in range(n):
            self.stmt(ind, depth, in_loop, in_try)

    def stmt(self, ind: int, depth: int, in_loop: bool, in_try: bool):
        r = self.rng
        kinds = ["assign", "assign", "ask", "ask", "call", "record"]
        if depth < 3:
            kinds += ["if", "while", "for", "for_list"]
            if not in_try:
                kinds.append("try")
        if in_loop:
            kinds += ["break", "continue"]
        if not in_try:
            kinds.append("return")
        kinds += ["closure", "lambda"]
        kind = r.choice(kinds)
        if kind == "assign":
            self.emit(ind, f"{r.choice(VARS)} = {self.expr()}")
        elif kind == "ask":
            target = r.choice(VARS)
            self.emit(ind, f"{target} = yield ('ask', {self.expr()})")
        elif kind == "call":
            target = r.choice(VARS)
            helper = r.choice(["helper_a", "helper_b"])
            if r.random() < 0.3:
                self.emit(ind, f"yield from {helper}(ctx, {self.expr()})")
            else:
                self.emit(ind, f"{target} = yield from {helper}(ctx, {self.expr()})")
        elif kind == "record":
            self.emit(ind, f"ctx.append({self.expr()})")
        elif kind == "if":
            if r.random() < 0.15:
                self.emit(ind, f"if (yield from helper_a(ctx, {self.expr()})):")
            else:
                self.emit(ind, f"if {self.cond()}:")
            self.block(ind + 1, depth + 1, in_loop, in_try, r.randint(1, 3))
            if r.random() < 0.6:
                if r.random() < 0.3:
                    self.emit(ind, f"elif {self.cond()}:")
                    self.block(ind + 1, depth + 1, in_loop, in_try, r.randint(1, 2))
                self.emit(ind, "else:")
                self.block(ind + 1, depth + 1, in_loop, in_try, r.randint(1, 2))
        elif kind == "while":
            c = self.fresh("c")
            self.emit(ind, f"{c} = 0")
            self.emit(ind, f"while {c} < {r.randint(1, 4)}:")
            self.emit(ind + 1, f"{c} += 1")
            self.block(ind + 1, depth + 1, True, in_try, r.randint(1, 3))
        elif kind == "for":
            i = self.fresh("i")
            self.emit(ind, f"for {i} in range({r.randint(0, 3)}):")
            self.emit(ind + 1, f"ctx.append({i})")
            self.block(ind + 1, depth + 1, True, in_try, r.randint(1, 3))
        elif kind == "for_list":
            lst, x = self.fresh("lst"), self.fresh("x")
            self.emit(ind, f"{lst} = [{self.expr()}, {self.expr()}]")
            self.emit(ind, f"for {x} in {lst}:")
            self.emit(ind + 1, f"if len({lst}) < 4:")
            self.emit(ind + 2, f"{lst}.append({x} + 1)")  # the loop reads the list live
            self.emit(ind + 1, f"ctx.append({x})")
            self.block(ind + 1, depth + 1, True, in_try, r.randint(1, 2))
        elif kind == "try":
            self.emit(ind, "try:")
            self.block(ind + 1, depth + 1, False, True, r.randint(1, 3))
            self.emit(ind, "finally:")
            self.emit(ind + 1, f"ctx.append(('fin', {self.expr()}))")
        elif kind == "break":
            self.emit(ind, f"if {self.cond()}:")
            self.emit(ind + 1, "break")
        elif kind == "continue":
            self.emit(ind, f"if {self.cond()}:")
            self.emit(ind + 1, "continue")
        elif kind == "return":
            self.emit(ind, f"if {self.cond()}:")
            self.emit(ind + 1, f"return ('early', {self.expr()}, ctx)")
        elif kind == "closure":
            name = self.fresh("g")
            var = r.choice(VARS)
            self.emit(ind, f"def {name}(k):")
            self.emit(ind + 1, f"return {var} + k")
            self.emit(ind, f"CLOSURES.append({name})")
        elif kind == "lambda":
            var = r.choice(VARS)
            self.emit(ind, f"CLOSURES.append(lambda k: {var} * 2 + k)")


HEADER = '''
CLOSURES = []


def helper_a(ctx, v):
    ctx.append(("a", v))
    w = yield ("helper_a", v)
    return (w + v) % 17


def helper_b(ctx, v):
    if v % 2:
        return v
        yield
    total = 0
    for k in range(2):
        got = yield ("helper_b", v, k)
        total += got
    return total
'''


def random_module(seed: int) -> str:
    rng = random.Random(seed)
    g = _Gen(rng)
    g.emit(0, "def main(ctx, a, b):")
    g.emit(1, "v0, v1, v2, v3 = a, b, a + b, 1")
    g.block(1, 0, False, False, rng.randint(3, 8))
    g.emit(1, "v0 = yield ('end', v0)")
    g.emit(1, "ctx.append([f(1) for f in CLOSURES])")
    g.emit(1, "return ('done', v0, v1, v2, v3, ctx)")
    return HEADER + "\n\n" + "\n".join(g.lines) + "\n"

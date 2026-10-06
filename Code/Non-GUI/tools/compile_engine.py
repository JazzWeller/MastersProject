#!/usr/bin/env python3
"""The Part R compiler (Agent Observation Plan, R3): every generator function
in the engine becomes an explicit state machine, a *routine* run by
`keyforge/vm.py`'s `Machine`.

    python -m tools.compile_engine            # (re)write keyforge/compiled/
    python -m tools.compile_engine --check    # exit 1 if any output is stale

For each module it parses the source, finds every generator function (its
own `yield`s, not a nested function's), checks the function against the
supported subset, and emits one routine per function into
`keyforge/compiled/<module>.py`. Card effects keep being written as
generators; this is the second way the same source runs.

**How a routine works.** It keeps the function's own `if`, `while`, `for`
and `try` statements, and its code between suspension points is the
original code, run natively. Each suspension point gets a resume id;
`_pc == 0` means running normally, and a resumed frame finds its way back to
the point it suspended at through guards on the way (see
`FunctionCompiler`). The function's locals live in the frame (`Frame.L`, a
fixed slot order) only while it is suspended.

- `x = yield Decision(...)` saves the locals and returns a `Suspended`
  holding the decision; the choice comes back as `_sent` when the frame is
  resumed.
- `x = yield from f(...)` calls `_kf_step(f)` -- `f`'s routine, or a runner
  for code that isn't compiled -- with the original arguments, inline. If it
  returns a value, the routine carries on with it; if it returns a
  `Suspended`, something under it is waiting on a decision, so the routine
  saves its locals, adds its own frame and returns the same `Suspended`.
- `return v` returns `v`.
- A `for` loop that suspends keeps its sequence and index in the frame and
  reads a list live, by index, exactly as Python's list iterator does.
- Variables a nested function or lambda captures are cells, as CPython makes
  them: the frame holds the cell, and the nested function is built from the
  ORIGINAL code object with a closure of those cells
  (`vm.make_function`), so it is indistinguishable from the one the native
  generator would have made -- same code, same qualname, same sharing.

**The supported subset** is what the 2026-09-30 census found. A yield may
be: a whole expression statement, the whole right-hand side of a plain
assignment (or of an augmented assignment to a local), the whole value of a
`return`, or the whole test of an `if`.
Statements that contain a suspension may be `if`/`elif`/`else`, `while`
(no `else`), `for` (no `else`) and `try/finally` (no `except`, `else`, or
`return`/`break`/`continue` out of it, no suspension inside the `finally`).
`yield from` takes a call without `*`/`**` arguments. `nonlocal`, `global`,
`with`, `del` of a local, a walrus or a yield inside a comprehension, and a
nested function inside a comprehension are not supported. Anything outside
the subset is a `CompileError` naming its source line -- never a silent
miscompile.

**Output** is checked in. Each generated module records the SHA-256 of its
source; the loader refuses a stale one, and `--check` (run by
tests/test_part_r_compiler.py) fails when any output is out of date.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import os
import sys
from typing import Dict, List, Optional, Sequence, Set, Tuple

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE_DIR = os.path.join(_HERE, "keyforge")
OUT_DIR = os.path.join(PACKAGE_DIR, "compiled")

# Source modules, relative to keyforge/: every module with generator
# functions.
MODULES = (
    "game.py",
    "match.py",
    "effects/steps.py",
    "effects/generic.py",
    "effects/named/brobnar.py",
    "effects/named/dis.py",
    "effects/named/logos.py",
    "effects/named/mars.py",
    "effects/named/sanctum.py",
    "effects/named/shadows.py",
    "effects/named/untamed.py",
)



class CompileError(Exception):
    def __init__(self, where: str, node: Optional[ast.AST], what: str):
        line = getattr(node, "lineno", "?")
        super().__init__(f"{where}:{line}: {what}")
        self.line = line


# ------------------------------------------------------------- analysis ----

_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
_COMPS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def iter_own(node: ast.AST):
    """Every node in `node`'s own scope: not inside nested functions,
    lambdas or classes (their headers' defaults and decorators ARE in this
    scope; their bodies are not). Comprehensions are included."""
    stack = [node]
    first = True
    while stack:
        n = stack.pop()
        if not first:
            yield n
        if isinstance(n, _SCOPES) and not first:
            # defaults/decorators evaluate in the enclosing scope
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                stack.extend(reversed(n.args.defaults))
                stack.extend(d for d in reversed(n.args.kw_defaults) if d is not None)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                stack.extend(reversed(n.decorator_list))
            continue
        first = False
        stack.extend(reversed(list(ast.iter_child_nodes(n))))


def own_yields(func) -> List[ast.AST]:
    return [n for n in iter_own(func) if isinstance(n, (ast.Yield, ast.YieldFrom))]


def has_own_yield(node: ast.AST) -> bool:
    """Whether `node` (a statement or expression of the function being
    compiled) contains a yield of that function's own. A nested `def`,
    lambda or class statement contains only its header's."""
    if isinstance(node, (ast.Yield, ast.YieldFrom)):
        return True
    if isinstance(node, _SCOPES):
        header = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            header += node.args.defaults + [d for d in node.args.kw_defaults if d is not None]
        header += getattr(node, "decorator_list", [])
        return any(has_own_yield(h) for h in header)
    return any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in iter_own(node))


def _comp_bound(comp) -> Set[str]:
    out = set()
    for gen in comp.generators:
        for n in ast.walk(gen.target):
            if isinstance(n, ast.Name):
                out.add(n.id)
    return out


def assigned_names(func) -> Set[str]:
    """Names bound in `func`'s own scope (parameters excluded)."""
    out: Set[str] = set()

    def visit(n, in_comp: Set[str]):
        if isinstance(n, _SCOPES):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out.add(n.name)
            for d in getattr(getattr(n, "args", None), "defaults", []) or []:
                visit(d, in_comp)
            return
        if isinstance(n, _COMPS):
            bound = _comp_bound(n)
            for c in ast.iter_child_nodes(n):
                visit(c, in_comp | bound)
            return
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)) and n.id not in in_comp:
            out.add(n.id)
        if isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                out.add((a.asname or a.name).split(".")[0])
        for c in ast.iter_child_nodes(n):
            visit(c, in_comp)

    for stmt in func.body:
        visit(stmt, set())
    return out


def param_names(args: ast.arguments) -> List[str]:
    names = [a.arg for a in args.posonlyargs + args.args]
    if args.vararg:
        names.append(args.vararg.arg)
    names += [a.arg for a in args.kwonlyargs]
    if args.kwarg:
        names.append(args.kwarg.arg)
    return names


def scope_locals(func) -> Set[str]:
    return set(param_names(func.args)) | assigned_names(func)


def free_names(func, enclosing_locals: Set[str]) -> Set[str]:
    """Names `func` (a nested function or lambda) reads or writes that
    resolve to `enclosing_locals` -- its free variables there, including
    those of functions nested in it."""
    mine = scope_locals(func) if not isinstance(func, ast.Lambda) else set(param_names(func.args))
    out: Set[str] = set()
    body = func.body if isinstance(func.body, list) else [func.body]

    def visit(n, shadow: Set[str]):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            out.update(free_names(n, enclosing_locals - (mine | shadow)) )
            return
        if isinstance(n, ast.ClassDef):
            return
        if isinstance(n, _COMPS):
            bound = _comp_bound(n)
            for c in ast.iter_child_nodes(n):
                visit(c, shadow | bound)
            return
        if isinstance(n, ast.Name) and n.id not in mine and n.id not in shadow and n.id in enclosing_locals:
            out.add(n.id)
        for c in ast.iter_child_nodes(n):
            visit(c, shadow)

    for b in body:
        visit(b, set())
    for d in func.args.defaults + [d for d in func.args.kw_defaults if d is not None]:
        pass  # defaults evaluate in the enclosing scope, not the nested one
    return out


def nested_functions(func) -> List[ast.AST]:
    """The nested `def`s and lambdas in `func`'s own scope, in source order
    (the order CPython puts their code objects in `co_consts`)."""
    found = [n for n in iter_own(func) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))]
    return sorted(found, key=lambda n: (n.lineno, n.col_offset))


# -------------------------------------------------------------- emission ----

_SUSPENDING = (ast.Yield, ast.YieldFrom)


class FunctionCompiler:
    """Compiles one generator function (`func`, its AST) into a routine's
    source text. `freevars`: names it closes over from enclosing functions
    (the routine gets the closure's cells).

    **Structured emission.** The routine keeps the function's own `if`,
    `while`, `for` and `try` statements. Its suspension points get resume ids
    1..n in source order, so the ids inside any one statement form a
    contiguous range. `_pc == 0` means "running normally"; a resumed frame
    starts with `_pc` at the id it suspended at, and the code finds its way
    back there: every run of plain statements that comes before a suspension
    point is guarded by `if not _pc:` (skipped while resuming), and every
    compound statement on the way in is entered when `_pc` falls in its
    range. Reaching the suspension point sets `_pc` back to 0. Running
    normally, a guard costs one test; there is no dispatch."""

    def __init__(self, where: str, func: ast.FunctionDef, qualname: str, freevars: Sequence[str]):
        self.where = where
        self.func = func
        self.qualname = qualname
        self.params = param_names(func.args)
        self.locals = sorted(assigned_names(func) - set(self.params))
        own = set(self.params) | set(self.locals)
        nested = nested_functions(func)
        cell: Set[str] = set()
        for n in nested:
            cell |= free_names(n, own | set(freevars))
        self.freevars = sorted(set(freevars))
        self.cellvars = sorted(cell & own)
        self.celled = set(self.cellvars) | set(self.freevars)
        self.nested = nested
        self.nested_index = {(n.lineno, n.col_offset): i for i, n in enumerate(nested)}
        self.temps: List[str] = []
        self.for_temps: Dict[int, Tuple[str, str]] = {}
        self.has_try = False
        self._check_subset()
        body = func.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            body = body[1:]  # the docstring
        self.body = self._normalize(body, in_try=False, in_loop=False)
        self.ids: Dict[int, int] = {}  # id(statement) -> resume id
        self.ranges: Dict[int, Tuple[int, int]] = {}  # id(compound statement) -> (lo, hi)
        self.line_of_pc: Dict[int, int] = {}
        self.exc_points: List[int] = []  # resume ids inside a try region
        self._number(self.body, in_try=False)
        # R7: what each resume id waits on, and what is still reachable
        # from it
        self.site_of_pc: Dict[int, str] = {}
        self.ops_of_pc: Dict[int, tuple] = {}
        self._remaining(self.body, [])

    # ------------------------------------------------------------ subset ----

    def err(self, node, what):
        raise CompileError(self.where, node, f"{self.qualname}: {what}")

    def _check_subset(self):
        a = self.func.args
        if a.vararg or a.kwarg or a.posonlyargs:
            self.err(self.func, "*args, **kwargs and positional-only parameters are not supported")
        if self.func.decorator_list:
            self.err(self.func, "decorated generator functions are not supported")
        for n in iter_own(self.func):
            if isinstance(n, (ast.Nonlocal, ast.Global)):
                self.err(n, "nonlocal/global is not supported")
            if isinstance(n, (ast.With, ast.AsyncWith, ast.AsyncFor, ast.Await, ast.Match)):
                self.err(n, f"{type(n).__name__} is not supported")
            if isinstance(n, ast.Delete):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        self.err(n, "del of a local is not supported")
            if isinstance(n, ast.NamedExpr):
                self.err(n, "assignment expressions are not supported")
            if isinstance(n, _COMPS):
                for m in ast.walk(n):
                    if isinstance(m, _SUSPENDING):
                        self.err(n, "a yield inside a comprehension is not supported")
                    if isinstance(m, (ast.Lambda, ast.FunctionDef)):
                        self.err(n, "a nested function inside a comprehension is not supported")
            if isinstance(n, ast.YieldFrom):
                v = n.value
                if not isinstance(v, ast.Call):
                    self.err(n, "yield from must take a call")
                if any(isinstance(x, ast.Starred) for x in v.args) or any(k.arg is None for k in v.keywords):
                    self.err(n, "yield from f(*args/**kwargs) is not supported")

    def temp(self, kind: str) -> str:
        name = f"_kf{kind}{len(self.temps)}"
        self.temps.append(name)
        return name

    @staticmethod
    def _suspension(s: ast.stmt):
        """The yield node if `s` is a suspension statement of the subset."""
        if isinstance(s, (ast.Expr, ast.Assign, ast.AugAssign, ast.Return)) and isinstance(s.value, _SUSPENDING):
            return s.value
        return None

    def _normalize(self, stmts: List[ast.stmt], *, in_try: bool, in_loop: bool) -> List[ast.stmt]:
        """A copy of `stmts` with `if (yield ...)` hoisted into a temporary,
        and everything outside the subset rejected."""
        out: List[ast.stmt] = []
        for s in stmts:
            if not has_own_yield(s):
                if in_try and self._returns_inside(s):
                    self.err(s, "return inside a suspending try/finally is not supported")
                if in_try and self._breaks_out(s):
                    self.err(s, "break/continue out of a suspending try/finally is not supported")
                out.append(s)
                continue
            y = self._suspension(s)
            if y is not None:
                if isinstance(y, ast.Yield) and y.value is not None and has_own_yield(y.value):
                    self.err(s, "a nested yield is not supported")
                if isinstance(y, ast.YieldFrom) and any(has_own_yield(x) for x in [y.value.func] + y.value.args + [k.value for k in y.value.keywords]):
                    self.err(s, "a nested yield is not supported")
                if isinstance(s, ast.AugAssign) and not isinstance(s.target, ast.Name):
                    self.err(s, "augmented assignment of a yield to anything but a local is not supported")
                if isinstance(s, ast.Assign) and any(has_own_yield(t) for t in s.targets):
                    self.err(s, "a yield inside an assignment target is not supported")
                if isinstance(s, ast.Return) and in_try:
                    self.err(s, "return inside a suspending try/finally is not supported")
                out.append(s)
            elif isinstance(s, ast.If):
                test = s.test
                if isinstance(test, _SUSPENDING):
                    t = self.temp("t")
                    out.append(ast.copy_location(ast.Assign(targets=[ast.Name(t, ast.Store())], value=test), s))
                    test = ast.Name(t, ast.Load())
                elif has_own_yield(test):
                    self.err(s, "a yield inside an if test must be the whole test")
                out.append(ast.copy_location(ast.If(test=test, body=self._normalize(s.body, in_try=in_try, in_loop=in_loop),
                                                    orelse=self._normalize(s.orelse, in_try=in_try, in_loop=in_loop)), s))
            elif isinstance(s, ast.While):
                if s.orelse:
                    self.err(s, "while/else is not supported")
                if has_own_yield(s.test):
                    self.err(s, "a yield inside a while test is not supported")
                out.append(ast.copy_location(ast.While(test=s.test, body=self._normalize(s.body, in_try=False, in_loop=True), orelse=[]), s))
            elif isinstance(s, ast.For):
                if s.orelse:
                    self.err(s, "for/else is not supported")
                if has_own_yield(s.iter) or has_own_yield(s.target):
                    self.err(s, "a yield inside a for header is not supported")
                node = ast.copy_location(ast.For(target=s.target, iter=s.iter, body=self._normalize(s.body, in_try=False, in_loop=True),
                                                 orelse=[], type_comment=None), s)
                # the loop's sequence and index live in the frame
                self.for_temps[id(node)] = (self.temp("s"), self.temp("i"))
                out.append(node)
            elif isinstance(s, ast.Try):
                if s.handlers or s.orelse:
                    self.err(s, "only try/finally is supported around a suspension")
                if any(has_own_yield(x) for x in s.finalbody):
                    self.err(s, "a suspension inside a finally block is not supported")
                if any(self._breaks_out(x) for x in s.body):
                    self.err(s, "break/continue out of a suspending try/finally is not supported")
                self.has_try = True
                out.append(ast.copy_location(ast.Try(body=self._normalize(s.body, in_try=True, in_loop=in_loop), handlers=[], orelse=[],
                                                     finalbody=s.finalbody), s))
            else:
                self.err(s, f"a suspension inside a {type(s).__name__} statement is not supported")
        return out

    def _breaks_out(self, stmt) -> bool:
        """Whether `stmt` contains a break/continue that belongs to a loop
        outside it."""

        def visit(n, in_loop):
            if isinstance(n, _SCOPES):
                return False
            if isinstance(n, (ast.Break, ast.Continue)) and not in_loop:
                return True
            if isinstance(n, (ast.For, ast.While)):
                return any(visit(c, True) for c in n.body + n.orelse)
            return any(visit(c, in_loop) for c in ast.iter_child_nodes(n))

        return visit(stmt, False)

    def _returns_inside(self, stmt) -> bool:
        if isinstance(stmt, _SCOPES):
            return False  # a nested function's return is its own
        return isinstance(stmt, ast.Return) or any(isinstance(n, ast.Return) for n in iter_own(stmt))

    def _number(self, stmts: List[ast.stmt], *, in_try: bool) -> Optional[Tuple[int, int]]:
        lo = hi = None
        for s in stmts:
            r = None
            if self._suspension(s) is not None:
                k = len(self.ids) + 1
                self.ids[id(s)] = k
                self.line_of_pc[k] = s.lineno
                if in_try:
                    self.exc_points.append(k)
                r = (k, k)
            elif isinstance(s, ast.If):
                a = self._number(s.body, in_try=in_try)
                b = self._number(s.orelse, in_try=in_try)
                r = _join(a, b)
            elif isinstance(s, (ast.While, ast.For)):
                r = self._number(s.body, in_try=in_try)
            elif isinstance(s, ast.Try):
                r = self._number(s.body, in_try=True)
            if r is not None:
                self.ranges[id(s)] = r
                lo = r[0] if lo is None else lo
                hi = r[1]
        return None if lo is None else (lo, hi)

    # --------------------------------------------------------- remaining ----

    def _remaining(self, stmts: List[ast.stmt], outer: List[Tuple[Optional[ast.stmt], List[ast.stmt]]]):
        """Fills `site_of_pc` and `ops_of_pc` (Part R, R7). `outer`, innermost
        first: for each enclosing statement, the loop it is (or None) and
        the statements that follow it."""
        for i, s in enumerate(stmts):
            following = stmts[i + 1:]
            k = self.ids.get(id(s))
            if k is not None:
                y = s.value
                self.site_of_pc[k] = _callee_text(y.value.func) if isinstance(y, ast.YieldFrom) else "decision"
                ops = list(_ops_of(following))
                for loop, fol in outer:
                    if loop is not None:
                        ops.append(("loop", loop.lineno))
                        ops.extend(_ops_of(loop.body))
                    ops.extend(_ops_of(fol))
                self.ops_of_pc[k] = tuple(ops)
            elif isinstance(s, ast.If):
                self._remaining(s.body, [(None, following)] + outer)
                self._remaining(s.orelse, [(None, following)] + outer)
            elif isinstance(s, (ast.While, ast.For)):
                self._remaining(s.body, [(s, following)] + outer)
            elif isinstance(s, ast.Try):
                self._remaining(s.body, [(None, s.finalbody + following)] + outer)

    # ------------------------------------------------------------- emit ----

    def compile(self) -> "CompiledRoutine":
        self.slots = [f"_kfc_{n}" if n in self.cellvars else n for n in self.params] + \
                     [f"_kfc_{n}" if n in self.cellvars else n for n in self.locals] + \
                     [f"_kfc_{n}" for n in self.freevars] + self.temps
        if self.has_try:
            # an exception being delivered to this frame by the machine
            self.slots.append("_kfxe")
        return CompiledRoutine(self)


def _join(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return (min(a[0], b[0]), max(a[1], b[1]))


def _parse_stmt(src: str, line: int) -> ast.stmt:
    s = ast.parse(src).body[0]
    for n in ast.walk(s):
        n.lineno = line
    return s


class _Rewriter(ast.NodeTransformer):
    """Within one routine's own code: a celled variable `x` becomes
    `_kfc_x.cell_contents`, and a nested function or lambda is rebuilt from
    its original code object."""

    def __init__(self, fc: FunctionCompiler):
        self.fc = fc
        self.shadow: List[Set[str]] = []

    def _celled(self, name: str) -> bool:
        return name in self.fc.celled and not any(name in s for s in self.shadow)

    def visit_Name(self, node: ast.Name):
        if self._celled(node.id):
            return ast.copy_location(ast.Attribute(ast.Name(f"_kfc_{node.id}", ast.Load()), "cell_contents", node.ctx), node)
        return node

    def _comp(self, node):
        bound = _comp_bound(node)
        # the first generator's iterable evaluates in the enclosing scope
        first = node.generators[0]
        first.iter = self.visit(first.iter)
        self.shadow.append(bound)
        for i, gen in enumerate(node.generators):
            if i:
                gen.iter = self.visit(gen.iter)
            gen.ifs = [self.visit(x) for x in gen.ifs]
        for f in ("elt", "key", "value"):
            if hasattr(node, f):
                setattr(node, f, self.visit(getattr(node, f)))
        self.shadow.pop()
        return node

    visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = _comp

    def _make_fn(self, node):
        i = self.fc.nested_index[(node.lineno, node.col_offset)]
        defaults = ast.Tuple([self.visit(d) for d in node.args.defaults], ast.Load()) if node.args.defaults else ast.Constant(None)
        kw = [(a.arg, d) for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults) if d is not None]
        kwdefaults = ast.Dict([ast.Constant(k) for k, _ in kw], [self.visit(d) for _, d in kw]) if kw else ast.Constant(None)
        own = set(self.fc.params) | set(self.fc.locals) | set(self.fc.freevars)
        names = sorted(free_names(node, own))
        cells = ast.Dict([ast.Constant(n) for n in names], [ast.Name(f"_kfc_{n}", ast.Load()) for n in names])
        return ast.Call(ast.Name("_kf_fn", ast.Load()),
                        [ast.Subscript(ast.Name("_kfN", ast.Load()), ast.Constant(i), ast.Load()),
                         ast.Call(ast.Name("globals", ast.Load()), [], []), defaults, kwdefaults, cells], [])

    def visit_Lambda(self, node: ast.Lambda):
        return ast.copy_location(self._make_fn(node), node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if node.decorator_list:
            self.fc.err(node, "decorated nested functions are not supported")
        target = ast.Name(node.name, ast.Store())
        target = self.visit(target)
        return ast.copy_location(ast.Assign(targets=[target], value=self._make_fn(node)), node)

    def visit_ClassDef(self, node):
        self.fc.err(node, "a class inside a generator function is not supported")


class CompiledRoutine:
    def __init__(self, fc: FunctionCompiler):
        self.fc = fc
        self.rid = f"{fc.where[:-3].replace('/', '.')}:{fc.qualname}"
        safe = fc.qualname.replace(".<locals>.", "__").replace(".", "_").replace("<", "").replace(">", "")
        self.name = safe
        self.rw = _Rewriter(fc)
        self.save = f"[{', '.join(fc.slots)}]"

    # ------------------------------------------------------------ pieces ----

    def code(self, node) -> List[str]:
        n2 = self.rw.visit(copy.deepcopy(node))
        ast.fix_missing_locations(n2)
        return ast.unparse(n2).splitlines()

    def expr(self, node) -> str:
        n2 = self.rw.visit(copy.deepcopy(node))
        ast.fix_missing_locations(n2)
        return ast.unparse(n2)

    def call_expr(self, call: ast.Call) -> str:
        """`yield from f(args)` as a call of f's routine with the original
        arguments. `x.m(args)` with `m` a compiled method of `Game` goes
        straight to Game's routine when `x` really is a Game -- no lookup and
        no bound methods -- and through `_kf_step` otherwise."""
        generic = self.expr(ast.Call(ast.Call(ast.Name("_kf_step", ast.Load()), [call.func], []), call.args, call.keywords))
        f = call.func
        if (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.attr in GAME_ROUTINES):
            obj = self.expr(f.value)
            direct = self.expr(ast.Call(ast.Name(f"_kfgm_{f.attr}", ast.Load()), [f.value] + call.args, call.keywords))
            return f"({direct} if type({obj}) is _kfGame else {generic})"
        return generic

    def seq_expr(self, it: ast.expr) -> str:
        """What a suspending `for` loop iterates by index: the iterable itself
        when it is visibly a list, tuple or range, else `_kf_seq(...)`."""
        known = isinstance(it, (ast.List, ast.Tuple)) or (
            isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id in ("list", "tuple", "range", "sorted")
            and it.func.id not in self.fc.params and it.func.id not in self.fc.locals)
        return self.expr(it) if known else f"_kf_seq({self.expr(it)})"

    def signature(self) -> str:
        """The original parameters, each with a placeholder default (the
        loader installs the original's own defaults; a resumed frame passes
        none), then `_kfF` (the frame being resumed, None for a fresh call)
        and `_sent` (what it was waiting for) -- positional, so a fresh call
        fills them from the defaults tuple, not by name -- and, for a nested
        function, `_kf_closure` (the cells it closes over)."""
        a = self.fc.func.args
        params = [f"{p.arg}=_kf_U" for p in a.args]
        params += ["_kfF=None", "_sent=None", "*"]
        params += [f"{p.arg}=_kf_U" for p in a.kwonlyargs]
        params += ["_kf_closure=None"]
        return ", ".join(params)

    def _suspend_here(self, k: int, what: str) -> List[str]:
        """Suspends at resume id `k` with `what` (a `Suspended`): a fresh
        call makes its frame and adds it to the suspension's frames; a
        resumed frame updates its own."""
        return ["if _kfF is None:", f"    _kfF = _kf_Frame(_kfR, {self.save})", f"    _kfF.pc = {k}",
                f"    {what}.frames.append(_kfF)", "else:", f"    _kfF.L = {self.save}", f"    _kfF.pc = {k}",
                f"return {what}"]

    def _consume(self, s: ast.stmt) -> List[str]:
        """The rest of suspension statement `s`, once its value is `_sent`."""
        sent = ast.Name("_sent", ast.Load())
        if isinstance(s, ast.Expr):
            return []
        if isinstance(s, ast.Assign):
            return self.code(ast.Assign(targets=s.targets, value=sent, lineno=s.lineno))
        if isinstance(s, ast.AugAssign):
            return self.code(ast.AugAssign(target=s.target, op=s.op, value=sent, lineno=s.lineno))
        return self.code(ast.Return(value=sent, lineno=s.lineno))

    # -------------------------------------------------------- statements ----

    def emit_list(self, stmts: List[ast.stmt], ind: str) -> List[str]:
        fc = self.fc
        out: List[str] = []
        ranges = [fc.ranges.get(id(s)) for s in stmts]
        last = max((i for i, r in enumerate(ranges) if r is not None), default=-1)
        group: List[ast.stmt] = []

        def flush(guard: bool):
            if not group:
                return
            lines = [ln for st in group for ln in self.code(st)]
            if guard:
                out.append(ind + "if not _pc:")
                out.extend(ind + "    " + ln for ln in lines)
            else:
                out.extend(ind + ln for ln in lines)
            group.clear()

        for i, (s, r) in enumerate(zip(stmts, ranges)):
            if r is None:
                group.append(s)
                continue
            flush(True)
            out.extend(self.emit_stmt(s, r, ind))
        flush(False)  # after the last suspension point nothing is skipped
        if not out:
            out.append(ind + "pass")
        return out

    def emit_stmt(self, s: ast.stmt, r: Tuple[int, int], ind: str) -> List[str]:
        fc = self.fc
        i2 = ind + "    "
        out: List[str] = []
        k = fc.ids.get(id(s))
        if k is not None:
            y = s.value
            out.append(f"{ind}if not _pc:")
            if isinstance(y, ast.YieldFrom):
                # the callee, with the ORIGINAL arguments: `_kf_step(f)` is
                # its routine (or a native runner), called as `f` would be
                call = y.value
                out.append(f"{i2}_sent = {self.call_expr(call)}")
                out.append(f"{i2}if type(_sent) is _kf_S:")
                out.extend(i2 + "    " + ln for ln in self._suspend_here(k, "_sent"))
                out.extend(i2 + ln for ln in self._consume(s))
            else:
                value = self.expr(y.value) if y.value is not None else "None"
                out.append(f"{i2}_kfd = _kf_S({value})")
                out.extend(i2 + ln for ln in self._suspend_here(k, "_kfd"))
            out.append(f"{ind}elif _pc == {k}:")
            out.append(f"{i2}_pc = 0")
            if k in fc.exc_points:
                out.append(f"{i2}if _kfxe is not None:  # an exception from below, delivered by the machine")
                out.append(f"{i2}    _kfe = _kfxe")
                out.append(f"{i2}    _kfxe = None")
                out.append(f"{i2}    raise _kfe")
            out.extend(i2 + ln for ln in self._consume(s))
            return out
        if isinstance(s, ast.If):
            rb = _range_of(s.body, fc)
            re_ = _range_of(s.orelse, fc)
            c1 = f"not _pc and ({self.expr(s.test)})"
            if rb is not None:
                c1 = f"({c1}) or {_in(rb)}"
            out.append(f"{ind}if {c1}:")
            out.extend(self.emit_list(s.body, i2))
            if s.orelse:
                c2 = "not _pc" + (f" or {_in(re_)}" if re_ is not None else "")
                out.append(f"{ind}elif {c2}:")
                out.extend(self.emit_list(s.orelse, i2))
            return out
        if isinstance(s, ast.While):
            out.append(f"{ind}while (not _pc and ({self.expr(s.test)})) or {_in(r)}:")
            out.extend(self.emit_list(s.body, i2))
            return out
        if isinstance(s, ast.For):
            seq, idx = fc.for_temps[id(s)]
            out.append(f"{ind}if not _pc:")
            out.append(f"{i2}{seq} = {self.seq_expr(s.iter)}")
            out.append(f"{i2}{idx} = 0")
            out.append(f"{ind}while (not _pc and {idx} < len({seq})) or {_in(r)}:")
            out.append(f"{i2}if not _pc:")
            target = self.code(ast.Assign(targets=[s.target], value=ast.Subscript(ast.Name(seq, ast.Load()), ast.Name(idx, ast.Load()), ast.Load()), lineno=s.lineno))
            out.extend(i2 + "    " + ln for ln in target)
            out.append(f"{i2}    {idx} += 1")
            out.extend(self.emit_list(s.body, i2))
            return out
        if isinstance(s, ast.Try):
            fin = [ln for st in s.finalbody for ln in self.code(st)]
            out.append(f"{ind}if not _pc or {_in(r)}:")
            out.append(f"{i2}try:")
            out.extend(self.emit_list(s.body, i2 + "    "))
            out.append(f"{i2}except BaseException:")
            out.extend(i2 + "    " + ln for ln in fin)
            out.append(f"{i2}    raise")
            out.extend(i2 + ln for ln in fin)
            return out
        raise AssertionError(type(s).__name__)

    # ----------------------------------------------------------- routine ----

    def source(self) -> str:
        fc = self.fc
        slots = fc.slots
        out: List[str] = []
        # The factory closes the routine over its own constants: the nested
        # code objects it rebuilds functions from, and its Routine.
        out.append(f"def _kfmk_{self.name}(_kfN, _kfR):")
        out.append(f"  def _kfr_{self.name}({self.signature()}):")
        out.append(f"    # {fc.where}:{fc.func.lineno} {fc.qualname}")
        out.append("    if _kfF is None:  # a fresh call: Python bound the arguments")
        fresh = [n for n in slots if n not in fc.params and not n.startswith("_kfc_") and n != "_kfxe"]
        if fresh:
            out.append(f"        {' = '.join(fresh)} = _kf_U")
        for n in fc.cellvars:
            out.append(f"        _kfc_{n} = _kf_cell({n})" if n in fc.params else f"        _kfc_{n} = _kf_cell()")
        for i, n in enumerate(fc.freevars):
            out.append(f"        _kfc_{n} = _kf_closure[{i}]")
        if fc.has_try:
            out.append("        _kfxe = None")
        out.append("        _pc = 0")
        out.append("    else:  # resuming a suspended frame")
        out.append(f"        ({', '.join(slots)},) = _kfF.L" if slots else "        pass")
        out.append("        _pc = _kfF.pc")
        out.extend(self.emit_list(fc.body, "    "))
        out.append("    return None")
        out.append(f"  return _kfr_{self.name}")
        return "\n".join(out) + "\n"

    def entry(self) -> str:
        fc = self.fc
        nested = tuple((getattr(n, "name", "<lambda>"), n.lineno) for n in fc.nested)
        exc = (fc.slots.index("_kfxe"), tuple(fc.exc_points)) if fc.has_try else None
        return (f"    ({self.fc.qualname!r}, {fc.func.lineno}, _kfmk_{self.name}, {tuple(fc.slots)!r}, "
                f"{tuple(fc.freevars)!r}, {nested!r}, {fc.line_of_pc!r}, {exc!r}, {fc.site_of_pc!r}, {fc.ops_of_pc!r}),")


# Receivers whose method calls are rules operations worth listing in the
# remaining-operations tables (R7): the game, the effect steps module.
_OP_RECEIVERS = frozenset({"self", "game", "g", "steps"})


def _callee_text(f: ast.expr) -> str:
    return ast.unparse(f)


def _literal(e: ast.expr):
    """A call argument as data, if it is a literal the table can carry:
    numbers, strings, booleans, None, `Enum.MEMBER`; else "?"."""
    if isinstance(e, ast.Constant) and isinstance(e.value, (int, float, str, bool, type(None))):
        return e.value
    if isinstance(e, ast.Attribute) and isinstance(e.value, ast.Name) and e.value.id[:1].isupper():
        return f"{e.value.id}.{e.attr}"
    return "?"


def _ops_of(stmts: List[ast.stmt]):
    """The rules operations in `stmts`, in source order: `(kind, callee,
    literal positional arguments, literal keyword arguments)`, kind
    "pause" for a `yield from` (or "decision" for a `yield`), "step" for a
    call on the game or the steps module. Nested functions are skipped:
    they run later, if at all, as their own routines."""
    found = []
    pausing = set()
    for st in stmts:
        nodes = [st] + list(iter_own(st)) if not isinstance(st, _SCOPES) else []
        for n in nodes:
            if isinstance(n, ast.YieldFrom):
                pausing.add(id(n.value))
            elif isinstance(n, ast.Yield):
                found.append((n.lineno, n.col_offset, ("decision", "yield", (), ())))
        for n in nodes:
            if not isinstance(n, ast.Call):
                continue
            f = n.func
            is_pause = id(n) in pausing
            if not is_pause and not (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in _OP_RECEIVERS):
                continue
            args = tuple(_literal(a) for a in n.args)
            kwargs = tuple((k.arg, _literal(k.value)) for k in n.keywords if k.arg is not None)
            found.append((n.lineno, n.col_offset, ("pause" if is_pause else "step", _callee_text(f), args, kwargs)))
    found.sort(key=lambda t: (t[0], t[1]))
    return [op for _l, _c, op in found]


def _range_of(stmts: List[ast.stmt], fc: FunctionCompiler) -> Optional[Tuple[int, int]]:
    r = None
    for s in stmts:
        r = _join(r, fc.ranges.get(id(s)))
    return r


def _in(r: Tuple[int, int]) -> str:
    lo, hi = r
    return f"_pc == {lo}" if lo == hi else f"{lo} <= _pc <= {hi}"


# --------------------------------------------------------------- modules ----


def _game_routines() -> frozenset:
    """The generator methods of `Game` (keyforge/game.py): the calls
    `CompiledRoutine.call_expr` sends straight to Game's routine."""
    with open(os.path.join(PACKAGE_DIR, "game.py"), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Game":
            return frozenset(n.name for n in node.body if isinstance(n, ast.FunctionDef) and own_yields(n))
    return frozenset()


GAME_ROUTINES = _game_routines()


def _generator_functions(tree: ast.Module) -> List[Tuple[str, ast.FunctionDef, List[str]]]:
    """`(qualname, node, freevars)` for every generator function in the
    module, nested ones included, with the free variables each closes over
    from enclosing functions."""
    out = []

    def visit(node, prefix: str, enclosing: List[Set[str]]):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, f"{prefix}{child.name}.", [])
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qual = f"{prefix}{child.name}"
                outer = set().union(*enclosing) if enclosing else set()
                if own_yields(child):
                    fv = sorted(free_names(child, outer)) if enclosing else []
                    out.append((qual, child, fv))
                visit(child, f"{qual}.<locals>.", enclosing + [scope_locals(child)])
            elif not isinstance(child, ast.Lambda):
                visit(child, prefix, enclosing)

    visit(tree, "", [])
    return out


def compile_module(rel: str) -> Tuple[str, List[str]]:
    """The generated module's source for `keyforge/<rel>`, and the routines
    in it."""
    path = os.path.join(PACKAGE_DIR, rel)
    with open(path, "rb") as f:
        raw = f.read()
    return compile_source(raw, rel)


def compile_source(raw: bytes, rel: str) -> Tuple[str, List[str]]:
    """`compile_module` for source bytes: `rel` names it in messages."""
    tree = ast.parse(raw.decode("utf-8"), filename=rel)
    from keyforge.compiled import source_digest

    digest = source_digest(raw)
    funcs = _generator_functions(tree)
    parts = [
        f'"""Generated by tools/compile_engine.py from keyforge/{rel} -- do not edit.',
        "",
        "Each pausing function of the source, as a routine for keyforge/vm.py's Machine:",
        "`_kfmk_*` makes a routine: called with the original arguments it runs a fresh call;",
        "called as `routine(_kfF=frame, _sent=sent)` it resumes a frame.",
        '"""',
        "",
        "# fmt: off",
        "# flake8: noqa",
        f"SOURCE = {('keyforge/' + rel)!r}",
        f"SOURCE_SHA256 = {digest!r}",
        "",
        "# Placeholder defaults: the loader rebuilds every function here in the source module's globals,",
        "# with the original function's own defaults.",
        "_kf_U = None",
        "",
    ]
    entries = []
    names = []
    for qual, node, fv in funcs:
        fc = FunctionCompiler(rel, node, qual, fv)
        cr = fc.compile()
        parts.append("")
        parts.append(cr.source())
        entries.append(cr.entry())
        names.append(qual)
    parts.append("")
    parts.append("# (qualname, first line, routine factory, slots, free variables, nested code (name, line),")
    parts.append("#  line of each resume id, (exception slot, resume ids inside a try region) or None,")
    parts.append("#  what each resume id waits on, the operations still reachable from each resume id)")
    parts.append("ROUTINES = [")
    parts.extend(entries)
    parts.append("]")
    return "\n".join(parts) + "\n", names


def load_source_module(name: str, source: str):
    """For tests: `source` as a fresh module `name`, compiled, with its
    routines registered -- `(module, routines)`."""
    import types as _types

    from keyforge.compiled import register

    module = _types.ModuleType(name)
    module.__file__ = f"<{name}>"
    exec(compile(source, f"<{name}>", "exec"), module.__dict__)
    gen_src, _names = compile_source(source.encode("utf-8"), f"{name}.py")
    gen = {}
    exec(compile(gen_src, f"<compiled {name}>", "exec"), gen)
    return module, register(gen["ROUTINES"], module, name)


def out_path(rel: str) -> str:
    return os.path.join(OUT_DIR, rel[:-3].replace("/", "_") + ".py")


def build_all() -> Dict[str, str]:
    return {rel: compile_module(rel)[0] for rel in MODULES}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="exit 1 if any generated module is out of date")
    args = parser.parse_args()
    stale = []
    total = 0
    for rel in MODULES:
        src, names = compile_module(rel)
        total += len(names)
        path = out_path(rel)
        current = open(path, encoding="utf-8").read() if os.path.exists(path) else None
        if current != src:
            stale.append(rel)
            if not args.check:
                os.makedirs(OUT_DIR, exist_ok=True)
                with open(path, "w", encoding="utf-8", newline="\n") as f:
                    f.write(src)
    if args.check:
        if stale:
            print("stale compiled modules: " + ", ".join(stale))
            sys.exit(1)
        print(f"compiled modules up to date ({total} routines)")
        return
    print(f"{total} routines in {len(MODULES)} modules; rewrote {len(stale)}")


if __name__ == "__main__":
    main()

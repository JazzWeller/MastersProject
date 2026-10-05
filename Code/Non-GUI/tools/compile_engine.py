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

**How a routine works.** Its suspension points become numbered `pc` values
and its code is split into blocks at them. Code between two suspension
points is the original code, run natively. Control flow that contains a
suspension (`if`, `while`, `for`, `try/finally`) is lowered into jumps
between blocks; everything else is left as it was. The function's locals
live in the frame (`Frame.L`, a fixed slot order): a routine loads them on
entry and stores them when it suspends.

- `x = yield Decision(...)` returns `(SUSPEND, decision)`; the choice
  arrives as `_sent` in the next block, which assigns it.
- `x = yield from f(...)` evaluates `f` and its arguments, in Python's own
  order, and returns `(CALL, f, args, kwargs)`; the callee's return value
  arrives as `_sent`.
- `return v` returns `(RETURN, v)`.
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
from dataclasses import dataclass, field
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

CALL, SUSPEND, RETURN = 1, 2, 3


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


# ---------------------------------------------------------------- blocks ----


@dataclass
class Term:
    kind: str  # goto / branch / suspend / call / return / end_finally
    target: Optional[int] = None
    other: Optional[int] = None
    test: Optional[ast.expr] = None
    value: Optional[ast.expr] = None
    func: Optional[ast.expr] = None
    args: Optional[List[ast.expr]] = None
    kwargs: Optional[List[ast.keyword]] = None
    line: int = 0


@dataclass
class Block:
    bid: int
    line: int
    stmts: List[ast.stmt] = field(default_factory=list)
    term: Optional[Term] = None


@dataclass
class _Loop:
    head: int
    end: int


class FunctionCompiler:
    """Compiles one generator function (`func`, its AST) into a routine's
    source text. `freevars`: names it closes over from enclosing
    functions (the routine gets the closure's cells)."""

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
        self.blocks: List[Block] = []
        self.temps: List[str] = []
        self.loop_stack: List[_Loop] = []
        self.try_depth = 0
        self.has_try = False
        self.cur = self.new_block(func.lineno)
        self._check_subset()

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
                    if isinstance(m, (ast.Yield, ast.YieldFrom)):
                        self.err(n, "a yield inside a comprehension is not supported")
                    if isinstance(m, (ast.Lambda, ast.FunctionDef)):
                        self.err(n, "a nested function inside a comprehension is not supported")
            if isinstance(n, ast.YieldFrom):
                v = n.value
                if not isinstance(v, ast.Call):
                    self.err(n, "yield from must take a call")
                if any(isinstance(x, ast.Starred) for x in v.args) or any(k.arg is None for k in v.keywords):
                    self.err(n, "yield from f(*args/**kwargs) is not supported")

    # ------------------------------------------------------------ blocks ----

    def new_block(self, line: int) -> Block:
        b = Block(len(self.blocks), line)
        self.blocks.append(b)
        return b

    def temp(self, kind: str) -> str:
        name = f"_kf{kind}{len(self.temps)}"
        self.temps.append(name)
        return name

    def end(self, term: Term, nxt: Optional[Block] = None):
        if self.cur.term is None:
            self.cur.term = term
        if nxt is not None:
            self.cur = nxt

    def goto(self, target: Block, line: int):
        self.end(Term("goto", target=target.bid, line=line))

    # --------------------------------------------------------- lowering ----

    def _breaks_out(self, stmt) -> bool:
        """Whether `stmt` contains a break/continue that belongs to a loop
        outside it (one being lowered)."""

        def visit(n, in_loop):
            if isinstance(n, _SCOPES):
                return False
            if isinstance(n, (ast.Break, ast.Continue)) and not in_loop:
                return True
            if isinstance(n, (ast.For, ast.While)):
                return any(visit(c, True) for c in n.body + n.orelse) or visit(getattr(n, "iter", None) or n.test, in_loop)
            return any(visit(c, in_loop) for c in ast.iter_child_nodes(n))

        return visit(stmt, False)

    def _returns_inside(self, stmt) -> bool:
        if isinstance(stmt, _SCOPES):
            return False  # a nested function's return is its own
        return isinstance(stmt, ast.Return) or any(isinstance(n, ast.Return) for n in iter_own(stmt))

    def needs_lowering(self, stmt) -> bool:
        if has_own_yield(stmt):
            return True
        if self.loop_stack and self._breaks_out(stmt):
            return True
        if self.try_depth and self._returns_inside(stmt):
            self.err(stmt, "return inside a suspending try/finally is not supported")
        return False

    def compile_body(self, stmts: List[ast.stmt]):
        for s in stmts:
            if self.cur.term is not None:
                # unreachable code after a return/break/continue: its own block
                self.cur = self.new_block(s.lineno)
            self.compile_stmt(s)

    def compile_stmt(self, s: ast.stmt):
        if not self.needs_lowering(s):
            self.cur.stmts.append(s)
            return
        line = s.lineno
        if isinstance(s, ast.Expr) and isinstance(s.value, (ast.Yield, ast.YieldFrom)):
            self._suspend(s.value, None, line)
        elif isinstance(s, ast.Assign) and isinstance(s.value, (ast.Yield, ast.YieldFrom)):
            self._suspend(s.value, s.targets, line)
        elif isinstance(s, ast.AugAssign) and isinstance(s.value, (ast.Yield, ast.YieldFrom)):
            # `x += yield ...` on a plain local only: the callee can't touch
            # a local, so applying the operator after the call is exact.
            if not isinstance(s.target, ast.Name):
                self.err(s, "augmented assignment of a yield to anything but a local is not supported")
            resume = self._suspend(s.value, None, line)
            resume.stmts.append(ast.AugAssign(target=s.target, op=s.op, value=ast.Name("_sent", ast.Load()), lineno=line))
        elif isinstance(s, ast.Return) and isinstance(s.value, (ast.Yield, ast.YieldFrom)):
            resume = self._suspend(s.value, None, line)
            resume.stmts.append(ast.Return(value=ast.Name("_sent", ast.Load())))
            self.cur.term = Term("return_stmt", line=line)
        elif isinstance(s, ast.If):
            test = s.test
            if isinstance(test, (ast.Yield, ast.YieldFrom)):
                t = self.temp("t")
                self._suspend(test, [ast.Name(t, ast.Store())], line)
                test = ast.Name(t, ast.Load())
            elif has_own_yield(test):
                self.err(s, "a yield inside an if test must be the whole test")
            then_b = self.new_block(s.body[0].lineno)
            else_b = self.new_block(s.orelse[0].lineno if s.orelse else line)
            after = self.new_block(line)
            self.end(Term("branch", target=then_b.bid, other=else_b.bid, test=test, line=line))
            self.cur = then_b
            self.compile_body(s.body)
            self.goto(after, line)
            self.cur = else_b
            self.compile_body(s.orelse)
            self.goto(after, line)
            self.cur = after
        elif isinstance(s, ast.While):
            if s.orelse:
                self.err(s, "while/else is not supported")
            if has_own_yield(s.test):
                self.err(s, "a yield inside a while test is not supported")
            head = self.new_block(line)
            body = self.new_block(s.body[0].lineno)
            after = self.new_block(line)
            self.goto(head, line)
            self.cur = head
            self.end(Term("branch", target=body.bid, other=after.bid, test=s.test, line=line))
            self.cur = body
            self.loop_stack.append(_Loop(head.bid, after.bid))
            self.compile_body(s.body)
            self.loop_stack.pop()
            self.goto(head, line)
            self.cur = after
        elif isinstance(s, ast.For):
            if s.orelse:
                self.err(s, "for/else is not supported")
            if has_own_yield(s.iter):
                self.err(s, "a yield inside a for iterable is not supported")
            seq, idx = self.temp("s"), self.temp("i")
            self.cur.stmts.append(ast.Assign(targets=[ast.Name(seq, ast.Store())],
                                             value=ast.Call(ast.Name("_kf_seq", ast.Load()), [s.iter], []), lineno=line))
            self.cur.stmts.append(ast.Assign(targets=[ast.Name(idx, ast.Store())], value=ast.Constant(0), lineno=line))
            head = self.new_block(line)
            body = self.new_block(s.body[0].lineno)
            after = self.new_block(line)
            self.goto(head, line)
            self.cur = head
            test = ast.Compare(ast.Name(idx, ast.Load()), [ast.Lt()],
                               [ast.Call(ast.Name("len", ast.Load()), [ast.Name(seq, ast.Load())], [])])
            self.end(Term("branch", target=body.bid, other=after.bid, test=test, line=line))
            self.cur = body
            body.stmts.append(ast.Assign(targets=[s.target], value=ast.Subscript(ast.Name(seq, ast.Load()), ast.Name(idx, ast.Load()), ast.Load()), lineno=line))
            body.stmts.append(ast.AugAssign(target=ast.Name(idx, ast.Store()), op=ast.Add(), value=ast.Constant(1), lineno=line))
            self.loop_stack.append(_Loop(head.bid, after.bid))
            self.compile_body(s.body)
            self.loop_stack.pop()
            self.goto(head, line)
            self.cur = after
        elif isinstance(s, ast.Try):
            if s.handlers or s.orelse:
                self.err(s, "only try/finally is supported around a suspension")
            if any(has_own_yield(x) for x in s.finalbody):
                self.err(s, "a suspension inside a finally block is not supported")
            if any(self._breaks_out(x) for x in s.body):
                self.err(s, "break/continue out of a suspending try/finally is not supported")
            fin = self.new_block(s.finalbody[0].lineno)
            after = self.new_block(line)
            self.has_try = True
            self.cur.stmts.append(_parse_stmt(f"F.handlers = [{fin.bid}] if F.handlers is None else F.handlers + [{fin.bid}]", line))
            self.try_depth += 1
            self.compile_body(s.body)
            self.try_depth -= 1
            if self.cur.term is None:
                self.cur.stmts.append(_parse_stmt("F.handlers.pop()", line))
            self.goto(fin, line)
            self.cur = fin
            self.compile_body(s.finalbody)
            self.end(Term("end_finally", target=after.bid, line=line))
            self.cur = after
        elif isinstance(s, (ast.Break, ast.Continue)):
            loop = self.loop_stack[-1]
            self.end(Term("goto", target=loop.end if isinstance(s, ast.Break) else loop.head, line=line))
        elif self.loop_stack and self._breaks_out(s) and not has_own_yield(s):
            self.err(s, f"break/continue inside a {type(s).__name__} that isn't lowered")
        else:
            self.err(s, f"a suspension inside a {type(s).__name__} statement is not supported")

    def _suspend(self, y, targets: Optional[List[ast.expr]], line: int) -> Block:
        """Ends the current block at yield `y`; returns the resume block,
        which starts by assigning `_sent` to `targets`."""
        resume = self.new_block(line)
        if isinstance(y, ast.YieldFrom):
            call = y.value
            self.end(Term("call", target=resume.bid, func=call.func, args=call.args, kwargs=call.keywords, line=line))
        else:
            if y.value is not None and has_own_yield(y.value):
                self.err(y, "a nested yield is not supported")
            self.end(Term("suspend", target=resume.bid, value=y.value, line=line))
        self.cur = resume
        if targets:
            resume.stmts.append(ast.Assign(targets=targets, value=ast.Name("_sent", ast.Load()), lineno=line))
        return resume

    # ------------------------------------------------------------- emit ----

    def compile(self) -> "CompiledRoutine":
        self.compile_body(self.func.body)
        if self.cur.term is None:
            self.cur.term = Term("return", value=None, line=getattr(self.func, "end_lineno", self.func.lineno))
        self.slots = self.params + self.locals + [f"_kfc_{n}" for n in self.cellvars if n not in self.params]
        # a celled parameter keeps its own slot name; it holds the cell
        self.slots = [f"_kfc_{n}" if n in self.cellvars else n for n in self.params] + \
                     [f"_kfc_{n}" if n in self.cellvars else n for n in self.locals] + \
                     [f"_kfc_{n}" for n in self.freevars] + self.temps
        return CompiledRoutine(self)


def _parse_stmt(src: str, line: int) -> ast.stmt:
    s = ast.parse(src).body[0]
    for n in ast.walk(s):
        n.lineno = line
    return s


class _Rewriter(ast.NodeTransformer):
    """Within one routine's own code: a celled variable `x` becomes
    `_kfc_x.cell_contents`; `return v` becomes `return (RETURN, v)`; a
    nested function or lambda is rebuilt from its original code object."""

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

    def visit_Return(self, node: ast.Return):
        value = self.visit(node.value) if node.value is not None else ast.Constant(None)
        return ast.copy_location(ast.Return(ast.Tuple([ast.Constant(RETURN), value], ast.Load())), node)

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

    def source(self) -> str:
        fc = self.fc
        rw = _Rewriter(fc)
        slots = fc.slots
        load = f"({', '.join(slots)},) = F.L" if slots else "pass"
        save = f"F.L = [{', '.join(slots)}]"
        out: List[str] = []
        out.append(f"def _kfr_{self.name}(F, _sent, _kfN):")
        out.append(f"    # {fc.where}:{fc.func.lineno} {fc.qualname}")
        body_indent = "    "
        if fc.has_try:
            out.append("    try:")
            body_indent = "        "
        out.append(f"{body_indent}{load}")
        out.append(f"{body_indent}_pc = F.pc")
        out.append(f"{body_indent}while True:")
        for k, b in enumerate(fc.blocks):
            kw = "if" if k == 0 else "elif"
            out.append(f"{body_indent}    {kw} _pc == {b.bid}:  # line {b.line}")
            ind = body_indent + "        "
            lines: List[str] = []
            for s in b.stmts:
                s2 = rw.visit(copy.deepcopy(s))
                ast.fix_missing_locations(s2)
                lines.extend(ast.unparse(s2).splitlines())
            lines.extend(self._term(b.term, rw, save))
            for ln in lines:
                out.append(ind + ln)
        if fc.has_try:
            out.append("    except BaseException:")
            out.append(f"        {save}")
            out.append("        raise")
        return "\n".join(out) + "\n"

    def _expr(self, rw, e) -> str:
        e2 = rw.visit(copy.deepcopy(e))
        ast.fix_missing_locations(e2)
        return ast.unparse(e2)

    def _term(self, t: Term, rw, save: str) -> List[str]:
        if t.kind == "goto":
            return [f"_pc = {t.target}", "continue"]
        if t.kind == "branch":
            return [f"_pc = {t.target} if {self._expr(rw, t.test)} else {t.other}", "continue"]
        if t.kind == "return":
            return [f"return ({RETURN}, None)"]
        if t.kind == "return_stmt":
            return []  # the block already ends with its own (rewritten) return
        if t.kind == "suspend":
            value = self._expr(rw, t.value) if t.value is not None else "None"
            return [f"_kfd = {value}", save, f"F.pc = {t.target}", f"return ({SUSPEND}, _kfd)"]
        if t.kind == "call":
            f = self._expr(rw, t.func)
            args = ", ".join(self._expr(rw, a) for a in t.args)
            args = f"({args},)" if t.args else "()"
            if t.kwargs:
                kwargs = "{" + ", ".join(f"{k.arg!r}: {self._expr(rw, k.value)}" for k in t.kwargs) + "}"
            else:
                kwargs = "None"
            return [f"_kff = {f}", f"_kfa = {args}", f"_kfk = {kwargs}", save, f"F.pc = {t.target}",
                    f"return ({CALL}, _kff, _kfa, _kfk)"]
        if t.kind == "end_finally":
            return ["if F.exc is not None:", "    _kfe = F.exc", "    F.exc = None", "    raise _kfe",
                    f"_pc = {t.target}", "continue"]
        raise AssertionError(t.kind)

    def binder_source(self) -> str:
        """`_kfb_<name>(<the original signature, placeholder defaults>)`:
        returns the frame's initial locals."""
        fc = self.fc
        a = fc.func.args
        params = []
        n_pos = len(a.args)
        n_def = len(a.defaults)
        for i, p in enumerate(a.args):
            params.append(p.arg + ("=_kf_U" if i >= n_pos - n_def else ""))
        if a.kwonlyargs:
            params.append("*")
            for p, d in zip(a.kwonlyargs, a.kw_defaults):
                params.append(p.arg + ("=_kf_U" if d is not None else ""))
        values = []
        for s in fc.slots:
            if s.startswith("_kfc_"):
                name = s[len("_kfc_"):]
                values.append(f"_kf_cell({name})" if name in fc.params else ("_kf_cell()" if name in fc.cellvars else "_kf_U"))
            elif s in fc.params:
                values.append(s)
            else:
                values.append("_kf_U")
        return f"def _kfb_{self.name}({', '.join(params)}):\n    return [{', '.join(values)}]\n"

    def entry(self) -> str:
        fc = self.fc
        line_of_pc = {b.bid: b.line for b in fc.blocks}
        freevars = tuple((fc.slots.index(f"_kfc_{n}"), n) for n in fc.freevars)
        nested = tuple((getattr(n, "name", "<lambda>"), n.lineno) for n in fc.nested)
        return (f"    ({self.fc.qualname!r}, {fc.func.lineno}, _kfr_{self.name}, _kfb_{self.name}, "
                f"{tuple(fc.slots)!r}, {freevars!r}, {nested!r}, {line_of_pc!r}),")


# --------------------------------------------------------------- modules ----


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
        "`_kfr_*` runs it from a frame's pc, `_kfb_*` binds a call into a frame's locals.",
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
        parts.append(cr.binder_source())
        entries.append(cr.entry())
        names.append(qual)
    parts.append("")
    parts.append("# (qualname, first line, step, binder, slots, freevar slots, nested code (name, line), line of each pc)")
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

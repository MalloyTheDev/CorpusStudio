"""Static screening for the mode-3 custom-block path (your OWN model code).

Custom code is DATA that becomes CODE only through a deliberate, auditable admission step. This module is
that step's cheap first wall: a torch-free, stdlib-``ast`` static screen over a local bundle that pins the
exact bytes it looked at and records a :class:`ModelCodeVettingReport`.

Honesty boundary (do NOT overclaim): a static screen is NECESSARY, not SUFFICIENT. It executes nothing and
CANNOT prove code safe - string-built attribute access, reflective tricks, and novel escapes are not
statically decidable. Real containment is the (gated) worker sandbox; admission stays human-gated. This
screen exists to (a) reject the obvious dangerous surface fail-closed, (b) require the declared interface
shape, and (c) produce content-addressed provenance a plan can bind admission to. It never loads or runs
the bundle, and this path never uses HF ``trust_remote_code`` (which stays ``Literal[False]``).
"""

from __future__ import annotations

import ast
import hashlib
from typing import Literal

from corpus_studio.platform.contracts import ModelCodeVettingReport, VettingFinding

# Bump when the screening rules change so a report's provenance is legible.
# Bumped when the screening RULES change, so a report produced by an older analyzer cannot be reused to
# admit code the current rules would reject (platform-plan enforces the match). 1.1.0 added interface
# conformance (model-base resolution + forward signature).
ANALYZER_VERSION = "1.1.0"

SUPPORTED_INTERFACE_VERSIONS: frozenset[str] = frozenset({"custom_decoder_v1"})

# Allowlist (not blocklist): the ONLY top-level import roots a custom block may pull. Anything else is
# refused - a smaller surface is easier to reason about than an ever-growing deny list.
_ALLOWED_IMPORT_ROOTS: frozenset[str] = frozenset(
    {"torch", "math", "typing", "dataclasses", "__future__", "collections", "functools", "enum",
     "abc", "corpus_studio_custom_block"}
)

# Import roots that are always dangerous for a from-config model block (I/O, process, reflection, pickle).
_FORBIDDEN_IMPORT_ROOTS: frozenset[str] = frozenset(
    {"os", "sys", "subprocess", "socket", "ctypes", "pickle", "marshal", "shutil", "importlib",
     "builtins", "code", "pty", "multiprocessing", "threading", "asyncio", "http", "urllib", "requests",
     "pathlib", "tempfile", "inspect", "gc", "resource", "signal", "mmap", "fcntl"}
)

# Builtins that enable arbitrary execution or reflective escapes. Calling any of these is an error.
_FORBIDDEN_CALLS: frozenset[str] = frozenset(
    {"eval", "exec", "compile", "__import__", "open", "input", "globals", "locals", "vars",
     "getattr", "setattr", "delattr", "memoryview", "breakpoint", "exit", "quit"}
)

# Attribute names that walk out of the object graph into the interpreter (the classic sandbox escapes).
_DUNDER_ESCAPES: frozenset[str] = frozenset(
    {"__globals__", "__builtins__", "__subclasses__", "__bases__", "__mro__", "__class__", "__dict__",
     "__code__", "__closure__", "__import__", "__loader__", "__reduce__", "__reduce_ex__",
     "__getattribute__", "__base__", "__init_subclass__"}
)

# Statement node types allowed at MODULE level (no import-time side effects). Executable statements
# (loops, calls, with/try) at module level are refused; put logic inside the class/functions.
_ALLOWED_MODULE_STMTS: tuple[type[ast.stmt], ...] = (
    ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef, ast.Assign, ast.AnnAssign,
)

# custom_decoder_v1 interface conformance: the entry class must be a plausible causal-LM model - it
# resolves (directly or via a locally-defined base) to a recognized model base and defines a ``forward``
# that accepts ``input_ids``. Checked statically by NAME in the AST (no import, no torch). ``Module`` is
# ``torch.nn.Module`` (its import root ``torch`` is allowlisted); HF ``PreTrainedModel`` is intentionally
# NOT listed because ``transformers`` is not an allowlisted import root, and a CorpusStudio base joins
# when its (allowlisted) module ships with the worker ABI.
_MODEL_BASES: frozenset[str] = frozenset({"Module"})


def _err(code: str, message: str, lineno: int | None = None) -> VettingFinding:
    return VettingFinding(severity="error", code=code, message=message, lineno=lineno)


def _warn(code: str, message: str, lineno: int | None = None) -> VettingFinding:
    return VettingFinding(severity="warning", code=code, message=message, lineno=lineno)


def _import_roots(node: ast.Import | ast.ImportFrom) -> list[tuple[str, int]]:
    """The top-level module root(s) an import statement pulls, with line numbers."""
    if isinstance(node, ast.Import):
        return [(alias.name.split(".", 1)[0], node.lineno) for alias in node.names]
    # ImportFrom: a relative import (level > 0) has no root module we can allowlist -> treat as forbidden.
    if node.level and node.level > 0:
        return [("", node.lineno)]
    return [((node.module or "").split(".", 1)[0], node.lineno)]


def _screen_imports(node: ast.Import | ast.ImportFrom, findings: list[VettingFinding]) -> None:
    if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
        findings.append(_err("wildcard-import", "'from ... import *' cannot be allowlisted", node.lineno))
    for root, lineno in _import_roots(node):
        if root == "":  # _import_roots emits "" for a relative import (no allowlistable root module)
            findings.append(_err("relative-import", "relative imports are not allowed", lineno))
        elif root in _FORBIDDEN_IMPORT_ROOTS:
            findings.append(_err("forbidden-import", f"import of '{root}' is not allowed", lineno))
        elif root not in _ALLOWED_IMPORT_ROOTS:
            findings.append(
                _err("import-not-allowlisted", f"import of '{root}' is not on the allowlist", lineno)
            )


def _screen_node(node: ast.AST, findings: list[VettingFinding]) -> None:
    """Screen a single node for a forbidden call or a reflective dunder escape (scanned tree-wide)."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id in _FORBIDDEN_CALLS:
            findings.append(_err("forbidden-call", f"call to '{node.func.id}' is not allowed", node.lineno))
    if isinstance(node, ast.Name) and node.id in {"eval", "exec", "compile", "__import__"}:
        findings.append(_err("forbidden-name", f"reference to '{node.id}' is not allowed", node.lineno))
    if isinstance(node, ast.Attribute) and node.attr in _DUNDER_ESCAPES:
        findings.append(
            _err("reflective-escape", f"attribute '{node.attr}' is a sandbox escape", node.lineno)
        )


def _is_docstring(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _base_name(node: ast.expr) -> str:
    """The trailing name of a base-class expression (``nn.Module`` / ``torch.nn.Module`` -> ``Module``)."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _resolves_to_model_base(entry: ast.ClassDef, classes: dict[str, ast.ClassDef]) -> bool:
    """Whether ``entry`` reaches a recognized model base directly OR through a locally-defined base class
    (``class Base(nn.Module)`` then ``class Entry(Base)``). Bounded by the local class set + a visited
    guard, so a cycle cannot loop forever."""
    seen: set[str] = set()
    stack = [entry]
    while stack:
        cls = stack.pop()
        if cls.name in seen:
            continue
        seen.add(cls.name)
        for base in cls.bases:
            name = _base_name(base)
            if name in _MODEL_BASES:
                return True
            local = classes.get(name)
            if local is not None:
                stack.append(local)
    return False


def _forward_accepts_input_ids(fn: ast.FunctionDef) -> bool:
    """The worker calls the model with ``input_ids`` (and ``labels``); a forward that cannot accept it
    would fail at call time. Accept it by name, or a ``*args`` / ``**kwargs`` that could receive it."""
    names = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
    return "input_ids" in names or fn.args.vararg is not None or fn.args.kwarg is not None


def _entry_class_present(tree: ast.Module, entry_symbol: str, findings: list[VettingFinding]) -> None:
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    entry = classes.get(entry_symbol)
    if entry is None:
        findings.append(
            _err("entry-class-missing", f"the entry class '{entry_symbol}' is not defined at module level")
        )
        return
    if not _resolves_to_model_base(entry, classes):
        findings.append(
            _err(
                "entry-not-a-model",
                f"'{entry_symbol}' must subclass a model base ({', '.join(sorted(_MODEL_BASES))}) "
                "directly or through a locally-defined base",
                entry.lineno,
            )
        )
    forward = next(
        (n for n in entry.body if isinstance(n, ast.FunctionDef) and n.name == "forward"), None
    )
    if forward is None:
        findings.append(
            _err("entry-no-forward", f"'{entry_symbol}' must define a forward() method", entry.lineno)
        )
    elif not _forward_accepts_input_ids(forward):
        findings.append(
            _err(
                "entry-forward-signature",
                f"'{entry_symbol}'.forward must accept input_ids (by name, or via **kwargs)",
                forward.lineno,
            )
        )


def vet_source(
    source: str, *, entry_symbol: str, interface_version: str = "custom_decoder_v1"
) -> list[VettingFinding]:
    """Statically screen custom-block SOURCE. Returns findings (error-severity => the bundle is rejected).
    Never executes the source."""
    findings: list[VettingFinding] = []
    if interface_version not in SUPPORTED_INTERFACE_VERSIONS:
        findings.append(
            _err("unknown-interface", f"interface_version '{interface_version}' is not supported")
        )
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        findings.append(_err("syntax-error", f"the bundle does not parse: {exc.msg}", exc.lineno))
        return findings

    for stmt in tree.body:
        if not isinstance(stmt, _ALLOWED_MODULE_STMTS) and not _is_docstring(stmt):
            findings.append(
                _err(
                    "module-side-effect",
                    f"module-level {type(stmt).__name__} is not allowed (no import-time execution)",
                    stmt.lineno,
                )
            )

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            _screen_imports(node, findings)
        _screen_node(node, findings)

    _entry_class_present(tree, entry_symbol, findings)
    return findings


def build_report(
    bundle: bytes, *, entry_symbol: str, interface_version: str = "custom_decoder_v1"
) -> ModelCodeVettingReport:
    """Screen a single-file bundle's BYTES and record a content-addressed :class:`ModelCodeVettingReport`.
    The report pins ``bundle_sha256`` so a plan can bind admission to these exact bytes."""
    # Guard the interface BEFORE constructing the report (whose interface_version is a hard Literal): an
    # unsupported interface cannot produce a report at all, so this is a clean error, not a rejection.
    if interface_version not in SUPPORTED_INTERFACE_VERSIONS:
        raise ValueError(
            f"interface_version '{interface_version}' is not supported; "
            f"choose from: {', '.join(sorted(SUPPORTED_INTERFACE_VERSIONS))}"
        )
    bundle_sha256 = hashlib.sha256(bundle).hexdigest()
    try:
        source = bundle.decode("utf-8")
        findings = vet_source(
            source, entry_symbol=entry_symbol, interface_version=interface_version
        )
    except UnicodeDecodeError:
        findings = [_err("not-utf8", "the bundle is not valid UTF-8 text")]
    verdict: Literal["admitted", "rejected"] = (
        "rejected" if any(f.severity == "error" for f in findings) else "admitted"
    )
    return ModelCodeVettingReport(
        analyzer_version=ANALYZER_VERSION,
        bundle_sha256=bundle_sha256,
        entry_symbol=entry_symbol,
        interface_version=interface_version,  # type: ignore[arg-type]
        verdict=verdict,
        findings=findings,
    )

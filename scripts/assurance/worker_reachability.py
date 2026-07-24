"""Two-sided static worker-reachability analysis (re-review #16).

The ``worker-closure`` obligation flags only a DECLARED list of worker files - it does not know which
modules the worker entrypoints actually IMPORT. So a change to a module the worker imports but that is
not on the list (e.g. ``platform/backends.py``) fires no obligation, even though it ships in the worker
package. This module computes the real graph: from the worker roots, follow intra-repo ``import`` /
``from ... import`` edges (via the stdlib ``ast``) to the transitive closure of reachable modules (a
worklist walk - the reachable set is order-independent), on BOTH the base and candidate sides, and
report the union, the delta, the modules the declared list misses, and the dynamic imports the static
graph cannot resolve.

It is pure + fail-closed: a module that cannot be read or parsed is RECORDED (never silently dropped),
and a dynamically-computed import target (``importlib.import_module(x)`` for a non-literal ``x``,
``__import__``) is recorded as UNRESOLVED - so the graph is honest about what it could not trace.

The reader is INJECTED (repo-relative path -> bytes | None), so the same analysis runs over a git tree
(read from the object store) or the working tree with no checkout - exactly like the change-set kernel's
source views.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Callable

from assurance.git_state import AssuranceError

# The package root (dir that CONTAINS the top package) and the top import package. A repo module
# ``corpus_studio.platform.worker`` therefore lives at ``engine/corpus_studio/platform/worker.py``.
DEFAULT_PACKAGE_ROOT = "engine/"
DEFAULT_TOP_PACKAGE = "corpus_studio"

# The worker ENTRYPOINTS: the roots of the reachable graph. These mirror the ``worker-closure`` policy
# globs (a test asserts they stay in sync); reachability EXTENDS them with whatever they transitively
# import. Repo-relative POSIX paths.
DEFAULT_WORKER_ROOTS: tuple[str, ...] = (
    "engine/corpus_studio/platform/worker.py",
    "engine/corpus_studio/platform/runners.py",
    "engine/corpus_studio/platform/artifacts.py",
    "engine/corpus_studio/platform/supervisor.py",
    "engine/corpus_studio/platform/execution_config.py",
    "engine/corpus_studio/platform/planner.py",
    "engine/corpus_studio/training/trainer.py",
)

# repo-relative path -> file bytes, or None if the path does not exist in this view.
ReadBytes = Callable[[str], "bytes | None"]


class WorkerReachabilityError(AssuranceError):
    """The reachability analysis cannot be produced (e.g. a root that does not resolve to a module)."""


@dataclass(frozen=True)
class Reachability:
    """One side's reachable graph. ``reachable`` is the sorted set of repo-relative module paths reached
    from the roots (INCLUDING the roots that resolved). ``unresolved_dynamic`` records import targets the
    static graph could not resolve (a non-literal ``importlib``/``__import__`` target); ``unreadable``
    records a reachable module whose bytes could not be read or parsed (fail-closed: surfaced, not
    dropped)."""

    reachable: tuple[str, ...]
    unresolved_dynamic: tuple[dict[str, str], ...]
    unreadable: tuple[dict[str, str], ...]


def _module_root_prefix(package_root: str) -> str:
    return package_root if package_root.endswith("/") or not package_root else package_root + "/"


def path_to_module(path: str, *, package_root: str = DEFAULT_PACKAGE_ROOT) -> tuple[str, bool] | None:
    """Map a repo path to ``(dotted_module, is_package)``, or None if it is not a module under the
    package root. ``.../x/__init__.py`` -> the package ``...x`` (is_package True)."""
    prefix = _module_root_prefix(package_root)
    if not path.startswith(prefix) or not path.endswith(".py"):
        return None
    rel = path[len(prefix):]
    if rel.endswith("/__init__.py"):
        return rel[: -len("/__init__.py")].replace("/", "."), True
    return rel[: -len(".py")].replace("/", "."), False


def module_to_path(module: str, read: ReadBytes, *, package_root: str = DEFAULT_PACKAGE_ROOT
                   ) -> tuple[str, bool] | None:
    """Resolve a dotted module to ``(repo_path, is_package)`` that EXISTS in this view, or None. Tries the
    plain module (``a/b.py``) then the package form (``a/b/__init__.py``)."""
    prefix = _module_root_prefix(package_root)
    rel = module.replace(".", "/")
    module_path = f"{prefix}{rel}.py"
    if read(module_path) is not None:
        return module_path, False
    package_path = f"{prefix}{rel}/__init__.py"
    if read(package_path) is not None:
        return package_path, True
    return None


def _module_and_ancestors(module: str) -> "list[str]":
    """``module`` followed by each ancestor PACKAGE dotted name, most-specific first. For
    ``corpus_studio.platform.worker`` -> ``[corpus_studio.platform.worker, corpus_studio.platform,
    corpus_studio]`` - so a reachable module drags in the __init__.py of every package that runs to import it."""
    parts = module.split(".")
    return [".".join(parts[:i]) for i in range(len(parts), 0, -1)]


def _resolve_relative(current: str, is_package: bool, level: int, module: str | None) -> str | None:
    """Resolve a relative import (``from . / .. import``) to an absolute dotted module. ``current`` is the
    importing module; ``level`` is the number of leading dots. Returns None if it escapes the top package."""
    # The anchor package of ``current``: itself if it is a package, else its parent package.
    base = current.split(".")
    if not is_package:
        base = base[:-1]
    # Each extra dot beyond the first climbs one more package level. climb == len(base) already empties the
    # anchor (i.e. climbs ABOVE the top package), so it is an escape too - use >=, not > (off-by-one).
    climb = level - 1
    if climb >= len(base):
        return None  # escapes above the top package -> unresolvable (recorded by the caller)
    anchor = base[: len(base) - climb] if climb else base
    parts = [*anchor, *(module.split(".") if module else [])]
    return ".".join(p for p in parts if p) or None


def _import_targets(source: bytes, current: str, is_package: bool, top_package: str
                    ) -> tuple[set[str], list[dict[str, str]]]:
    """Extract the intra-repo import TARGET modules from one module's source, plus the dynamic imports it
    could not statically resolve. Only targets under ``top_package`` are returned (external deps ignored)."""
    tree = ast.parse(source)
    targets: set[str] = set()
    dynamic: list[dict[str, str]] = []

    def _intra(dotted: str | None) -> None:
        if dotted and (dotted == top_package or dotted.startswith(top_package + ".")):
            targets.add(dotted)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _intra(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative
                anchor = _resolve_relative(current, is_package, node.level, node.module)
                if anchor is None:
                    dynamic.append({"kind": "relative_escapes_package",
                                    "detail": f"level={node.level} module={node.module!r} in {current}"})
                    continue
                _intra(anchor)
                for alias in node.names:  # `from .pkg import sub` - sub may itself be a submodule
                    if alias.name != "*":
                        _intra(f"{anchor}.{alias.name}")
            else:  # absolute `from a.b import c`
                _intra(node.module)
                for alias in node.names:
                    if node.module and alias.name != "*":
                        _intra(f"{node.module}.{alias.name}")
        elif isinstance(node, ast.Call):
            dyn = _dynamic_import_target(node)
            if dyn is not None:
                literal, raw = dyn
                if literal is not None:
                    _intra(literal)
                else:
                    dynamic.append({"kind": "dynamic_import", "detail": f"{raw} in {current}"})
    return targets, dynamic


def _dynamic_import_target(node: ast.Call) -> tuple[str | None, str] | None:
    """If ``node`` is a dynamic import (``importlib.import_module(...)`` / a bare ``import_module(...)`` from
    ``from importlib import import_module`` / ``__import__(...)``), return ``(literal_module_or_None,
    raw_repr)``; a literal string arg is resolvable, anything else is not."""
    func = node.func
    is_import_module = ((isinstance(func, ast.Attribute) and func.attr == "import_module")
                        or (isinstance(func, ast.Name) and func.id == "import_module"))
    is_dunder = isinstance(func, ast.Name) and func.id == "__import__"
    if not (is_import_module or is_dunder) or not node.args:
        return None
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value, first.value
    label = "importlib.import_module" if is_import_module else "__import__"
    return None, f"{label}(<non-literal>)"


def reachable_from(roots: tuple[str, ...], read: ReadBytes, *,
                   package_root: str = DEFAULT_PACKAGE_ROOT, top_package: str = DEFAULT_TOP_PACKAGE
                   ) -> Reachability:
    """Traverse the intra-repo import graph from ``roots`` (repo paths) over one view's ``read``, returning
    the reachable module SET. The traversal is a simple worklist (a stack; order is irrelevant - a closure
    is the same set whichever order it is walked). A root or reachable module missing from this view is
    simply not in the graph (it may be added/removed on the other side); one that is present but
    unreadable/unparseable is RECORDED (fail-closed)."""
    seen_paths: set[str] = set()
    dynamic: list[dict[str, str]] = []
    unreadable: list[dict[str, str]] = []
    stack: list[tuple[str, str, bool]] = []  # (module, path, is_package)

    def _enqueue(module: str) -> None:
        # Enqueue ``module`` AND every ANCESTOR PACKAGE: Python executes each parent package's __init__.py
        # when a submodule is imported, so a change to an ancestor __init__ IS worker-reachable even though
        # no explicit import edge names it. Add each that resolves to a file in this view and is unseen.
        for name in _module_and_ancestors(module):
            resolved = module_to_path(name, read, package_root=package_root)
            if resolved is None:
                continue
            path, is_pkg = resolved
            if path not in seen_paths:
                seen_paths.add(path)
                stack.append((name, path, is_pkg))

    for root in roots:  # a root absent in this view is handled by the delta on the other side
        if path_to_module(root, package_root=package_root) is None:
            raise WorkerReachabilityError(f"worker root {root!r} is not a module under {package_root!r}")
        if read(root) is not None:
            _enqueue(path_to_module(root, package_root=package_root)[0])  # type: ignore[index]

    while stack:
        module, path, is_package = stack.pop()
        source = read(path)
        if source is None:  # a path we enqueued because it resolved; a race/read failure is fail-closed
            unreadable.append({"path": path, "detail": "could not be read"})
            continue
        try:
            targets, dyn = _import_targets(source, module, is_package, top_package)
        except (SyntaxError, ValueError) as exc:  # ValueError e.g. a NUL byte in source -> record, do not abort
            unreadable.append({"path": path, "detail": f"unparseable: {exc}"})
            continue
        dynamic.extend(dyn)
        for target in targets:
            _enqueue(target)

    # De-duplicate dynamic records deterministically (same target imported from many sites -> one entry).
    dyn_unique = sorted({(d["kind"], d["detail"]): d for d in dynamic}.values(),
                        key=lambda d: (d["kind"], d["detail"]))
    return Reachability(
        reachable=tuple(sorted(seen_paths)),
        unresolved_dynamic=tuple(dyn_unique),
        unreadable=tuple(sorted(unreadable, key=lambda d: d["path"])),
    )


@dataclass(frozen=True)
class TwoSidedReachability:
    """The base-vs-candidate reachability comparison (the payload of the sealed record)."""

    worker_roots: tuple[str, ...]
    base: Reachability
    candidate: Reachability
    added_reachable: tuple[str, ...] = field(default_factory=tuple)      # reachable on candidate, not base
    removed_reachable: tuple[str, ...] = field(default_factory=tuple)    # reachable on base, not candidate
    undeclared_reachable: tuple[str, ...] = field(default_factory=tuple)  # reachable (union) but NOT a root
    distribution_impacting_paths: tuple[str, ...] = field(default_factory=tuple)  # changed AND reachable


def analyze_two_sided(roots: tuple[str, ...], base_read: ReadBytes, candidate_read: ReadBytes,
                      changed_paths: tuple[str, ...] = (), *, package_root: str = DEFAULT_PACKAGE_ROOT,
                      top_package: str = DEFAULT_TOP_PACKAGE) -> TwoSidedReachability:
    """Compute the base + candidate reachable graphs and their derived sets. ``changed_paths`` (from a
    change set) yields the distribution-impacting subset: changed files that ARE in the worker's reachable
    union - the ones that actually alter what the worker package ships."""
    base = reachable_from(roots, base_read, package_root=package_root, top_package=top_package)
    candidate = reachable_from(roots, candidate_read, package_root=package_root, top_package=top_package)
    base_set, cand_set = set(base.reachable), set(candidate.reachable)
    union = base_set | cand_set
    roots_set = set(roots)
    return TwoSidedReachability(
        worker_roots=roots,
        base=base,
        candidate=candidate,
        added_reachable=tuple(sorted(cand_set - base_set)),
        removed_reachable=tuple(sorted(base_set - cand_set)),
        undeclared_reachable=tuple(sorted(union - roots_set)),
        distribution_impacting_paths=tuple(sorted(set(changed_paths) & union)),
    )


def _reachability_to_record(r: Reachability) -> dict[str, Any]:
    return {
        "reachable_count": len(r.reachable),
        "reachable": list(r.reachable),
        "unresolved_dynamic": [dict(d) for d in r.unresolved_dynamic],
        "unreadable": [dict(d) for d in r.unreadable],
    }


def two_sided_to_payload(a: TwoSidedReachability) -> dict[str, Any]:
    """Serialize the analysis to the deterministic record payload."""
    return {
        "worker_roots": list(a.worker_roots),
        "base": _reachability_to_record(a.base),
        "candidate": _reachability_to_record(a.candidate),
        "added_reachable": list(a.added_reachable),
        "removed_reachable": list(a.removed_reachable),
        "undeclared_reachable": list(a.undeclared_reachable),
        "undeclared_reachable_count": len(a.undeclared_reachable),
        "distribution_impacting_paths": list(a.distribution_impacting_paths),
    }


WORKER_REACHABILITY_RECORD_TYPE = "worker_static_reachability"
WORKER_REACHABILITY_SCHEMA_VERSION = 1
_SUPPORTED_SCOPES = ("workspace", "head")


def _absent_or_bytes(read_disk: Callable[[], bytes]) -> "bytes | None":
    """Read bytes, mapping a genuinely-absent path to None; a real read error (permissions) propagates
    (fail-closed) rather than masquerading as 'module absent'."""
    try:
        return read_disk()
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
        return None


def build_worker_reachability_record(*, start_dir: Any, base_ref: str = "main", scope: str = "workspace",
                                     roots: tuple[str, ...] = DEFAULT_WORKER_ROOTS) -> dict[str, Any]:
    """Build a sealed ``worker_static_reachability`` record: the base vs candidate reachable graphs, their
    delta, the modules the declared worker list misses (``undeclared_reachable``), unresolved dynamic
    imports, and the changed paths that land inside the worker's reachable closure. Read-only. Fail-closed
    on an unsupported scope / a root that is not a module / a git failure."""
    from assurance import KERNEL_VERSION  # noqa: PLC0415 - lazy to keep the analyzer import light
    from assurance.git_state import (  # noqa: PLC0415
        discover_git_context,
        merge_base,
        read_committed_file,
        resolve_commit,
    )
    from assurance.records import RECORD_SCHEMA_VERSION, build_change_set_record, seal_record  # noqa: PLC0415

    if scope not in _SUPPORTED_SCOPES:
        raise WorkerReachabilityError(
            f"scope {scope!r} is not supported for reachability (supported: {', '.join(_SUPPORTED_SCOPES)})"
        )
    ctx = discover_git_context(start_dir)
    base_oid = merge_base(ctx, resolve_commit(ctx.root, base_ref))

    def base_read(path: str) -> "bytes | None":
        return read_committed_file(ctx, base_oid, path)

    if scope == "head":
        if not ctx.head_oid:
            raise WorkerReachabilityError("the 'head' scope needs a committed HEAD; HEAD is unborn")
        head_oid = ctx.head_oid

        def candidate_read(path: str) -> "bytes | None":
            return read_committed_file(ctx, head_oid, path)
    else:  # workspace: read the live working tree
        def candidate_read(path: str) -> "bytes | None":
            return _absent_or_bytes((ctx.root / path).read_bytes)

    change_set = build_change_set_record(start_dir=start_dir, scope=scope, base_ref=base_ref)
    changed = tuple(cp["path"] for cp in change_set["payload"]["changed_paths"])
    analysis = analyze_two_sided(roots, base_read, candidate_read, changed)

    payload = two_sided_to_payload(analysis)
    payload.update({"scope": scope, "base_oid": base_oid, "changed_path_count": len(changed)})
    provenance = {
        "tool": "cs_assure",
        "tool_version": KERNEL_VERSION,
        "record_kernel_schema_version": RECORD_SCHEMA_VERSION,
        "subcommand": "worker-reachability",
        "base_ref": base_ref,
        "base_oid": base_oid,
        "head_oid": ctx.head_oid,
        "is_shallow": ctx.is_shallow,
    }
    return seal_record(WORKER_REACHABILITY_RECORD_TYPE, WORKER_REACHABILITY_SCHEMA_VERSION, payload, provenance)

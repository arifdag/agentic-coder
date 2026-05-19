"""ProjectTest benchmark loader.

Dataset: 20 Python projects + 20 JavaScript projects (multi-file).
Source: https://github.com/YiboWANG214/ProjectTest
"""

import ast
import json
import logging
import posixpath
import re
from pathlib import Path
from typing import List, Optional, Set

from ..models import BenchmarkCase
from .utils import DEFAULT_DATA_DIR, ensure_repo

log = logging.getLogger(__name__)

REPO_URL = "https://github.com/YiboWANG214/ProjectTest.git"
LOCAL_DIR_NAME = "ProjectTest"


class ProjectTestDataset:
    """Loads the ProjectTest benchmark."""

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR, language_filter: Optional[str] = None):
        self._root = data_dir / LOCAL_DIR_NAME
        self._lang_filter = language_filter

    @property
    def name(self) -> str:
        return "projecttest"

    @property
    def language(self) -> Optional[str]:
        return self._lang_filter

    def download(self) -> Path:
        return ensure_repo(REPO_URL, self._root)

    def _detect_language(self, file_path: Path) -> str:
        ext = file_path.suffix.lower()
        if ext in (".js", ".jsx", ".ts", ".tsx"):
            return "javascript"
        return "python"

    # Canonical repo layout (from github.com/YiboWANG214/ProjectTest):
    #   ProjectTest/dataset/Python/<project>/*.py     -> 20 Python projects
    #   ProjectTest/dataset/JS/<project>/*.js         -> 20 JS projects
    #   ProjectTest/dataset/JAVA/<project>/*.java     -> 19 Java projects
    # Each project is a multi-file package; we concatenate source files with
    # header separators so the pipeline sees the whole module surface.
    _LANG_DIRS = (
        ("Python", "python", ("*.py",)),
        ("JS", "javascript", ("*.js",)),
        # JAVA is intentionally omitted: our pipeline has Python + JS runners
        # only. Reviewers can spin up a Java runner later; for now Java files
        # would fail every gate, which is noise.
    )

    # Projects that can't run in an isolated sandbox for structural reasons
    # that are not a failure of the pipeline under test. We exclude them so
    # that the aggregate metrics aren't skewed by environment issues.
    #
    #   - doudizhu: expects sibling ``./jsondata/*.txt,json`` data files at
    #     runtime; our sandbox copies source code only, not arbitrary data
    #     assets.
    #   - uno: imports ``rlcard`` (the parent ML library) at module level,
    #     which is ~50 MB with torch transitive deps and out of scope.
    _SKIP_PYTHON_PROJECTS = frozenset({"doudizhu", "uno"})

    @staticmethod
    def _top_level_names(text: str) -> Set[str]:
        """Collect top-level ``def``, ``async def``, ``class`` and simple
        assignment target names from a Python source string. Best-effort:
        anything that fails to parse yields an empty set.
        """
        names: Set[str] = set()
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return names
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
        return names

    @staticmethod
    def _module_level_reads(text: str) -> Set[str]:
        """Collect every Name that's *read* anywhere in the file.

        We deliberately over-approximate by walking the whole AST rather
        than just module-level statements. Rationale: some names that look
        like they're "inside a function body" actually run at import time
        (metaclass ``__prepare__`` and ``__init_subclass__``, decorator
        targets, default values, base-class metaclass keyword args, and so
        on). Missing any of those creates a NameError at benchmark load
        time, which is exactly what we're trying to prevent.

        The downside of over-approximation is that it can create dependency
        cycles between files that legitimately reference each other only
        inside function bodies. Those cycles are fine here: the topo sort
        falls back to alphabetical order for cycle members, which matches
        the original repo's intended load order.
        """
        reads: Set[str] = set()
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return reads

        for sub in ast.walk(tree):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                reads.add(sub.id)
            elif isinstance(sub, ast.Attribute):
                root = sub
                while isinstance(root, ast.Attribute):
                    root = root.value
                if isinstance(root, ast.Name):
                    reads.add(root.id)
        return reads

    @classmethod
    def _topo_sort_files(cls, files: list) -> list:
        """Deterministically topologically sort a list of
        ``(rel, stem, text, ns_needed, top_names)`` tuples so that if
        file A references a name defined in file B, B comes before A.

        On cycles we fall back to alphabetical order for the cycle members
        (the cycle will just generate runtime NameError for those files --
        which is exactly what happens in the real project too when run in
        the wrong order, so this matches the benchmark's intent).
        """
        # Map each top-level name to the file index that defines it.
        # We ALSO register each file's stem as a "name" it provides, so
        # that a reader like ``@utils.check_for_none`` creates a dependency
        # on ``utils.py`` (the file that populates the ``utils``
        # SimpleNamespace) even though ``utils`` isn't itself a top-level
        # name defined by any file.
        name_to_file = {}
        for idx, (_, stem, _, _, top_names) in enumerate(files):
            for nm in top_names:
                name_to_file.setdefault(nm, idx)
            if stem and stem != "__init__":
                name_to_file.setdefault(stem, idx)

        # Build adjacency: edge A -> B means A depends on B (B must come first).
        deps = {i: set() for i in range(len(files))}
        for idx, (_, _, text, _, _) in enumerate(files):
            reads = cls._module_level_reads(text)
            for nm in reads:
                owner = name_to_file.get(nm)
                if owner is not None and owner != idx:
                    deps[idx].add(owner)

        # Kahn's algorithm with alphabetical tie-break on (rel path).
        from collections import defaultdict, deque

        in_degree = defaultdict(int)
        rev = defaultdict(set)
        for a, bs in deps.items():
            for b in bs:
                rev[b].add(a)
                in_degree[a] += 1

        # Seed with all nodes having no dependencies, sorted alphabetically.
        def order_key(i):
            return files[i][0]  # relative path

        ready = sorted([i for i in range(len(files)) if in_degree[i] == 0], key=order_key)
        ready = deque(ready)
        out_idx: list = []
        seen: Set[int] = set()

        while ready:
            i = ready.popleft()
            if i in seen:
                continue
            seen.add(i)
            out_idx.append(i)
            for j in sorted(rev[i], key=order_key):
                in_degree[j] -= 1
                if in_degree[j] == 0:
                    ready.append(j)

        # Any remaining nodes (cycles) get appended in alphabetical order.
        remaining = [i for i in range(len(files)) if i not in seen]
        remaining.sort(key=order_key)
        out_idx.extend(remaining)
        return [files[i] for i in out_idx]

    @staticmethod
    def _intra_project_roots(project_dir: Path, patterns) -> Set[str]:
        """Collect names that refer to files/packages *inside* this project.

        When we concatenate the project into one ``source_module.py`` those
        intra-project imports become broken (the sub-module no longer exists
        as a separate file). We therefore treat any import whose root matches
        one of these names as "internal" and strip it.
        """
        roots: Set[str] = {project_dir.name}
        for pattern in patterns:
            for src in project_dir.rglob(pattern):
                if src.is_file():
                    roots.add(src.stem)  # foo.py -> foo
                    # nested packages: package dir names too
                    for parent in src.relative_to(project_dir).parents:
                        if parent.name:
                            roots.add(parent.name)
        return roots

    @classmethod
    def _rewrite_python_file(
        cls,
        text: str,
        intra_roots: Set[str],
        submodule_namespaces: Set[str],
    ) -> tuple:
        """Return ``(future_imports, body, namespace_names)``.

        When we flatten a multi-file Python project into one ``source_module``,
        intra-project imports no longer mean what they used to. We handle the
        three distinct cases:

        1.  ``from pkg.sub import X``  -> drop. X is already a top-level
            definition in the concatenated file.

        2.  ``from pkg.sub import X as Y``  -> replace with ``Y = X``.
            Otherwise callers that used the alias (``Y(...)``) fail with
            ``NameError: 'Y' is not defined``.

        3.  ``from pkg import submodule``  /  ``import pkg.submodule`` ->
            we synthesise a ``types.SimpleNamespace`` named ``submodule`` and
            let the builder populate it later with every top-level definition
            that came from ``submodule.py``. This preserves patterns like
            ``@utils.check_for_none`` and ``utils.full_process(...)``.

        - ``future_imports``: set of feature names from ``from __future__``
          statements (stripped here, hoisted to the top by the caller).
        - ``namespace_names``: names from case (3) above that the caller must
          back-fill after the concatenated source has been emitted.
        """
        future_features: Set[str] = set()
        namespace_names: Set[str] = set()

        try:
            tree = ast.parse(text)
        except SyntaxError:
            # Give up on AST; fall back to a regex-only best effort so the
            # case still loads (even if tests will likely fail on that file).
            body = re.sub(r"^\s*from\s+__future__\s+import\s+[^\n]*\n", "", text, flags=re.M)
            for root in intra_roots:
                body = re.sub(
                    rf"^\s*(?:from\s+{re.escape(root)}(?:\.[\w\.]+)?\s+import[^\n]*"
                    rf"|import\s+{re.escape(root)}(?:\.[\w\.]+)?[^\n]*)\n",
                    "",
                    body,
                    flags=re.M,
                )
            return future_features, body, namespace_names

        lines = text.splitlines(keepends=True)
        # Each entry: (start_line_inclusive, end_line_inclusive, replacement)
        rewrites: list = []

        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                for alias in node.names:
                    future_features.add(alias.name)
                rewrites.append((node.lineno, node.end_lineno or node.lineno, ""))
                continue

            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                root = module.split(".")[0]
                if not root or root not in intra_roots:
                    continue

                replacement_lines: list = []
                for alias in node.names:
                    name = alias.name  # what's being imported
                    asname = alias.asname  # alias in current scope, if any
                    if name == "*":
                        # `from pkg import *` is rare; drop it entirely since
                        # we've already concatenated everything.
                        continue
                    if name in submodule_namespaces:
                        # `from pkg import utils` where utils is a *file*.
                        # The builder creates a module-level SimpleNamespace
                        # for the canonical name at the top of the combined
                        # source, so we only need to emit an alias here if
                        # the caller bound it under a different name.
                        namespace_names.add(name)
                        bind = asname or name
                        if bind != name:
                            replacement_lines.append(f"{bind} = {name}")
                        continue
                    if asname and asname != name:
                        # `from pkg.mod import Foo as Bar` -> Bar = Foo
                        replacement_lines.append(f"{asname} = {name}")
                    # else: plain `from pkg.mod import Foo` -> drop.
                rewrites.append(
                    (
                        node.lineno,
                        node.end_lineno or node.lineno,
                        "\n".join(replacement_lines),
                    )
                )
                continue

            if isinstance(node, ast.Import):
                replacement_lines = []
                all_intra = True
                for alias in node.names:
                    name = alias.name
                    asname = alias.asname
                    root = name.split(".")[0]
                    if root not in intra_roots:
                        all_intra = False
                        continue
                    # `import pkg.mod` or `import pkg.mod as m`
                    parts = name.split(".")
                    bind = asname or parts[-1]
                    leaf = parts[-1]
                    if leaf in submodule_namespaces:
                        namespace_names.add(leaf)
                        if bind != leaf:
                            replacement_lines.append(f"{bind} = {leaf}")
                    # else: bare `import pkg` -> nothing to do
                if not all_intra and not replacement_lines:
                    continue
                # If ALL aliases are intra-project, replace the whole line.
                # (Mixed external+internal in one `import` is extremely rare.)
                if all_intra:
                    rewrites.append(
                        (
                            node.lineno,
                            node.end_lineno or node.lineno,
                            "\n".join(replacement_lines),
                        )
                    )

        if not rewrites:
            return future_features, text, namespace_names

        # Apply rewrites bottom-up so line numbers stay valid.
        out_lines = list(lines)
        for start, end, replacement in sorted(rewrites, key=lambda r: -r[0]):
            replacement_text = (replacement + "\n") if replacement else ""
            out_lines[start - 1 : end] = [replacement_text]
        return future_features, "".join(out_lines), namespace_names

    @staticmethod
    def _js_identifier(value: str) -> str:
        """Return a conservative JavaScript identifier derived from a file stem."""
        ident = re.sub(r"\W", "_", value.strip() or "default_export")
        if not re.match(r"^[A-Za-z_$]", ident):
            ident = f"_{ident}"
        return ident

    @staticmethod
    def _js_rel_no_ext(rel_path: str) -> str:
        return re.sub(r"\.(?:m?js|cjs|jsx|ts|tsx)$", "", rel_path.replace("\\", "/"))

    @classmethod
    def _js_relative_import_targets(cls, rel_path: str, text: str) -> Set[str]:
        """Collect normalized relative import targets for one JavaScript file."""
        targets: Set[str] = set()
        patterns = (
            r"""^\s*import\s+[^;\n]+?\s+from\s+['"](\.{1,2}/[^'"]+)['"]\s*;?\s*$""",
            r"""^\s*import\s+['"](\.{1,2}/[^'"]+)['"]\s*;?\s*$""",
        )
        rel_dir = posixpath.dirname(rel_path)
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.MULTILINE):
                spec = match.group(1)
                normalized = posixpath.normpath(posixpath.join(rel_dir, spec))
                targets.add(cls._js_rel_no_ext(normalized))
        return targets

    @classmethod
    def _topo_sort_js_files(cls, files: list) -> list:
        """Sort JavaScript files so local import providers appear first."""
        rel_to_idx = {cls._js_rel_no_ext(rel): idx for idx, (rel, _, _, _, _) in enumerate(files)}
        deps = {i: set() for i in range(len(files))}
        for idx, (rel, _, text, _, _) in enumerate(files):
            for target in cls._js_relative_import_targets(rel, text):
                owner = rel_to_idx.get(target)
                if owner is not None and owner != idx:
                    deps[idx].add(owner)

        from collections import defaultdict, deque

        in_degree = defaultdict(int)
        rev = defaultdict(set)
        for dependent, providers in deps.items():
            for provider in providers:
                rev[provider].add(dependent)
                in_degree[dependent] += 1

        def order_key(i):
            return files[i][0]

        ready = deque(sorted([i for i in range(len(files)) if in_degree[i] == 0], key=order_key))
        out_idx: list = []
        seen: Set[int] = set()
        while ready:
            i = ready.popleft()
            if i in seen:
                continue
            seen.add(i)
            out_idx.append(i)
            for j in sorted(rev[i], key=order_key):
                in_degree[j] -= 1
                if in_degree[j] == 0:
                    ready.append(j)

        remaining = [i for i in range(len(files)) if i not in seen]
        remaining.sort(key=order_key)
        out_idx.extend(remaining)
        return [files[i] for i in out_idx]

    @classmethod
    def _rewrite_javascript_file(cls, text: str, stem: str) -> tuple[str, Set[str]]:
        """Best-effort ESM-to-CommonJS flattening for ProjectTest JS cases.

        ProjectTest stores small multi-file ESM snippets. The existing JS sandbox
        writes a single ``source_module.js`` and the generated Jest tests require
        it via CommonJS, so leaving ESM ``import``/``export`` syntax intact makes
        every JS ProjectTest case fail before tests run. This is intentionally
        conservative; it only rewrites local imports and common export forms.
        """
        exports: Set[str] = set()
        default_name = cls._js_identifier(stem)

        # Local imports are satisfied by concatenating provider files first.
        text = re.sub(
            r"""^\s*import\s+[^;\n]+?\s+from\s+['"]\.{1,2}/[^'"]+['"]\s*;?\s*$""",
            "",
            text,
            flags=re.MULTILINE,
        )
        text = re.sub(
            r"""^\s*import\s+['"]\.{1,2}/[^'"]+['"]\s*;?\s*$""",
            "",
            text,
            flags=re.MULTILINE,
        )

        def export_decl(match: re.Match) -> str:
            exports.add(match.group(2))
            return f"{match.group(1)} {match.group(2)}"

        text = re.sub(
            r"\bexport\s+(const|let|var|function|class)\s+([A-Za-z_$][\w$]*)",
            export_decl,
            text,
        )

        def default_named_decl(match: re.Match) -> str:
            kind = match.group(1)
            name = match.group(2)
            exports.add(name)
            return f"{kind} {name}"

        text = re.sub(
            r"\bexport\s+default\s+(function|class)\s+([A-Za-z_$][\w$]*)",
            default_named_decl,
            text,
        )

        def default_anon_decl(match: re.Match) -> str:
            kind = match.group(1)
            exports.add(default_name)
            return f"{kind} {default_name}"

        text = re.sub(r"\bexport\s+default\s+(function|class)\s*(?=[({])", default_anon_decl, text)

        def default_object(match: re.Match) -> str:
            exports.add(default_name)
            return f"const {default_name} = {{"

        text = re.sub(r"\bexport\s+default\s*\{", default_object, text)

        def default_expr(match: re.Match) -> str:
            expr = match.group(1).strip()
            if re.match(r"^[A-Za-z_$][\w$]*$", expr):
                exports.add(expr)
                if expr != default_name:
                    exports.add(default_name)
                    return f"const {default_name} = {expr};"
                return ""
            exports.add(default_name)
            return f"const {default_name} = {expr};"

        text = re.sub(
            r"^\s*export\s+default\s+([^;\n]+)\s*;?\s*$",
            default_expr,
            text,
            flags=re.MULTILINE,
        )

        def export_block(match: re.Match) -> str:
            for raw in match.group(1).replace("\n", " ").split(","):
                item = raw.strip()
                if not item:
                    continue
                source_name = re.split(r"\s+as\s+", item)[0].strip()
                if re.match(r"^[A-Za-z_$][\w$]*$", source_name):
                    exports.add(source_name)
            return ""

        text = re.sub(
            r"^\s*export\s*\{([^}]+)\}\s*;?\s*$",
            export_block,
            text,
            flags=re.MULTILINE,
        )
        return text, exports

    @staticmethod
    def _import_module_from_target(target_file: str) -> Optional[str]:
        """Convert a project-root-relative Python file path to a module path."""
        if not target_file.endswith(".py"):
            return None
        module = target_file[:-3].replace("\\", "/")
        if module.endswith("/__init__"):
            module = module[: -len("/__init__")]
        module = ".".join(part for part in module.split("/") if part)
        return module or None

    def load(self) -> List[BenchmarkCase]:
        self.download()
        cases: List[BenchmarkCase] = []

        dataset_dir = self._root / "dataset"
        if not dataset_dir.exists():
            # Older release may have dropped files at repo root or under data/
            for candidate in ("data", ""):
                c = self._root / candidate if candidate else self._root
                if c.exists() and any((c / sub).exists() for sub, _, _ in self._LANG_DIRS):
                    dataset_dir = c
                    break

        for lang_sub, lang_name, patterns in self._LANG_DIRS:
            if self._lang_filter and lang_name != self._lang_filter:
                continue
            base = dataset_dir / lang_sub
            if not base.exists():
                continue

            for project_dir in sorted(base.iterdir()):
                if not project_dir.is_dir():
                    continue
                if lang_name == "python" and project_dir.name in self._SKIP_PYTHON_PROJECTS:
                    log.info("Skipping unsupported project: %s", project_dir.name)
                    continue

                if lang_name == "python":
                    intra_roots = self._intra_project_roots(project_dir, patterns)
                    # Set of file stems (e.g. {"utils", "dealer", "game"}) —
                    # these are the names that may be referenced as *submodule
                    # namespaces* (`from pkg import utils`, `utils.foo()`).
                    submodule_stems = {
                        src.stem
                        for pattern in patterns
                        for src in project_dir.rglob(pattern)
                        if src.is_file() and src.stem != "__init__"
                    }
                else:
                    intra_roots = set()
                    submodule_stems = set()

                # Pass 1: rewrite every file individually, collecting both the
                # rewritten text AND the set of submodule namespaces each file
                # still depends on (e.g. `@utils.check_for_none`). We need to
                # know the union before we can lay out the combined module in
                # the right order.
                rewritten_files: list = []  # (rel_path, stem, text, ns_needed, top_names)
                all_future: Set[str] = set()

                for pattern in patterns:
                    for src in sorted(project_dir.rglob(pattern)):
                        if src.name.startswith("test_") or src.name.endswith("_test.py"):
                            continue
                        if "node_modules" in str(src):
                            continue
                        try:
                            text = src.read_text(encoding="utf-8", errors="replace")
                        except Exception:
                            continue

                        if lang_name == "python":
                            futures, text, ns_needed = self._rewrite_python_file(
                                text,
                                intra_roots,
                                submodule_stems,
                            )
                            all_future.update(futures)
                            top_names = self._top_level_names(text)
                        else:
                            ns_needed = set()
                            top_names = set()

                        rewritten_files.append(
                            (
                                src.relative_to(project_dir).as_posix(),
                                src.stem,
                                text,
                                ns_needed,
                                top_names,
                            )
                        )

                if not rewritten_files:
                    continue

                # Pass 2: lay out the combined module.
                #
                # Decorators and module-level references like
                # ``@utils.check_for_none`` execute at import time, so for a
                # user file to be importable, the namespace object ``utils``
                # and all its attributes must already be populated BEFORE that
                # file's body runs. We therefore:
                #   1. Create every referenced SimpleNamespace up front.
                #   2. Emit provider files (those whose stem is referenced as
                #      a namespace) FIRST, each followed by inline bindings
                #      (``utils.foo = foo``) so downstream files see a fully
                #      populated namespace.
                #   3. Emit all other files after.
                if lang_name == "python":
                    used_namespaces: Set[str] = set()
                    for _, _, _, ns_needed, _ in rewritten_files:
                        used_namespaces.update(ns_needed)

                    parts: list = []
                    # Always inject PEP 563 lazy annotations. This makes every
                    # type hint evaluated as a *string* rather than a real
                    # expression at class-definition time, so forward
                    # references to names that are defined later in the
                    # concatenated module (e.g. ``def f(c: BridgeCard)``
                    # where ``BridgeCard`` lives in a file that comes later
                    # in the alphabet) no longer NameError at import time.
                    all_future.add("annotations")
                    parts.append("from __future__ import " + ", ".join(sorted(all_future)))
                    if used_namespaces:
                        parts.append("import types as _types")
                        for ns in sorted(used_namespaces):
                            parts.append(f"{ns} = _types.SimpleNamespace()")

                    # __init__.py always runs LAST (it typically only holds
                    # re-export aliases like ``Dealer = BlackjackDealer``
                    # that can only resolve after the target classes are
                    # defined). Everything else goes through topological
                    # sort driven by actual module-level name usage.
                    init_files = [f for f in rewritten_files if f[1] == "__init__"]
                    body_files = [f for f in rewritten_files if f[1] != "__init__"]
                    body_files = self._topo_sort_files(body_files)

                    for rel, stem, text, _ns, top_names in body_files + init_files:
                        block = f"# --- {rel} ---\n{text}"
                        if stem in used_namespaces and top_names:
                            # This file is referenced as a namespace from
                            # somewhere else (``from pkg import utils``).
                            # Back-fill the synthesised SimpleNamespace with
                            # every top-level name this file defines so
                            # later files can use ``utils.check_for_none``.
                            binding_lines = [f"{stem}.{nm} = {nm}" for nm in sorted(top_names)]
                            block = (
                                block
                                + ("\n" if not block.endswith("\n") else "")
                                + "\n".join(binding_lines)
                                + "\n"
                            )
                        parts.append(block)
                    combined = "\n\n".join(parts)
                else:
                    js_exports: Set[str] = set()
                    js_parts: list = []
                    js_files = self._topo_sort_js_files(rewritten_files)
                    for rel, stem, text, _, _ in js_files:
                        rewritten_js, exports = self._rewrite_javascript_file(text, stem)
                        js_exports.update(exports)
                        js_parts.append(f"// --- {rel} ---\n{rewritten_js}")
                    valid_exports = sorted(
                        name for name in js_exports if re.match(r"^[A-Za-z_$][\w$]*$", name)
                    )
                    if valid_exports:
                        js_parts.append(
                            "module.exports = {\n"
                            + ",\n".join(f"  {name}" for name in valid_exports)
                            + "\n};"
                        )
                    combined = "\n\n".join(js_parts)

                # Build metadata for repo-context execution (Python only)
                metadata: dict = {"project": project_dir.name, "file_count": len(rewritten_files)}
                if lang_name == "python":
                    metadata["execution_context"] = "repo"
                    is_package_project = (project_dir / "__init__.py").exists()
                    repo_root = project_dir.parent if is_package_project else project_dir
                    package_rel = (
                        project_dir.relative_to(repo_root).as_posix() if is_package_project else ""
                    )
                    metadata["project_root"] = str(repo_root.resolve())
                    metadata["package_project_root"] = str(project_dir.resolve())
                    metadata["pythonpath_entries"] = [package_rel] if package_rel else []
                    if package_rel:
                        metadata["package_root"] = package_rel
                    metadata["project_name"] = project_dir.name
                    # Best-effort target file: prefer a file matching the project name, else first .py
                    target_candidates = [
                        rel
                        for rel, stem, _, _, _ in body_files + init_files
                        if stem == project_dir.name
                    ]
                    if not target_candidates:
                        target_candidates = [rel for rel, _, _, _, _ in body_files + init_files]
                    if target_candidates:
                        target_file = (
                            f"{package_rel}/{target_candidates[0]}"
                            if package_rel
                            else target_candidates[0]
                        )
                        metadata["target_file"] = target_file
                        import_module = self._import_module_from_target(target_file)
                        if import_module:
                            metadata["import_module"] = import_module
                    dep_files = (
                        list(project_dir.rglob("requirements*.txt"))
                        + list(project_dir.rglob("setup.py"))
                        + list(project_dir.rglob("pyproject.toml"))
                    )
                    metadata["dependency_files"] = [
                        str(f.relative_to(repo_root).as_posix()) for f in dep_files
                    ]
                    metadata["flattened_source_available"] = True

                cases.append(
                    BenchmarkCase(
                        id=f"pt-{lang_name}-{project_dir.name}",
                        code=combined,
                        language=lang_name,
                        metadata=metadata,
                        user_request="Generate comprehensive unit tests",
                    )
                )

        # Fallback: scan root for JSONL manifests (older releases)
        if not cases:
            for jsonl in sorted(self._root.rglob("*.jsonl")):
                for i, line in enumerate(jsonl.read_text(encoding="utf-8").splitlines()):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    code = entry.get("code") or entry.get("source", "")
                    if not code:
                        continue
                    lang = entry.get("language", "python").lower()
                    cases.append(
                        BenchmarkCase(
                            id=f"pt-{i}",
                            code=code,
                            language=lang,
                            metadata=entry,
                            user_request="Generate comprehensive unit tests",
                        )
                    )

        log.info("Loaded %d cases from ProjectTest benchmark", len(cases))
        return cases

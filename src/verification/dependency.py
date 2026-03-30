"""Gate 2: Dependency validation against PyPI and npm registries."""

import ast
import re
import sys
from typing import List, Set, Optional

import httpx

from .models import GateResult, Finding, Severity

KNOWN_TEST_PACKAGES = {
    "pytest", "pytest_cov", "coverage", "unittest", "mock",
    "pytest_timeout", "pytest_mock", "hypothesis", "faker",
}

KNOWN_JS_TEST_PACKAGES = {
    "jest", "mocha", "chai", "sinon", "supertest", "ava",
    "jasmine", "tap", "tape", "vitest", "cypress",
    "@jest/globals", "@testing-library/jest-dom",
    "@testing-library/react", "@testing-library/dom",
}

NODE_BUILTINS = {
    "assert", "async_hooks", "buffer", "child_process", "cluster",
    "console", "constants", "crypto", "dgram", "diagnostics_channel",
    "dns", "domain", "events", "fs", "http", "http2", "https",
    "inspector", "module", "net", "os", "path", "perf_hooks",
    "process", "punycode", "querystring", "readline", "repl",
    "stream", "string_decoder", "sys", "timers", "tls", "trace_events",
    "tty", "url", "util", "v8", "vm", "wasi", "worker_threads", "zlib",
    "node:assert", "node:buffer", "node:child_process", "node:crypto",
    "node:events", "node:fs", "node:http", "node:https", "node:net",
    "node:os", "node:path", "node:process", "node:querystring",
    "node:readline", "node:stream", "node:timers", "node:tls",
    "node:url", "node:util", "node:vm", "node:worker_threads", "node:zlib",
}

IMPORT_TO_PYPI = {
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "attr": "attrs",
    "bs4": "beautifulsoup4",
    "dateutil": "python-dateutil",
    "gi": "PyGObject",
    "wx": "wxPython",
    "serial": "pyserial",
    "usb": "pyusb",
    "Crypto": "pycryptodome",
}


def _get_stdlib_modules() -> Set[str]:
    """Get the set of standard library module names."""
    if hasattr(sys, "stdlib_module_names"):
        return set(sys.stdlib_module_names)
    return {
        "abc", "aifc", "argparse", "array", "ast", "asyncio", "atexit",
        "base64", "binascii", "bisect", "builtins", "bz2", "calendar",
        "cgi", "cgitb", "chunk", "cmath", "cmd", "code", "codecs",
        "collections", "colorsys", "compileall", "concurrent",
        "configparser", "contextlib", "contextvars", "copy", "copyreg",
        "cProfile", "csv", "ctypes", "curses", "dataclasses", "datetime",
        "dbm", "decimal", "difflib", "dis", "distutils", "doctest",
        "email", "encodings", "enum", "errno", "faulthandler", "fcntl",
        "filecmp", "fileinput", "fnmatch", "fractions", "ftplib",
        "functools", "gc", "getopt", "getpass", "gettext", "glob",
        "grp", "gzip", "hashlib", "heapq", "hmac", "html", "http",
        "idlelib", "imaplib", "imghdr", "imp", "importlib", "inspect",
        "io", "ipaddress", "itertools", "json", "keyword", "lib2to3",
        "linecache", "locale", "logging", "lzma", "mailbox", "mailcap",
        "marshal", "math", "mimetypes", "mmap", "modulefinder",
        "multiprocessing", "netrc", "nis", "nntplib", "numbers",
        "operator", "optparse", "os", "ossaudiodev", "pathlib",
        "pdb", "pickle", "pickletools", "pipes", "pkgutil", "platform",
        "plistlib", "poplib", "posix", "posixpath", "pprint",
        "profile", "pstats", "pty", "pwd", "py_compile",
        "pyclbr", "pydoc", "queue", "quopri", "random", "re",
        "readline", "reprlib", "resource", "rlcompleter", "runpy",
        "sched", "secrets", "select", "selectors", "shelve",
        "shlex", "shutil", "signal", "site", "smtpd", "smtplib",
        "sndhdr", "socket", "socketserver", "sqlite3", "ssl",
        "stat", "statistics", "string", "stringprep", "struct",
        "subprocess", "sunau", "symtable", "sys", "sysconfig",
        "syslog", "tabnanny", "tarfile", "telnetlib", "tempfile",
        "termios", "test", "textwrap", "threading", "time",
        "timeit", "tkinter", "token", "tokenize", "tomllib", "trace",
        "traceback", "tracemalloc", "tty", "turtle", "turtledemo",
        "types", "typing", "unicodedata", "unittest", "urllib",
        "uu", "uuid", "venv", "warnings", "wave", "weakref",
        "webbrowser", "winreg", "winsound", "wsgiref", "xdrlib",
        "xml", "xmlrpc", "zipapp", "zipfile", "zipimport", "zlib",
        "_thread", "__future__",
    }


STDLIB_MODULES = _get_stdlib_modules()


def extract_imports(code: str) -> Set[str]:
    """Extract top-level package names from Python source using AST.

    Args:
        code: Python source code

    Returns:
        Set of top-level package names (e.g. ``requests``, ``numpy``)
    """
    packages: Set[str] = set()

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return packages

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                packages.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                packages.add(node.module.split(".")[0])

    return packages


def extract_js_imports(code: str) -> Set[str]:
    """Extract package names from JavaScript/TypeScript source using regex.

    Handles ``require('pkg')`` and ``import ... from 'pkg'`` forms.
    Skips relative paths (``./`` or ``../``).
    """
    packages: Set[str] = set()

    # require('pkg') or require("pkg")
    for m in re.finditer(r"""require\s*\(\s*['"]([^'"]+)['"]\s*\)""", code):
        pkg = m.group(1)
        if not pkg.startswith("."):
            packages.add(pkg.split("/")[0] if not pkg.startswith("@") else "/".join(pkg.split("/")[:2]))

    # import ... from 'pkg'
    for m in re.finditer(r"""(?:from|import\s+.*?\s+from)\s+['"]([^'"]+)['"]""", code):
        pkg = m.group(1)
        if not pkg.startswith("."):
            packages.add(pkg.split("/")[0] if not pkg.startswith("@") else "/".join(pkg.split("/")[:2]))

    return packages


class DependencyValidator:
    """Validate that imported packages exist on PyPI or npm."""

    def __init__(
        self,
        pypi_timeout: int = 10,
        extra_known: Optional[Set[str]] = None,
    ):
        self.pypi_timeout = pypi_timeout
        self.extra_known = extra_known or set()

    def _is_known_safe(self, package: str) -> bool:
        """Check if a Python package is known-safe (stdlib, test infra, etc.)."""
        if package in STDLIB_MODULES:
            return True
        if package in KNOWN_TEST_PACKAGES:
            return True
        if package in self.extra_known:
            return True
        if package == "source_module":
            return True
        return False

    def _is_known_safe_js(self, package: str) -> bool:
        """Check if a JS package is known-safe (builtins, test infra, etc.)."""
        if package in NODE_BUILTINS:
            return True
        bare = package.replace("node:", "")
        if bare in NODE_BUILTINS:
            return True
        if package in KNOWN_JS_TEST_PACKAGES:
            return True
        if package in self.extra_known:
            return True
        if package in ("source_module", "./source_module"):
            return True
        return False

    def _check_pypi(self, package: str) -> bool:
        """Check if a package exists on PyPI."""
        pypi_name = IMPORT_TO_PYPI.get(package, package)

        try:
            resp = httpx.get(
                f"https://pypi.org/pypi/{pypi_name}/json",
                timeout=self.pypi_timeout,
                follow_redirects=True,
            )
            return resp.status_code == 200
        except httpx.HTTPError:
            return True

    def _check_npm(self, package: str) -> bool:
        """Check if a package exists on npm."""
        try:
            resp = httpx.get(
                f"https://registry.npmjs.org/{package}",
                timeout=self.pypi_timeout,
                follow_redirects=True,
            )
            return resp.status_code == 200
        except httpx.HTTPError:
            return True

    def validate(self, code: str, language: Optional[str] = None) -> GateResult:
        """Validate all imports in the given code.

        Args:
            code: Source code to check
            language: Language hint (``python``, ``javascript``, ``typescript``)

        Returns:
            GateResult indicating pass/fail and any phantom packages
        """
        if language in ("javascript", "typescript"):
            return self._validate_js(code)
        return self._validate_python(code)

    def _validate_python(self, code: str) -> GateResult:
        imports = extract_imports(code)
        findings: List[Finding] = []

        third_party = {
            pkg for pkg in imports
            if not self._is_known_safe(pkg)
        }

        for pkg in sorted(third_party):
            exists = self._check_pypi(pkg)
            if not exists:
                findings.append(Finding(
                    severity=Severity.ERROR,
                    code="PHANTOM-PKG",
                    message=(
                        f"Package '{pkg}' not found on PyPI. "
                        f"This may be a hallucinated dependency."
                    ),
                    suggestion=f"Remove or replace the import of '{pkg}'.",
                ))

        passed = not any(f.severity == Severity.ERROR for f in findings)

        return GateResult(
            gate_name="dependency",
            passed=passed,
            findings=findings,
        )

    def _validate_js(self, code: str) -> GateResult:
        imports = extract_js_imports(code)
        findings: List[Finding] = []

        third_party = {
            pkg for pkg in imports
            if not self._is_known_safe_js(pkg)
        }

        for pkg in sorted(third_party):
            exists = self._check_npm(pkg)
            if not exists:
                findings.append(Finding(
                    severity=Severity.ERROR,
                    code="PHANTOM-PKG",
                    message=(
                        f"Package '{pkg}' not found on npm. "
                        f"This may be a hallucinated dependency."
                    ),
                    suggestion=f"Remove or replace the import of '{pkg}'.",
                ))

        passed = not any(f.severity == Severity.ERROR for f in findings)

        return GateResult(
            gate_name="dependency",
            passed=passed,
            findings=findings,
        )

"""Gate 2: Dependency validation against PyPI registry."""

import ast
import sys
from typing import List, Set, Optional

import httpx

from .models import GateResult, Finding, Severity

KNOWN_TEST_PACKAGES = {
    "pytest", "pytest_cov", "coverage", "unittest", "mock",
    "pytest_timeout", "pytest_mock", "hypothesis", "faker",
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


class DependencyValidator:
    """Validate that imported packages exist on PyPI."""

    def __init__(
        self,
        pypi_timeout: int = 10,
        extra_known: Optional[Set[str]] = None,
    ):
        self.pypi_timeout = pypi_timeout
        self.extra_known = extra_known or set()

    def _is_known_safe(self, package: str) -> bool:
        """Check if a package is known-safe (stdlib, test infra, etc.)."""
        if package in STDLIB_MODULES:
            return True
        if package in KNOWN_TEST_PACKAGES:
            return True
        if package in self.extra_known:
            return True
        if package == "source_module":
            return True
        return False

    def _check_pypi(self, package: str) -> bool:
        """Check if a package exists on PyPI.

        Args:
            package: Package name

        Returns:
            True if the package exists
        """
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

    def validate(self, code: str) -> GateResult:
        """Validate all imports in the given code.

        Args:
            code: Python source to check

        Returns:
            GateResult indicating pass/fail and any phantom packages
        """
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

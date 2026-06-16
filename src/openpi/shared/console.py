"""Tiny ANSI-colored string helpers for log messages.

These functions only *return* a (colored) string -- they do not print -- so they
compose with ``logging`` calls, e.g. ``logging.info(console.ok("done"))``.

This is a lightweight stand-in for the helper module referenced by the value
training / labeling scripts (``from openpi.shared import console``).
"""

from __future__ import annotations

_RESET = "\033[0m"


def _wrap(msg: object, code: str) -> str:
    return f"{code}{msg}{_RESET}"


def info(msg: object) -> str:
    """Cyan informational message."""
    return _wrap(msg, "\033[36m")


def ok(msg: object) -> str:
    """Green success message."""
    return _wrap(msg, "\033[1;32m")


def warn(msg: object) -> str:
    """Yellow warning message."""
    return _wrap(msg, "\033[1;33m")


def error(msg: object) -> str:
    """Red error message."""
    return _wrap(msg, "\033[1;31m")


def debug(msg: object) -> str:
    """Dim debug message."""
    return _wrap(msg, "\033[2m")

"""Small helpers around tqdm progress bars used by the value-function scripts.

Lightweight stand-in for the helper module referenced as
``from openpi.shared import progress``.

Two usage styles are supported:

* Pass an explicit tqdm bar, e.g. ``progress.sync_pbar_color(pbar)``.
* Operate on a module-global "current" bar (set implicitly by the first call
  that receives a bar), e.g. ``progress.set_postfix_str("...")``.

Every helper is defensive: if no bar is available, or the underlying tqdm call
fails, it silently no-ops so progress reporting never breaks the actual work.
"""

from __future__ import annotations

_DEFAULT_COLOUR = "cyan"
_current = None


def _resolve(pbar):
    global _current
    if pbar is not None:
        _current = pbar
    return _current


def set_current(pbar) -> None:
    """Register ``pbar`` as the module-global current bar."""
    global _current
    _current = pbar


def sync_pbar_color(pbar=None, colour: str = _DEFAULT_COLOUR) -> None:
    """Keep the bar's colour consistent (tqdm resets it after ``write``)."""
    bar = _resolve(pbar)
    if bar is None:
        return
    try:
        bar.colour = colour
    except Exception:
        pass


def set_postfix(*args, pbar=None, **kwargs) -> None:
    bar = _resolve(pbar)
    if bar is None:
        return
    try:
        bar.set_postfix(*args, **kwargs)
    except Exception:
        pass


def set_postfix_str(s: str = "", pbar=None, refresh: bool = True) -> None:
    bar = _resolve(pbar)
    if bar is None:
        return
    try:
        bar.set_postfix_str(s, refresh=refresh)
    except Exception:
        pass


def update(n: int = 1, pbar=None) -> None:
    bar = _resolve(pbar)
    if bar is None:
        return
    try:
        bar.update(n)
    except Exception:
        pass


def close(pbar=None) -> None:
    global _current
    bar = _resolve(pbar)
    if bar is None:
        return
    try:
        bar.close()
    except Exception:
        pass
    finally:
        _current = None

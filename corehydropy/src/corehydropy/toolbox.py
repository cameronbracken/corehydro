"""The Numerics toolbox surface. Every verb serializes its options to the
``toolbox_runner.hpp`` grammar and runs one method through ``_core.toolbox_run``; bulk data goes
across as numeric vectors, not JSON. Mirrors ``corehydror``'s ``R/toolbox.R`` verb for verb.
"""

from __future__ import annotations

import json

import numpy as np

from . import _core

__all__ = ["correlation"]


def _toolbox_run(group: str, method: str, data=None, options=None) -> dict:
    """Internal: one call into the shared runner."""
    vectors = [np.asarray(d, dtype=float).ravel().tolist() for d in (data or [])]
    return _core.toolbox_run(group, method, vectors, json.dumps(options or {}))


def _check_pair(x, y, x_name: str = "x", y_name: str = "y"):
    """Internal: reject the two mistakes every paired-series verb can make, naming the argument."""
    xa = np.asarray(x, dtype=float).ravel()
    ya = np.asarray(y, dtype=float).ravel()
    if xa.size != ya.size:
        raise ValueError(
            f"`{x_name}` and `{y_name}` must have the same length; got {xa.size} and {ya.size}"
        )
    if xa.size < 2:
        raise ValueError(f"`{x_name}` and `{y_name}` must have at least two elements")
    return xa, ya


def correlation(x, y, method: str = "pearson") -> float:
    """Correlation between two samples.

    Mirrors the C# ``Correlation`` class of the Numerics library. Upstream's matrix overloads
    are not ported, so only the paired-vector forms are available here.

    Parameters
    ----------
    x, y : array_like
        Numeric vectors of equal length, at least two elements.
    method : {"pearson", "spearman", "kendall"}
        Which coefficient to compute.

    Returns
    -------
    float

    Examples
    --------
    >>> from corehydropy import correlation
    >>> round(correlation([14, 8, 32, 7, 3, 15], [10, 5, 7, 4, 3, 8]), 6)
    0.545027
    """
    if method not in ("pearson", "spearman", "kendall"):
        raise ValueError(f"`method` must be one of 'pearson', 'spearman', 'kendall'; got {method!r}")
    xa, ya = _check_pair(x, y)
    return float(_toolbox_run("correlation", method, [xa, ya])["values"][0])

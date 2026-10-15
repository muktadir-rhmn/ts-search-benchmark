"""The two-dimensional Müller–Brown model potential."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


# Standard Müller–Brown parameters.
_A = np.array([-200.0, -100.0, -170.0, 15.0])
_a = np.array([-1.0, -1.0, -6.5, 0.7])
_b = np.array([0.0, 0.0, 11.0, 0.6])
_c = np.array([-10.0, -10.0, -6.5, 0.7])
_x0 = np.array([1.0, 0.0, -0.5, -1.0])
_y0 = np.array([0.0, 0.5, 1.5, 1.0])

DEFAULT_REACTANT = np.array([-0.55822363, 1.44172584])
DEFAULT_PRODUCT = np.array([0.62349940, 0.02803776])
PLOT_BOUNDS = ((-1.5, 1.2), (-0.2, 2.0))


def muller_brown(point: ArrayLike) -> float | NDArray[np.float64]:
    """Return the Müller–Brown energy at one point or an array of points.

    ``point`` must have final dimension two. Thus ``[x, y]`` returns a float,
    while an array with shape ``(..., 2)`` returns an array with shape ``...``.
    """

    p = np.asarray(point, dtype=float)
    if p.shape[-1] != 2:
        raise ValueError("Müller–Brown coordinates must have final dimension 2")
    dx = p[..., 0, None] - _x0
    dy = p[..., 1, None] - _y0
    exponent = _a * dx**2 + _b * dx * dy + _c * dy**2
    value = np.sum(_A * np.exp(np.clip(exponent, -700.0, 100.0)), axis=-1)
    return float(value) if value.ndim == 0 else value


def muller_brown_gradient(point: ArrayLike) -> NDArray[np.float64]:
    """Return the analytic gradient, with the same leading shape as ``point``."""

    p = np.asarray(point, dtype=float)
    if p.shape[-1] != 2:
        raise ValueError("Müller–Brown coordinates must have final dimension 2")
    dx = p[..., 0, None] - _x0
    dy = p[..., 1, None] - _y0
    exponent = _a * dx**2 + _b * dx * dy + _c * dy**2
    terms = _A * np.exp(np.clip(exponent, -700.0, 100.0))
    gx = np.sum(terms * (2.0 * _a * dx + _b * dy), axis=-1)
    gy = np.sum(terms * (_b * dx + 2.0 * _c * dy), axis=-1)
    return np.stack((gx, gy), axis=-1)


# Short aliases make the module compatible with the generic string solvers.
energy = muller_brown
gradient = muller_brown_gradient


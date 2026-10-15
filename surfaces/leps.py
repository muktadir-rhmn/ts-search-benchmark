"""London–Eyring–Polanyi–Sato (LEPS) potential for a triatomic reaction.

The general function accepts the three internuclear distances.  The collinear
function uses coordinates ``(r_AB, r_BC)`` and imposes ``r_AC = r_AB + r_BC``.
Energies are in eV and distances are in Å for the default H + H2 parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class LEPSParameters:
    """Bond parameters ordered as AB, BC, AC."""

    dissociation_energy: tuple[float, float, float] = (4.746, 4.746, 4.746)
    equilibrium_distance: tuple[float, float, float] = (0.7414, 0.7414, 0.7414)
    morse_alpha: tuple[float, float, float] = (1.942, 1.942, 1.942)
    sato: tuple[float, float, float] = (0.15, 0.15, 0.15)


DEFAULT_PARAMETERS = LEPSParameters()
DEFAULT_REACTANT = np.array([0.7414, 3.0])
DEFAULT_PRODUCT = np.array([3.0, 0.7414])
PLOT_BOUNDS = ((0.45, 3.2), (0.45, 3.2))


def _q_j_and_derivatives(
    distances: NDArray[np.float64], parameters: LEPSParameters
) -> tuple[NDArray[np.float64], ...]:
    d = np.asarray(parameters.dissociation_energy)
    r0 = np.asarray(parameters.equilibrium_distance)
    alpha = np.asarray(parameters.morse_alpha)
    delta = np.asarray(parameters.sato)
    z = np.exp(np.clip(-alpha * (distances - r0), -100.0, 100.0))
    # The factor of four recovers a Morse well of depth D in the separated
    # diatomic limit (for delta=0, Q=D(3z^2-2z)/4 and J=D(z^2-6z)/4).
    prefactor = d / (4.0 * (1.0 + delta))

    q = prefactor * ((3.0 + delta) * z**2 - (2.0 + 6.0 * delta) * z)
    j = prefactor * ((1.0 + 3.0 * delta) * z**2 - (6.0 + 2.0 * delta) * z)
    dq = prefactor * alpha * (
        -2.0 * (3.0 + delta) * z**2 + (2.0 + 6.0 * delta) * z
    )
    dj = prefactor * alpha * (
        -2.0 * (1.0 + 3.0 * delta) * z**2 + (6.0 + 2.0 * delta) * z
    )
    return q, j, dq, dj


def leps_potential(
    distances: ArrayLike, parameters: LEPSParameters = DEFAULT_PARAMETERS
) -> float | NDArray[np.float64]:
    """Return the LEPS energy for ``(..., 3)`` distances ``(AB, BC, AC)``."""

    r = np.asarray(distances, dtype=float)
    if r.shape[-1] != 3:
        raise ValueError("LEPS distances must have final dimension 3: (AB, BC, AC)")
    q, j, _, _ = _q_j_and_derivatives(r, parameters)
    radicand = (
        np.sum(j**2, axis=-1)
        - j[..., 0] * j[..., 1]
        - j[..., 1] * j[..., 2]
        - j[..., 0] * j[..., 2]
    )
    value = np.sum(q, axis=-1) - np.sqrt(np.maximum(radicand, 1.0e-30))
    return float(value) if value.ndim == 0 else value


def leps_gradient(
    distances: ArrayLike, parameters: LEPSParameters = DEFAULT_PARAMETERS
) -> NDArray[np.float64]:
    """Return derivatives with respect to the three independent distances."""

    r = np.asarray(distances, dtype=float)
    if r.shape[-1] != 3:
        raise ValueError("LEPS distances must have final dimension 3: (AB, BC, AC)")
    _, j, dq, dj = _q_j_and_derivatives(r, parameters)
    radicand = (
        np.sum(j**2, axis=-1)
        - j[..., 0] * j[..., 1]
        - j[..., 1] * j[..., 2]
        - j[..., 0] * j[..., 2]
    )
    root = np.sqrt(np.maximum(radicand, 1.0e-30))
    sum_j = np.sum(j, axis=-1, keepdims=True)
    # d/dJ_i [sum(J_i^2) - sum_{i<j}(J_i J_j)] = 3 J_i - sum(J)
    dradicand = dj * (3.0 * j - sum_j)
    return dq - 0.5 * dradicand / root[..., None]


def leps_collinear(
    point: ArrayLike, parameters: LEPSParameters = DEFAULT_PARAMETERS
) -> float | NDArray[np.float64]:
    """Return the collinear A–B–C energy at ``(..., 2)`` = ``(r_AB, r_BC)``."""

    p = np.asarray(point, dtype=float)
    if p.shape[-1] != 2:
        raise ValueError("Collinear LEPS coordinates must have final dimension 2")
    distances = np.concatenate((p, np.sum(p, axis=-1, keepdims=True)), axis=-1)
    return leps_potential(distances, parameters)


def leps_collinear_gradient(
    point: ArrayLike, parameters: LEPSParameters = DEFAULT_PARAMETERS
) -> NDArray[np.float64]:
    """Return the collinear gradient, including ``r_AC = r_AB + r_BC``."""

    p = np.asarray(point, dtype=float)
    if p.shape[-1] != 2:
        raise ValueError("Collinear LEPS coordinates must have final dimension 2")
    distances = np.concatenate((p, np.sum(p, axis=-1, keepdims=True)), axis=-1)
    g = leps_gradient(distances, parameters)
    return np.stack((g[..., 0] + g[..., 2], g[..., 1] + g[..., 2]), axis=-1)


# Generic string-method interface uses the collinear surface.
energy = leps_collinear
gradient = leps_collinear_gradient

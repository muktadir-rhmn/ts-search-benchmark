"""Shared numerical utilities for freezing and growing string methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Union
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import LinearConstraint, NonlinearConstraint, minimize, minimize_scalar, root


EnergyFunction = Callable[[ArrayLike], Union[float, NDArray[np.float64]]]
GradientFunction = Callable[[ArrayLike], NDArray[np.float64]]

# Selectable algorithms; the first entry of each tuple is the default.
# FSM freezing step: minimize the energy in the hyperplane ⊥ growth direction.
FSM_NODE_OPTIMIZERS = ("SLSQP", "trust-constr", "L-BFGS-B (subspace)", "Brent (1-D line)")
# GSM string relaxation: drive the perpendicular gradient of every node to zero.
GSM_STRING_OPTIMIZERS = ("Quadratic line search", "Steepest descent", "FIRE", "Quick-min")
# Transition-state refinement started from the highest-energy path image.
TS_REFINEMENTS = ("Root (hybr)", "Newton-Raphson", "P-RFO", "Dimer", "None")


def _check_choice(value: str, choices: tuple[str, ...], what: str) -> None:
    if value not in choices:
        raise ValueError(f"unknown {what} {value!r}; choose one of {', '.join(choices)}")


@dataclass
class StringResult:
    """Result returned by either string algorithm."""

    images: NDArray[np.float64]
    energies: NDArray[np.float64]
    transition_state: NDArray[np.float64]
    transition_state_energy: float
    converged: bool
    iterations: int
    max_perpendicular_gradient: float
    method: str
    metadata: dict[str, float | int | str] = field(default_factory=dict)
    # Cost of the whole run. "calls" counts function invocations; "evaluations"
    # counts points, since one vectorized call on a (n, dim) string evaluates n.
    energy_calls: int = 0
    gradient_calls: int = 0
    energy_evaluations: int = 0
    gradient_evaluations: int = 0

    @property
    def barrier_from_reactant(self) -> float:
        return self.transition_state_energy - float(self.energies[0])


class CountedFunction:
    """Wrap an energy or gradient function and count how often it is used."""

    def __init__(self, function: Callable[[ArrayLike], object]) -> None:
        self.function = function
        self.calls = 0  # number of invocations
        self.evaluations = 0  # number of points evaluated across all invocations

    def __call__(self, point: ArrayLike):
        array = np.asarray(point)
        self.calls += 1
        self.evaluations += 1 if array.ndim <= 1 else int(np.prod(array.shape[:-1]))
        return self.function(point)


def evaluation_counts(energy: CountedFunction, gradient: CountedFunction) -> dict[str, int]:
    """Counts so far, as keyword arguments for StringResult or a step snapshot."""

    return {
        "energy_calls": energy.calls,
        "gradient_calls": gradient.calls,
        "energy_evaluations": energy.evaluations,
        "gradient_evaluations": gradient.evaluations,
    }


def record_evaluation_counts(
    result: StringResult, energy: CountedFunction, gradient: CountedFunction
) -> StringResult:
    for name, value in evaluation_counts(energy, gradient).items():
        setattr(result, name, value)
    return result


def unit(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    norm = np.linalg.norm(vector)
    if norm < 1.0e-14:
        return np.zeros_like(vector)
    return vector / norm


def redistribute(images: NDArray[np.float64], count: int | None = None) -> NDArray[np.float64]:
    """Redistribute images uniformly in arc length while preserving endpoints."""

    images = np.asarray(images, dtype=float)
    count = len(images) if count is None else count
    distances = np.linalg.norm(np.diff(images, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(distances)))
    if arc[-1] < 1.0e-14:
        return np.repeat(images[:1], count, axis=0)
    targets = np.linspace(0.0, arc[-1], count)
    output = np.empty((count, images.shape[1]))
    for coordinate in range(images.shape[1]):
        output[:, coordinate] = np.interp(targets, arc, images[:, coordinate])
    output[0], output[-1] = images[0], images[-1]
    return output


def tangents(images: NDArray[np.float64], energies: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute stable, energy-weighted string tangents."""

    n = len(images)
    result = np.zeros_like(images)
    result[0] = unit(images[1] - images[0])
    result[-1] = unit(images[-1] - images[-2])
    for i in range(1, n - 1):
        forward = images[i + 1] - images[i]
        backward = images[i] - images[i - 1]
        e_prev, e_here, e_next = energies[i - 1 : i + 2]
        if e_next > e_here > e_prev:
            tangent = forward
        elif e_next < e_here < e_prev:
            tangent = backward
        else:
            de_forward = abs(e_next - e_here)
            de_backward = abs(e_prev - e_here)
            high, low = max(de_forward, de_backward), min(de_forward, de_backward)
            if e_next > e_prev:
                tangent = high * forward + low * backward
            else:
                tangent = low * forward + high * backward
        result[i] = unit(tangent)
    return result


def perpendicular_gradients(
    images: NDArray[np.float64], energy: EnergyFunction, gradient: GradientFunction
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    energies = np.asarray(energy(images), dtype=float)
    grads = np.asarray(gradient(images), dtype=float)
    tau = tangents(images, energies)
    perpendicular = grads - np.sum(grads * tau, axis=1, keepdims=True) * tau
    perpendicular[[0, -1]] = 0.0
    return energies, perpendicular, tau


def relax_string(
    images: NDArray[np.float64],
    energy: EnergyFunction,
    gradient: GradientFunction,
    bounds: tuple[tuple[float, float], ...] | None,
    max_iterations: int,
    tolerance: float,
    step_size: float,
    redistribute_every: int = 1,
    callback: Callable[[int, NDArray[np.float64], float], None] | None = None,
    optimizer: str = GSM_STRING_OPTIMIZERS[0],
) -> tuple[NDArray[np.float64], int, float, bool]:
    """Relax a string with reparametrization, using one of ``GSM_STRING_OPTIMIZERS``.

    * ``"Steepest descent"``: x += step * F with a simple step controller.
    * ``"FIRE"``: fast inertial relaxation engine (Bitzek et al., 2006), the
      usual choice for NEB/string methods. Damped MD whose velocity is mixed
      toward the force and whose time step grows while the motion is downhill.
    * ``"Quick-min"``: per-node damped MD that keeps only the velocity
      component along the current force (Jónsson et al.).

    Here F is the perpendicular force. It is not the gradient of any objective
    (and reparametrization changes the variables every step), so energy-based
    line searches do not apply; all three methods use the force only.
    ``step_size`` is the steepest-descent step; the MD methods start from the
    equivalent time step sqrt(step_size), since one MD step moves dt**2 * F.

    If given, ``callback(steps_taken, images, max_perpendicular_gradient)`` is
    called for the starting string, after every step, and for the final string.
    """

    _check_choice(optimizer, GSM_STRING_OPTIMIZERS, "string optimizer")
    x = np.array(images, dtype=float, copy=True)
    current_step = step_size
    last_max = np.inf
    # MD state for FIRE / Quick-min (standard FIRE parameters).
    velocity = np.zeros_like(x)
    time_step = np.sqrt(step_size)
    max_time_step = 10.0 * time_step
    fire_alpha = 0.1
    downhill_steps = 0
    for iteration in range(1, max_iterations + 1):
        _, perpendicular, tau = perpendicular_gradients(x, energy, gradient)
        norms = np.linalg.norm(perpendicular[1:-1], axis=1)
        max_norm = float(np.max(norms, initial=0.0))
        if callback is not None:
            callback(iteration - 1, x.copy(), max_norm)
        if max_norm < tolerance:
            return x, iteration, max_norm, True

        force = -perpendicular
        if optimizer == "Steepest descent":
            displacement = current_step * force
        elif optimizer == "FIRE":
            # Tangents rotate as the string moves; keep the velocity perpendicular.
            velocity -= np.sum(velocity * tau, axis=1, keepdims=True) * tau
            velocity[[0, -1]] = 0.0
            power = float(np.sum(force * velocity))
            if power > 0.0:
                velocity = (1.0 - fire_alpha) * velocity + fire_alpha * np.linalg.norm(velocity) * unit(force)
                if downhill_steps > 5:
                    time_step = min(time_step * 1.1, max_time_step)
                    fire_alpha *= 0.99
                downhill_steps += 1
            else:
                # Uphill: stop, shrink the time step and restart the mixing.
                velocity[:] = 0.0
                time_step *= 0.5
                fire_alpha = 0.1
                downhill_steps = 0
            velocity += time_step * force
            displacement = time_step * velocity
        else:  # Quick-min
            directions = force / np.maximum(np.linalg.norm(force, axis=1, keepdims=True), 1.0e-30)
            along = np.sum(velocity * directions, axis=1, keepdims=True)
            velocity = np.maximum(along, 0.0) * directions
            velocity += time_step * force
            displacement = time_step * velocity

        # Cap every displacement, which is essential for the steep LEPS walls.
        displacement_norm = np.linalg.norm(displacement, axis=1, keepdims=True)
        displacement *= np.minimum(1.0, 0.08 / np.maximum(displacement_norm, 1.0e-30))
        trial = x + displacement
        if bounds is not None:
            low = np.array([item[0] for item in bounds])
            high = np.array([item[1] for item in bounds])
            trial[1:-1] = np.clip(trial[1:-1], low, high)
        if iteration % redistribute_every == 0:
            trial = redistribute(trial)

        # A simple trust-step controller damps oscillation without requiring the
        # sum of image energies (which changes under reparametrization) to fall.
        # FIRE adapts its own time step; Quick-min uses a fixed one otherwise,
        # which is unstable on stiff surfaces (Muller-Brown), so it shares the
        # controller and also drops its velocity when oscillation starts.
        if optimizer == "FIRE":
            pass
        elif max_norm > 1.25 * last_max:
            current_step = max(current_step * 0.5, 1.0e-6)
            time_step = max(time_step * 0.5, 1.0e-6)
            velocity[:] = 0.0
        elif max_norm < last_max:
            current_step = min(current_step * 1.03, step_size)
            time_step = min(time_step * 1.03, np.sqrt(step_size))
        x, last_max = trial, max_norm

    _, perpendicular, _ = perpendicular_gradients(x, energy, gradient)
    max_norm = float(np.max(np.linalg.norm(perpendicular[1:-1], axis=1), initial=0.0))
    if callback is not None:
        callback(max_iterations, x.copy(), max_norm)
    return x, max_iterations, max_norm, False


def _orthonormal_complement(direction: NDArray[np.float64]) -> NDArray[np.float64]:
    """Columns form an orthonormal basis of the hyperplane ⊥ ``direction``."""

    dimension = len(direction)
    basis, _ = np.linalg.qr(np.column_stack((direction, np.eye(dimension))))
    return basis[:, 1:dimension]


def _line_interval(
    seed: NDArray[np.float64],
    direction: NDArray[np.float64],
    bounds: tuple[tuple[float, float], ...] | None,
    trust_radius: float | None,
) -> tuple[float, float]:
    """Range of t for which seed + t * direction (unit) obeys the trust radius and box."""

    low, high = (-np.inf, np.inf) if trust_radius is None else (-trust_radius, trust_radius)
    if bounds is not None:
        for value, step, (lo, hi) in zip(seed, direction, bounds):
            if abs(step) > 1.0e-14:
                t1, t2 = sorted(((lo - value) / step, (hi - value) / step))
                low, high = max(low, t1), min(high, t2)
    if low > high:  # seed outside the box; stay at the seed
        low = high = 0.0
    return float(low), float(high)


def minimize_in_perpendicular_hyperplane(
    seed: NDArray[np.float64],
    growth_direction: NDArray[np.float64],
    energy: EnergyFunction,
    gradient: GradientFunction,
    bounds: tuple[tuple[float, float], ...] | None,
    max_iterations: int,
    tolerance: float,
    trust_radius: float | None = None,
    trajectory: list[NDArray[np.float64]] | None = None,
    method: str = FSM_NODE_OPTIMIZERS[0],
) -> tuple[NDArray[np.float64], bool, int]:
    """Minimize the energy within the hyperplane through ``seed`` perpendicular
    to ``growth_direction`` (the FSM freezing step), staying within
    ``trust_radius`` of the seed when one is given.

    ``method`` is one of ``FSM_NODE_OPTIMIZERS``:

    * ``"SLSQP"``: the hyperplane, trust sphere and box as explicit constraints.
    * ``"trust-constr"``: the same constraints, interior-point/SQP trust region.
    * ``"L-BFGS-B (subspace)"``: the linear hyperplane constraint is eliminated
      by writing q = seed + B y (B an orthonormal basis of the hyperplane), so
      only the box remains. For a 1-D hyperplane (2-D surfaces) the trust
      radius and box give the exact interval for y; in higher dimensions y is
      kept in the cube inscribed in the trust sphere and the final point is
      clipped to the box.
    * ``"Brent (1-D line)"``: on a 2-D surface the hyperplane is a line and the
      feasible set an interval, so bounded Brent minimization is exact and
      needs no gradients. Not available in higher dimensions.

    If ``trajectory`` is a list, every optimizer iterate is appended to it.
    """

    _check_choice(method, FSM_NODE_OPTIMIZERS, "FSM node optimizer")
    seed = np.asarray(seed, dtype=float)
    growth_direction = unit(growth_direction)
    record = (lambda q: None) if trajectory is None else (lambda q: trajectory.append(np.array(q, dtype=float)))

    if method in ("L-BFGS-B (subspace)", "Brent (1-D line)"):
        basis = _orthonormal_complement(growth_direction)
        free = basis.shape[1]
        if free == 1:
            y_bounds = [_line_interval(seed, basis[:, 0], bounds, trust_radius)]
        elif method == "Brent (1-D line)":
            raise ValueError("Brent (1-D line) needs a 2-D surface (a 1-D perpendicular hyperplane)")
        else:
            half = np.inf if trust_radius is None else trust_radius / np.sqrt(free)
            y_bounds = [(-half, half)] * free

        def to_q(y: ArrayLike) -> NDArray[np.float64]:
            q = seed + basis @ np.atleast_1d(np.asarray(y, dtype=float))
            if bounds is not None and free > 1:
                q = np.clip(q, [b[0] for b in bounds], [b[1] for b in bounds])
            return q

        if method == "Brent (1-D line)":
            low, high = y_bounds[0]
            if high - low < 1.0e-12:
                return seed.copy(), True, 0
            best = [np.inf]

            def line_energy(t: float) -> float:
                value = float(energy(to_q(t)))
                # Brent jumps around the interval; record only improvements so
                # the trajectory shows the best point so far, like an iterate.
                if value < best[0]:
                    best[0] = value
                    record(to_q(t))
                return value

            solved = minimize_scalar(
                line_energy, bounds=(low, high), method="bounded",
                options={"maxiter": max_iterations, "xatol": tolerance * 1.0e-2},
            )
            return to_q(solved.x), bool(solved.success), int(getattr(solved, "nit", solved.nfev))

        solved = minimize(
            lambda y: float(energy(to_q(y))),
            np.zeros(free),
            # Chain rule: dE/dy = B^T dE/dq.
            jac=lambda y: basis.T @ np.asarray(gradient(to_q(y)), dtype=float),
            method="L-BFGS-B",
            bounds=y_bounds,
            options={"maxiter": max_iterations, "ftol": tolerance * 1.0e-2},
            callback=lambda y, *_: record(to_q(y)),
        )
        return to_q(solved.x), bool(solved.success), int(solved.nit)

    # Both SLSQP and trust-constr take the same constraint objects; SciPy
    # converts them to SLSQP's internal "eq"/"ineq" form when needed.
    # Hyperplane constraint: growth_direction . q == growth_direction . seed,
    # i.e. (q - seed) . growth_direction == 0. Keeps the image at the same
    # distance along the growth direction as the seed, so it can only move
    # sideways. Without it the minimizer would slide the image downhill into
    # the nearest basin (back toward the reactant or product), and the front
    # would stop advancing.
    level = float(growth_direction @ seed)
    constraints: list[LinearConstraint | NonlinearConstraint] = [
        LinearConstraint(growth_direction[None, :], level, level)
    ]
    if trust_radius is not None:
        # Trust-region constraint: |q - seed|^2 <= trust_radius^2.
        # Limits how far the image may move sideways from its seed. The
        # hyperplane can cut across a whole valley, so without a limit the
        # image could land far from the string, e.g. in a different basin.
        constraints.append(
            NonlinearConstraint(
                lambda q: np.dot(q - seed, q - seed), -np.inf, trust_radius**2,
                jac=lambda q: 2.0 * (q - seed)[None, :],
            )
        )

    if method == "trust-constr":
        with warnings.catch_warnings():
            # trust-constr warns when it approximates constraint Hessians.
            warnings.simplefilter("ignore")
            solved = minimize(
                lambda q: float(energy(q)),
                seed.copy(),
                jac=lambda q: np.asarray(gradient(q), dtype=float),
                method="trust-constr",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": max_iterations, "gtol": tolerance, "xtol": 1.0e-8},
                callback=lambda q, *_: record(q),
            )
        return np.asarray(solved.x, dtype=float), bool(solved.success), int(solved.nit)

    # SLSQP.
    with warnings.catch_warnings():
        # SLSQP can briefly test an out-of-box step and then clips it itself.
        warnings.filterwarnings("ignore", message="Values in x were outside bounds")
        solved = minimize(
            lambda q: float(energy(q)),
            np.asarray(seed, dtype=float),
            jac=lambda q: np.asarray(gradient(q), dtype=float),
            method="SLSQP",
            # Box constraint: keep every coordinate inside the region where
            # the potential is meant to be explored (e.g. the plot area).
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": max_iterations, "ftol": tolerance * 1.0e-2},
            callback=record,
        )
    return np.asarray(solved.x, dtype=float), bool(solved.success), int(solved.nit)


def finite_difference_hessian(gradient: GradientFunction, point: NDArray[np.float64], h: float = 1.0e-4) -> NDArray[np.float64]:
    """Symmetrized Hessian from central differences of the analytic gradient."""

    dimension = len(point)
    hessian = np.empty((dimension, dimension))
    for column in range(dimension):
        delta = np.zeros(dimension)
        delta[column] = h
        hessian[:, column] = (np.asarray(gradient(point + delta)) - np.asarray(gradient(point - delta))) / (2 * h)
    # Finite differences leave small asymmetries; eigvalsh assumes symmetry.
    return 0.5 * (hessian + hessian.T)


def _cap(step: NDArray[np.float64], max_step: float) -> NDArray[np.float64]:
    norm = np.linalg.norm(step)
    return step if norm <= max_step else step * (max_step / norm)


def refine_saddle(
    method: str,
    gradient: GradientFunction,
    start: NDArray[np.float64],
    guide: NDArray[np.float64],
    max_step: float,
    max_iterations: int = 200,
) -> tuple[NDArray[np.float64], bool]:
    """Search for a first-order saddle near ``start`` with one of ``TS_REFINEMENTS``.

    ``guide`` is the string tangent at ``start``: the path's estimate of the
    reaction coordinate, i.e. of the Hessian mode that should be climbed.
    Newton-type steps are capped at ``max_step``.

    * ``"Root (hybr)"``: Powell hybrid root solve of grad E = 0. Converges to
      whichever stationary point is nearest (minimum, saddle or maximum).
    * ``"Newton-Raphson"``: full Newton steps with the finite-difference
      Hessian. Quadratic convergence, but equally indifferent to the kind of
      stationary point, so it relies on the string's good starting point.
    * ``"P-RFO"``: partitioned rational-function optimization (eigenvector
      following). Maximizes along the Hessian mode most parallel to ``guide``
      and minimizes along all others, so it targets first-order saddles.
    * ``"Dimer"``: Hessian-free. Rotates a short dimer to the lowest-curvature
      mode using only gradient differences, then moves uphill along it and
      downhill in all other directions. The method that scales to many
      dimensions; slower to converge on these small surfaces.
    """

    _check_choice(method, TS_REFINEMENTS, "TS refinement")
    x = np.array(start, dtype=float, copy=True)
    if method == "None":
        return x, False
    if method == "Root (hybr)":
        solved = root(lambda q: np.asarray(gradient(q), dtype=float), x)
        return np.asarray(solved.x, dtype=float), bool(solved.success)

    g = np.asarray(gradient(x), dtype=float)
    # Converged when the gradient has fallen by many orders of magnitude.
    gtol = (1.0e-6 if method == "Dimer" else 1.0e-9) * max(1.0, float(np.linalg.norm(g)))
    mode = unit(np.asarray(guide, dtype=float))

    if method == "Dimer":
        h = 1.0e-3  # dimer half-length

        def hessian_times(v: NDArray[np.float64]) -> NDArray[np.float64]:
            return (np.asarray(gradient(x + h * v)) - np.asarray(gradient(x - h * v))) / (2 * h)

        scale = None  # translation step / effective force, adapted below
        last_force = np.inf
        for _ in range(max_iterations):
            if np.linalg.norm(g) < gtol:
                return x, True
            # Rotation: minimize the curvature N.HN over the plane of N and the
            # rotational force; in that plane the minimum angle is exact.
            for _ in range(8):
                hn = hessian_times(mode)
                curvature = float(mode @ hn)
                rotational = -(hn - curvature * mode)
                if np.linalg.norm(rotational) < 1.0e-8 * max(1.0, abs(curvature)):
                    break
                theta = unit(rotational)
                h_theta = hessian_times(theta)
                angle = 0.5 * np.arctan2(-2.0 * float(mode @ h_theta), -(curvature - float(theta @ h_theta)))
                mode = unit(np.cos(angle) * mode + np.sin(angle) * theta)
                if abs(angle) < 1.0e-4:
                    break
            curvature = float(mode @ hessian_times(mode))
            force = -g
            # Invert the force along the mode so a saddle becomes a minimum of
            # the effective landscape; on a convex region only climb the mode.
            effective = force - 2.0 * (force @ mode) * mode if curvature < 0 else -(force @ mode) * mode
            effective_norm = float(np.linalg.norm(effective))
            if scale is None:
                scale = 0.5 / max(abs(curvature), 1.0e-12)
            elif effective_norm > last_force:
                scale *= 0.5
            else:
                scale = min(scale * 1.2, 1.0 / max(abs(curvature), 1.0e-12))
            last_force = effective_norm
            x = x + _cap(scale * effective, max_step)
            g = np.asarray(gradient(x), dtype=float)
        return x, bool(np.linalg.norm(g) < gtol)

    for _ in range(max_iterations):
        if np.linalg.norm(g) < gtol:
            return x, True
        hessian = finite_difference_hessian(gradient, x)
        if method == "Newton-Raphson":
            step = -np.linalg.lstsq(hessian, g, rcond=None)[0]
        else:  # P-RFO
            values, vectors = np.linalg.eigh(hessian)
            components = vectors.T @ g
            # Follow the mode that overlaps most with the previous one (at
            # first, with the string tangent).
            k = int(np.argmax(np.abs(vectors.T @ mode)))
            mode = vectors[:, k] * np.sign(vectors[:, k] @ mode or 1.0)
            # Maximize along mode k: shift above its eigenvalue.
            shift_up = 0.5 * values[k] + 0.5 * np.sqrt(values[k] ** 2 + 4.0 * components[k] ** 2)
            others = [i for i in range(len(values)) if i != k]
            # Minimize along the rest: lowest eigenvalue of the augmented RFO matrix.
            augmented = np.zeros((len(others) + 1, len(others) + 1))
            augmented[:-1, :-1] = np.diag(values[others])
            augmented[:-1, -1] = augmented[-1, :-1] = components[others]
            shift_down = float(np.linalg.eigvalsh(augmented)[0])
            shifts = np.full(len(values), shift_down)
            shifts[k] = shift_up
            denominators = values - shifts
            denominators[np.abs(denominators) < 1.0e-14] = 1.0e-14
            step = vectors @ (-components / denominators)
        step = _cap(step, max_step)
        x = x + step
        g = np.asarray(gradient(x), dtype=float)
        if np.linalg.norm(step) < 1.0e-12 * max(1.0, float(np.linalg.norm(x))):
            break
    return x, bool(np.linalg.norm(g) < gtol)


def build_result_with_transition_state(
    images: NDArray[np.float64],
    energy: EnergyFunction,
    gradient: GradientFunction,
    method: str,
    converged: bool,
    iterations: int,
    max_perpendicular_gradient: float,
    bounds: tuple[tuple[float, float], ...] | None,
    ts_refinement: str = TS_REFINEMENTS[0],
) -> StringResult:
    """Evaluate the path and optionally polish a path image as a saddle.

    ``ts_refinement`` picks the saddle search (see ``refine_saddle``);
    ``"None"`` reports the highest path image unchanged.

    A successful stationary-point refinement is written back into the nearest
    interior image. This invariant ensures that the reported transition state
    is always part of the returned and plotted pathway.
    """

    # Default TS estimate: the highest-energy image on the string.
    # Copy the images so writing a refined saddle back into the path below
    # cannot modify the caller's array. This default is always valid, so if
    # every refinement check below fails we still return a sensible answer.
    images = np.array(images, dtype=float, copy=True)
    energies = np.asarray(energy(images), dtype=float)
    highest = int(np.argmax(energies))
    ts = images[highest].copy()
    ts_index = highest
    ts_source = "highest path image"

    # Stationary-point refinement, starting from the highest image. The string
    # only samples the path at discrete points, so the true saddle usually lies
    # between images; the chosen search locates it exactly. Reject points that
    # wander too far (> 0.6) from the starting image, since those belong to
    # some other stationary point on the surface.
    spacing = float(np.median(np.linalg.norm(np.diff(images, axis=0), axis=1)))
    guide = tangents(images, energies)[highest]
    refined, success = refine_saddle(ts_refinement, gradient, ts, guide, max_step=max(spacing, 1.0e-8))
    if success and np.linalg.norm(refined - ts) < 0.6:
        candidate = refined
        # Box check: discard points outside the bounds, i.e. outside the region
        # the potential is meant to be explored in (and plotted over).
        inside = bounds is None or all(lo <= v <= hi for v, (lo, hi) in zip(candidate, bounds))
        if inside:
            # Classify the stationary point with the finite-difference Hessian,
            # because grad E = 0 holds at minima and maxima as well as saddles.
            hessian = finite_difference_hessian(gradient, candidate)
            # Saddle check: at least one negative eigenvalue means the root is
            # not a minimum (a minimum would be the reactant or product, not a
            # TS).
            if np.min(np.linalg.eigvalsh(hessian)) < 0.0:
                # Anchor the saddle to the path: replace the nearest interior
                # image with it. Refining the TS independently used to leave
                # the marker off the FSM polyline, so the saddle is accepted
                # only when it lies within ~2.5 image spacings of the string
                # (i.e. belongs to the same local path). Endpoints are excluded
                # by the clip because they are the fixed reactant and product.
                nearest = int(np.argmin(np.linalg.norm(images - candidate, axis=1)))
                nearest = int(np.clip(nearest, 1, len(images) - 2))
                if np.linalg.norm(images[nearest] - candidate) <= max(2.5 * spacing, 1.0e-8):
                    images[nearest] = candidate
                    ts = images[nearest].copy()
                    ts_index = nearest
                    ts_source = "stationary point anchored to path"

    # Final TS selection: recompute energies, since an image may have been
    # replaced above.
    energies = np.asarray(energy(images), dtype=float)
    # The TS reported by a discrete string method is, by definition, the
    # highest-energy image on that string. Re-select after optional anchoring so
    # the coordinates, energy, metadata, and plotted polyline cannot diverge.
    ts_index = int(np.argmax(energies))
    ts = images[ts_index].copy()
    # If the anchored saddle is not the highest image (another image sits
    # higher), record that the reported TS came from the path, not the solve.
    if ts_source == "stationary point anchored to path" and ts_index != nearest:
        ts_source = "highest path image after saddle anchoring"

    # Package the path, energies, TS and convergence info. The metadata
    # records which image is the TS and how it was found, for later reporting.
    return StringResult(
        images=images,
        energies=energies,
        transition_state=ts,
        transition_state_energy=float(energies[ts_index]),
        converged=converged,
        iterations=iterations,
        max_perpendicular_gradient=max_perpendicular_gradient,
        method=method,
        metadata={
            "transition_state_index": ts_index,
            "transition_state_source": ts_source,
            "ts_refinement": ts_refinement,
        },
    )

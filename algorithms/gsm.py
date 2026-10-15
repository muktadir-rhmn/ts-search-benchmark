"""Growing String Method (GSM) of Peters, Heyden, Bell and Chakraborty,
J. Chem. Phys. 120, 7877 (2004).

The string starts as two fragments, one at the reactant and one at the
product, each holding its endpoint plus one frontier node (four nodes in all,
as in the paper). Every iteration:

    check the perpendicular gradient |f⊥| at the two frontier nodes
    for each frontier with |f⊥| <= growth_tolerance, while num_images < max_images:
        grow that fragment by one node (Eq. 9)
    evolve (relax) every interior node along its perpendicular force (Eq. 1)
    reparametrize: equal arc length within each fragment, vacant gap between them (Eq. 8)
    stop once the fragments have joined and |f⊥| < tolerance, or after max_iterations

A cubic spline through all nodes (both fragments, across the gap) gives the
tangents and the positions used by reparametrization. A newly grown node is
therefore placed on the spline's interpolation into the gap, not on a straight
frontier-to-frontier line. Once the fragments have joined, the parametrization
density is uniform and the iterations are those of the string method.
"""

# --- Module dependencies -----
# Purpose: Import postponed annotations, containers, numeric types, splines, and shared GSM utilities.
# Reason: The implementation needs typed arrays plus common result, counting, tangent, and TS helpers.
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import CubicSpline

from .string_utils import (
    GSM_STRING_OPTIMIZERS,
    TS_REFINEMENTS,
    CountedFunction,
    EnergyFunction,
    GradientFunction,
    StringResult,
    _check_choice,
    build_result_with_transition_state,
    evaluation_counts,
    record_evaluation_counts,
    tangents,
    unit,
)

# --- Numerical safeguards -----
# Purpose: Define module-wide limits that control evolution stability and arc-length accuracy.
# Reason: Every optimizer and reparametrization call must use consistent numerical settings.

# Purpose: MAX_NODE_DISPLACEMENT stores the largest permitted node movement per evolution step.
# Reason: Capping movement prevents unstable jumps on steep potential-energy surfaces.
# Largest distance any node may move in one evolution step; essential for the
# steep LEPS walls.
MAX_NODE_DISPLACEMENT = 0.08

# Purpose: ARC_LENGTH_SAMPLES stores the number of spline samples per node interval.
# Reason: Dense, consistent sampling is needed to approximate spline arc length for reparametrization.
# Spline samples per node interval when measuring arc length.
ARC_LENGTH_SAMPLES = 64


# --- Callback snapshot model -----
# Purpose: Define the state record delivered to callers after meaningful GSM stages.
# Reason: Callbacks need a stable representation of geometry, progress, and evaluation counts.
@dataclass
class GSMStep:
    """Snapshot of a GSM stage, passed to ``growing_string``'s callback.

    ``stage`` is one of (in the order they occur within an iteration):
      * ``"grow"``: a node was just added to a frontier (placed on the spline);
        ``previous_images`` is the string before the growth;
      * ``"evolve"``: every interior node moved along its normal force;
        ``previous_images`` is the string before the move;
      * ``"reparametrize"``: the evolved string's nodes redistributed along
        the spline; ``previous_images`` is the evolved string.
    """

    # Purpose: stage identifies which GSM operation produced this snapshot.
    # Reason: Callback consumers may handle growth, evolution, and reparametrization differently.
    stage: str

    # Purpose: addition records the zero-based count of node additions reached so far.
    # Reason: Callback consumers need to track progress toward the fully joined string.
    addition: int  # 0-based node addition so far (the first two are automatic)

    # Purpose: total_additions records how many additions are required for a complete string.
    # Reason: It provides the denominator for growth progress reporting.
    total_additions: int

    # Purpose: iteration records the one-based evolution step associated with the snapshot.
    # Reason: It lets callback consumers order stages within the optimization run.
    iteration: int  # 1-based evolution step ("grow" belongs to the step about to run)

    # Purpose: images stores a copy of the current reactant-to-product string.
    # Reason: Callback consumers need the geometry produced by the reported stage.
    images: NDArray[np.float64]  # current string, reactant -> product

    # Purpose: previous_images stores the geometry immediately before the reported stage.
    # Reason: It enables visualization or comparison of the stage's movement.
    previous_images: NDArray[np.float64] | None  # string before this stage moved it

    # Purpose: reactant_count stores the reactant-fragment size while the string is growing.
    # Reason: It identifies the two fragments until they join, after which the value is unnecessary.
    reactant_count: int | None  # images[:reactant_count] grew from the reactant; None once joined

    # Purpose: new_node stores the coordinates added during a growth stage.
    # Reason: Growth callbacks may need to highlight the newly created frontier.
    new_node: NDArray[np.float64] | None  # only for "grow"

    # Purpose: max_perpendicular_gradient stores the largest interior driving-gradient magnitude.
    # Reason: It exposes the convergence measure associated with evolution and reparametrization.
    max_perpendicular_gradient: float | None  # max interior |f⊥| driving this step; None for "grow"

    # Purpose: energy_calls stores the running number of energy function invocations.
    # Reason: Callbacks use it to report computational work independently of batched evaluations.
    # Running totals of energy/gradient use since the run started.
    energy_calls: int = 0

    # Purpose: gradient_calls stores the running number of gradient function invocations.
    # Reason: Callbacks use it to report gradient-call cost independently of evaluated geometries.
    gradient_calls: int = 0

    # Purpose: energy_evaluations stores the number of geometries evaluated for energy.
    # Reason: Vectorized calls may cover multiple nodes, so this count complements energy_calls.
    energy_evaluations: int = 0

    # Purpose: gradient_evaluations stores the number of geometries evaluated for gradients.
    # Reason: It captures total gradient workload even when evaluations are batched.
    gradient_evaluations: int = 0


# Purpose: GSMCallback stores the callable signature accepted for GSM progress reporting.
# Reason: A shared alias keeps the driver interface and callback records type-consistent.
GSMCallback = Callable[[GSMStep], None]


# --- String geometry --------------------------------------------------------
# Purpose: Group helpers that construct the path spline, tangents, forces, and equal-spacing geometry.
# Reason: Growth and relaxation both depend on the same geometric representation of the string.
def _spline(nodes: NDArray[np.float64]) -> tuple[CubicSpline, NDArray[np.float64]]:
    """Natural ("free end") cubic spline through the nodes, parametrized by
    normalized chord length. Returns the spline and the nodes' parameters."""

    # --- Build normalized spline parameters -----
    # Purpose: Convert distances between adjacent nodes into monotonically increasing spline knots.
    # Reason: Chord-length parametrization follows the current path geometry better than node indices.

    # Purpose: chords stores Euclidean distances between each adjacent pair of nodes.
    # Reason: Their cumulative sum supplies a geometry-aware spline coordinate.
    chords = np.linalg.norm(np.diff(nodes, axis=0), axis=1)

    # Purpose: knots stores each node's normalized cumulative chord-length coordinate.
    # Reason: CubicSpline requires a strictly increasing independent coordinate for every node.
    knots = np.concatenate(([0.0], np.cumsum(chords)))
    knots /= max(knots[-1], 1.0e-14)

    # --- Enforce distinct knots -----
    # Purpose: Nudge repeated or numerically tied knot values into increasing order.
    # Reason: Coincident nodes would otherwise violate CubicSpline's input requirements.
    # Coincident nodes would give repeated knots; nudge them apart.
    knots = np.maximum.accumulate(knots + 1.0e-12 * np.arange(len(knots)))

    # --- Construct spline -----
    # Purpose: Return the natural cubic path and the node parameters used to evaluate it.
    # Reason: Downstream tangent and reparametrization calculations require both objects.
    return CubicSpline(knots, nodes, bc_type="natural"), knots


def string_tangents(
    nodes: NDArray[np.float64], energies: NDArray[np.float64], frontiers: tuple[int, ...] = ()
) -> NDArray[np.float64]:
    """Unit tangents: upwind (energy-weighted) differences at every node,
    except the ``frontiers``, which take the cubic-spline tangent.

    The paper uses spline tangents because finite differences point at a
    distant node when the spacing is uneven, i.e. at a frontier next to the
    vacant gap (its Fig. 1). Elsewhere the spacing is uniform, and there the
    spline (like centered differences) lets kinks grow at sharp bends of the
    path once nodes are close, whereas the upwind tangent is stable for all
    spacings (Henkelman and Jonsson). Using upwind tangents away from the
    frontiers avoids the dekinking force of the paper's appendix.
    """

    # --- Initialize robust tangents -----
    # Purpose: Compute upwind energy-weighted unit tangents for every string node.
    # Reason: These tangents remain stable at sharp bends and unequal spacings away from the gap.

    # Purpose: tau stores the tangent direction used to remove force components along the path.
    # Reason: It is returned to the optimizer and selectively replaced at growing frontiers.
    tau = tangents(nodes, energies)

    # --- Correct frontier tangents -----
    # Purpose: Replace finite-difference frontier directions with derivatives of the global spline.
    # Reason: A frontier's adjacent node lies across the vacant gap and is too distant for a local difference.
    if frontiers:
        # Purpose: spline stores the interpolating path used to obtain frontier derivatives.
        # Reason: It supplies directions that follow the path through the currently vacant region.
        spline, knots = _spline(nodes)

        # Purpose: index stores frontier indices in a form supported by NumPy advanced indexing.
        # Reason: Both selected rows of tau must be updated together when two frontiers exist.
        index = list(frontiers)

        # Purpose: d iterates over spline derivatives evaluated at the frontier knots.
        # Reason: Each derivative must be normalized independently before becoming a tangent.
        tau[index] = [unit(d) for d in spline(knots[index], 1)]

    # --- Return path directions -----
    # Purpose: Return tangent vectors aligned with the current string geometry.
    # Reason: Perpendicular-force projection depends on these directions.
    return tau


def perpendicular_forces(
    nodes: NDArray[np.float64],
    energy: EnergyFunction,
    gradient: GradientFunction,
    frontiers: tuple[int, ...] = (),
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Perpendicular gradient g - (t·g)t of Eq. (1) at every node (the normal
    force is its negative) and the tangents. Endpoints are fixed minima, so
    their entries are zero."""

    # --- Evaluate path state -----
    # Purpose: Evaluate energies and Cartesian gradients at all nodes in one batch.
    # Reason: Energies select stable tangents, while gradients provide the optimization force.

    # Purpose: energies stores one scalar potential energy for every node.
    # Reason: The tangent scheme uses local energy ordering to choose upwind directions.
    energies = np.asarray(energy(nodes), dtype=float)

    # Purpose: grads stores the full coordinate gradient at every node.
    # Reason: Its component normal to the string drives path relaxation.
    grads = np.asarray(gradient(nodes), dtype=float)

    # Purpose: tau stores the unit path tangent at every node.
    # Reason: Gradient projection requires the local direction parallel to the string.
    tau = string_tangents(nodes, energies, frontiers)

    # --- Project gradients -----
    # Purpose: Remove each gradient's tangent component and fix both endpoint entries at zero.
    # Reason: GSM relaxes only normal motion while keeping the endpoint minima stationary.

    # Purpose: perpendicular stores the gradient component orthogonal to the path at each node.
    # Reason: Its negative is the normal force used by every supported optimizer.
    perpendicular = grads - np.sum(grads * tau, axis=1, keepdims=True) * tau
    perpendicular[[0, -1]] = 0.0

    # --- Return force geometry -----
    # Purpose: Return both projected gradients and their defining tangents.
    # Reason: The caller needs the force for motion and the tangent for velocity projection.
    return perpendicular, tau


def reparametrize(
    nodes: NDArray[np.float64], reactant_count: int, product_count: int, max_images: int
) -> NDArray[np.float64]:
    """Redistribute the nodes along the spline through them (Eqs. 4-8).

    While growing, the density rho(s) of Eq. (8) is 1 on [0, a1] and [a2, 1]
    and 0 on the vacant gap (a1, a2). The frontiers sit at
    a1 = (reactant_count - 1) ds and a2 = 1 - (product_count - 1) ds with
    ds = 1 / (max_images - 1), so each fragment keeps an equal node spacing
    and the gap closes to one ordinary spacing exactly when the fragments hold
    ``max_images`` nodes between them. Once joined the density is uniform.

    The counts may exceed the current number of nodes: growing a fragment is
    just a call with its count raised by one, and the new frontier lands on
    the spline's interpolation into the gap.
    """

    # --- Sample spline arc length -----
    # Purpose: Approximate normalized cumulative distance along the interpolated string.
    # Reason: Equal target parameters must represent equal physical arc lengths, not equal spline parameters.

    # Purpose: spline stores the smooth path through all current nodes, including across the gap.
    # Reason: Both existing-node redistribution and new-frontier placement use this same path.
    spline, _ = _spline(nodes)

    # Purpose: u stores dense, uniformly spaced spline parameter samples.
    # Reason: Sampling provides a numerical map from spline parameter to physical arc length.
    # Arc length along the spline, sampled densely, then normalized to [0, 1].
    u = np.linspace(0.0, 1.0, ARC_LENGTH_SAMPLES * (len(nodes) - 1) + 1)

    # Purpose: s stores normalized cumulative arc length at every sampled spline parameter.
    # Reason: Interpolation against s converts desired path distances back into spline parameters.
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(spline(u), axis=0), axis=1))))
    s /= max(s[-1], 1.0e-14)

    # --- Choose target arc positions -----
    # Purpose: Build uniform targets for a joined string or two equal-density fragment ranges while growing.
    # Reason: The vacant gap must close progressively without redistributing nodes into it prematurely.
    if reactant_count + product_count >= max_images:
        # Purpose: targets stores normalized arc positions for every node in the joined string.
        # Reason: A complete string uses uniform density over the entire path.
        targets = np.linspace(0.0, 1.0, max_images)
    else:
        # Purpose: ds stores the final joined-string spacing in normalized arc length.
        # Reason: Growing fragments retain the density they will have after joining.
        ds = 1.0 / (max_images - 1)

        # Purpose: left stores target arc positions for the reactant-side fragment.
        # Reason: These targets advance from the reactant endpoint toward the gap.
        left = ds * np.arange(reactant_count)

        # Purpose: right stores target arc positions for the product-side fragment.
        # Reason: These targets advance backward from the product endpoint toward the gap.
        right = 1.0 - ds * np.arange(product_count)[::-1]

        # Purpose: targets stores both fragment target ranges in reactant-to-product order.
        # Reason: One spline evaluation can then place every current node consistently.
        targets = np.concatenate((left, right))

    # --- Interpolate redistributed nodes -----
    # Purpose: Map target arc lengths onto the spline and restore the exact endpoint coordinates.
    # Reason: Interpolation equalizes spacing, while explicit endpoint assignment prevents numerical drift.

    # Purpose: output stores the reparametrized string coordinates.
    # Reason: Callers use it as the new geometry after growth or evolution.
    output = spline(np.interp(targets, s, u))
    output[0], output[-1] = nodes[0], nodes[-1]
    return output


# --- Evolution step ---------------------------------------------------------
# Purpose: Group optimizer state and routines that relax nodes along the normal force.
# Reason: GSM supports several evolution algorithms that share displacement limits and persistent memory.
@dataclass
class _OptimizerState:
    """Step-size / velocity memory of the string optimizer across iterations."""

    # Purpose: step_size stores the user-requested upper scale for optimizer steps.
    # Reason: Adaptive optimizers need a stable reference when shrinking or regrowing their steps.
    step_size: float

    # Purpose: current_step stores the adaptive steepest-descent step multiplier.
    # Reason: It changes across iterations in response to oscillation or improvement.
    current_step: float = 0.0

    # Purpose: time_step stores the adaptive integration step for FIRE and Quick-min.
    # Reason: Their velocity updates require persistent time scaling across iterations.
    time_step: float = 0.0

    # Purpose: velocity stores optimizer momentum for the current node array.
    # Reason: FIRE and Quick-min use prior motion to accelerate relaxation.
    velocity: NDArray[np.float64] | None = None

    # Purpose: fire_alpha stores FIRE's current velocity-force mixing factor.
    # Reason: FIRE gradually reduces mixing after sustained downhill progress.
    fire_alpha: float = 0.1

    # Purpose: downhill_steps counts consecutive FIRE steps with positive force-velocity alignment.
    # Reason: The count controls when FIRE may increase its time step.
    downhill_steps: int = 0

    # Purpose: last_max stores the preceding iteration's maximum perpendicular-gradient norm.
    # Reason: Steepest descent and Quick-min use it to detect improvement or oscillation.
    last_max: float = np.inf

    # Purpose: line_search_step stores the next quadratic line-search probe spacing.
    # Reason: Reusing accepted scales reduces poor probes on subsequent iterations.
    line_search_step: float = 0.0

    def __post_init__(self) -> None:
        # --- Initialize adaptive scales -----
        # Purpose: Derive each optimizer's starting scale from the configured step size.
        # Reason: Dataclass defaults cannot depend directly on another instance field.
        self.current_step = self.step_size
        self.time_step = np.sqrt(self.step_size)
        self.line_search_step = self.step_size

    def reset_after_growth(self) -> None:
        # --- Reset size-dependent memory -----
        # Purpose: Discard momentum and convergence history after a node is inserted.
        # Reason: Their shapes and comparisons are invalid once the string geometry changes.
        # The node count changed: drop MD velocities and the oscillation
        # reference (a new node's |f⊥| would otherwise look like an increase).
        self.velocity = None
        self.downhill_steps = 0
        self.fire_alpha = 0.1
        self.last_max = np.inf


def _quadratic_line_search_step(
    nodes: NDArray[np.float64],
    perpendicular: NDArray[np.float64],
    energy: EnergyFunction,
    gradient: GradientFunction,
    state: _OptimizerState,
    groups: list[slice],
    frontiers: tuple[int, ...],
) -> NDArray[np.float64]:
    """Paper's evolution step: move along v = (f⊥_0, ..., f⊥_n) (Eq. 10a) to
    the minimum of a quadratic fit of F = sum |f⊥|^2 (Eq. 3) through three
    points on that line, capped at the maximum allowed step.

    While the string is growing, F is split into F_left and F_right (one per
    fragment, ``groups``) and each fragment gets its own step length.
    """

    # --- Establish line-search direction -----
    # Purpose: Form the normal-force direction and a safe upper bound for its scalar step.
    # Reason: The quadratic search must not move any node beyond the global displacement cap.

    # Purpose: force stores the negative perpendicular gradient at every node.
    # Reason: The line search minimizes the objective along this downhill direction.
    force = -perpendicular

    # Purpose: largest stores the maximum node-wise normal-force magnitude.
    # Reason: It converts the displacement cap into a safe scalar step bound.
    largest = float(np.max(np.linalg.norm(force, axis=1)))

    # --- Handle stationary string -----
    # Purpose: Return an unchanged copy when no meaningful normal-force direction exists.
    # Reason: Dividing by a near-zero largest force would make the step bound unstable.
    if largest < 1.0e-14:
        return nodes.copy()

    # Purpose: max_alpha stores the largest scalar step consistent with the displacement cap.
    # Reason: Every accepted fragment step must remain numerically safe.
    max_alpha = MAX_NODE_DISPLACEMENT / largest

    # Purpose: h stores the spacing between the three objective samples.
    # Reason: It uses prior accepted scales while keeping the furthest probe within max_alpha.
    h = min(state.line_search_step, 0.5 * max_alpha)

    # --- Sample quadratic objectives -----
    # Purpose: Evaluate each fragment's squared perpendicular-force objective at three trial positions.
    # Reason: Three values determine the local quadratic used to choose a step length.

    # Purpose: samples stores F for every probe position and fragment group.
    # Reason: Each fragment receives an independent quadratic fit while the string is growing.
    # F of each fragment at alpha = 0, h, 2h.
    samples = np.empty((3, len(groups)))

    # Purpose: i indexes the row in samples for the current probe position.
    # Reason: Each objective vector must be stored in the row matching its alpha value.
    # Purpose: alpha is the scalar distance currently being probed along the normal force.
    # Reason: The three prescribed values supply the quadratic fit data.
    for i, alpha in enumerate((0.0, h, 2.0 * h)):
        # Purpose: trial stores projected gradients at the current probe geometry.
        # Reason: Fragment objectives are defined as sums of their squared perpendicular gradients.
        trial = perpendicular if alpha == 0.0 else perpendicular_forces(nodes + alpha * force, energy, gradient, frontiers)[0]

        # Purpose: g iterates over fragment slices for the current objective sample.
        # Reason: The growing string fits and advances each fragment independently.
        samples[i] = [np.sum(trial[g] ** 2) for g in groups]

    # --- Fit and apply fragment steps -----
    # Purpose: Fit a quadratic per fragment, select a safe forward step, and assemble displacement rows.
    # Reason: Independent fragment relaxation implements the paper's split objective during growth.

    # Purpose: displacement stores the coordinate movement selected for every node.
    # Reason: Per-fragment step lengths must be combined before updating the string.
    displacement = np.zeros_like(nodes)

    # Purpose: alphas stores accepted scalar steps for all fragment groups.
    # Reason: Their median becomes the next line-search probe scale.
    alphas = []

    # Purpose: k indexes the fragment's column in samples.
    # Reason: Each fragment must read the objective values belonging to its own quadratic.
    # Purpose: g is the node slice belonging to the current fragment.
    # Reason: The accepted scalar step applies only to those nodes.
    for k, g in enumerate(groups):
        # Purpose: f0, f1, and f2 store the current fragment's objective at 0, h, and 2h.
        # Reason: These three samples determine the quadratic slope and curvature.
        f0, f1, f2 = samples[:, k]

        # Purpose: curvature stores the fitted quadratic coefficient for alpha squared.
        # Reason: A positive value indicates that the model has a forward minimum.
        curvature = (f2 - 2.0 * f1 + f0) / (2.0 * h * h)  # F ~ f0 + b a + c a^2

        # Purpose: slope stores the fitted objective derivative at alpha zero.
        # Reason: Together with curvature it locates the quadratic minimum.
        slope = (4.0 * f1 - 3.0 * f0 - f2) / (2.0 * h)

        # Purpose: alpha stores the candidate or fallback step for the current fragment.
        # Reason: It selects a forward reduction while respecting the displacement limit.
        alpha = -slope / (2.0 * curvature) if curvature > 0.0 else max_alpha
        # No minimum ahead, or too far: take the maximum allowed step. A
        # minimum behind (alpha <= 0) means the fit is unreliable; step short.
        alpha = max_alpha if alpha > max_alpha else (0.5 * h if alpha <= 0.0 else alpha)
        displacement[g] = alpha * force[g]
        alphas.append(alpha)
    # --- Update probe memory -----
    # Purpose: Set the next probe spacing from the typical accepted fragment step.
    # Reason: Adapting the scale improves later quadratic samples without allowing it to vanish.
    # The next probe spacing follows the accepted steps.
    state.line_search_step = max(float(np.median(alphas)), 1.0e-8)

    # --- Return evolved geometry -----
    # Purpose: Apply the assembled line-search displacement to the current nodes.
    # Reason: Reparametrization is deliberately left to the caller after evolution.
    return nodes + displacement


def evolve_string(
    nodes: NDArray[np.float64],
    perpendicular: NDArray[np.float64],
    tau: NDArray[np.float64],
    energy: EnergyFunction,
    gradient: GradientFunction,
    state: _OptimizerState,
    optimizer: str,
    bounds: tuple[tuple[float, float], ...] | None,
    groups: list[slice],
    frontiers: tuple[int, ...] = (),
) -> NDArray[np.float64]:
    """One evolution step: move every interior node along its normal force
    f⊥ = -perpendicular (Eq. 1). Endpoints stay fixed. ``optimizer`` is one of
    ``GSM_STRING_OPTIMIZERS``:

    * ``"Quadratic line search"``: the paper's scheme (see
      ``_quadratic_line_search_step``); costs two extra string gradients;
    * ``"Steepest descent"``: x += step * f⊥ with a simple step controller;
    * ``"FIRE"``: fast inertial relaxation engine (Bitzek et al., 2006);
    * ``"Quick-min"``: per-node damped MD keeping only the velocity along f⊥.

    The result is not reparametrized; call ``reparametrize`` afterwards.
    """

    # --- Initialize evolution state -----
    # Purpose: Prepare the normal force, movable-node slice, convergence magnitude, and velocity storage.
    # Reason: Every optimizer branch shares these values and must leave endpoints fixed.

    # Purpose: force stores the normal force applied to each node.
    # Reason: All supported optimizers advance downhill along this direction.
    force = -perpendicular

    # Purpose: interior identifies every movable node while excluding the fixed endpoints.
    # Reason: Endpoint minima must remain unchanged during evolution and bounds enforcement.
    interior = slice(1, -1)

    # Purpose: max_norm stores the largest current perpendicular-gradient magnitude among interior nodes.
    # Reason: The adaptive controller compares it with the preceding iteration's value.
    max_norm = float(np.max(np.linalg.norm(perpendicular[interior], axis=1), initial=0.0))

    # --- Ensure compatible velocity memory -----
    # Purpose: Initialize optimizer momentum when absent or invalidated by a node-count change.
    # Reason: FIRE and Quick-min require one velocity vector for every node coordinate.
    if state.velocity is None or state.velocity.shape != nodes.shape:
        state.velocity = np.zeros_like(nodes)

    # --- Select optimizer update -----
    # Purpose: Compute a raw displacement using the requested relaxation algorithm.
    # Reason: Each optimizer uses different state while producing the same displacement interface.
    if optimizer == "Quadratic line search":
        # Purpose: trial stores the geometry selected by the paper's quadratic line search.
        # Reason: Subtracting nodes converts that geometry into the shared displacement representation.
        trial = _quadratic_line_search_step(nodes, perpendicular, energy, gradient, state, groups, frontiers)

        # Purpose: displacement stores the coordinate movement proposed by the active optimizer.
        # Reason: Later code applies common endpoint, magnitude, and bounds constraints to it.
        displacement = trial - nodes
    elif optimizer == "Steepest descent":
        displacement = state.current_step * force
    elif optimizer == "FIRE":
        # Purpose: velocity references persistent FIRE momentum for in-place adjustment.
        # Reason: FIRE mixes prior motion with the current force across iterations.
        velocity = state.velocity

        # --- Project FIRE velocity -----
        # Purpose: Remove tangent momentum and keep endpoint velocities zero.
        # Reason: Motion parallel to the string changes parametrization rather than improving the path.
        # Tangents rotate as the string moves; keep the velocity perpendicular.
        velocity -= np.sum(velocity * tau, axis=1, keepdims=True) * tau
        velocity[[0, -1]] = 0.0

        # --- Adapt FIRE dynamics -----
        # Purpose: Mix aligned motion and accelerate sustained descent, or restart after uphill motion.
        # Reason: FIRE gains speed when momentum is useful and damps it when the direction becomes unstable.
        if float(np.sum(force * velocity)) > 0.0:
            velocity[:] = (1.0 - state.fire_alpha) * velocity + state.fire_alpha * np.linalg.norm(velocity) * unit(force)
            if state.downhill_steps > 5:
                state.time_step = min(state.time_step * 1.1, 10.0 * np.sqrt(state.step_size))
                state.fire_alpha *= 0.99
            state.downhill_steps += 1
        else:
            # Uphill: stop, shrink the time step and restart the mixing.
            velocity[:] = 0.0
            state.time_step *= 0.5
            state.fire_alpha = 0.1
            state.downhill_steps = 0

        # --- Integrate FIRE motion -----
        # Purpose: Update velocity with the force and convert it into a coordinate displacement.
        # Reason: FIRE uses a velocity-based time integration rather than a direct gradient step.
        velocity += state.time_step * force
        displacement = state.time_step * velocity
    else:  # Quick-min
        # --- Project Quick-min momentum -----
        # Purpose: Retain only nonnegative velocity along each node's current force before accelerating it.
        # Reason: Removing opposing and transverse momentum damps oscillation around the path.

        # Purpose: directions stores a unit normal-force direction for every node.
        # Reason: Quick-min projects existing velocity independently at each node.
        directions = force / np.maximum(np.linalg.norm(force, axis=1, keepdims=True), 1.0e-30)

        # Purpose: along stores each velocity's signed component in its force direction.
        # Reason: Quick-min preserves only components that already move downhill.
        along = np.sum(state.velocity * directions, axis=1, keepdims=True)
        state.velocity = np.maximum(along, 0.0) * directions + state.time_step * force
        displacement = state.time_step * state.velocity

    # --- Constrain proposed movement -----
    # Purpose: Fix endpoints, cap each node's displacement, and enforce optional coordinate bounds.
    # Reason: Shared safety constraints must apply identically to every optimizer.
    displacement[[0, -1]] = 0.0

    # Purpose: displacement_norm stores the movement magnitude proposed for each node.
    # Reason: It supplies an independent scale factor for enforcing the maximum movement cap.
    displacement_norm = np.linalg.norm(displacement, axis=1, keepdims=True)
    displacement *= np.minimum(1.0, MAX_NODE_DISPLACEMENT / np.maximum(displacement_norm, 1.0e-30))

    # Purpose: trial stores the bounded candidate geometry returned by this evolution step.
    # Reason: Optimizer state can be updated before the caller reparametrizes this geometry.
    trial = nodes + displacement
    if bounds is not None:
        # Purpose: low and high store per-coordinate lower and upper limits.
        # Reason: Interior nodes must remain inside the caller-specified search domain.
        low, high = np.array(bounds, dtype=float).T
        trial[interior] = np.clip(trial[interior], low, high)

    # --- Adapt direct-step optimizers -----
    # Purpose: Shrink oscillatory steps or cautiously restore steps when the force improves.
    # Reason: Steepest descent and Quick-min do not have FIRE or line-search step selection.
    # Step controller for steepest descent and Quick-min (FIRE and the line
    # search choose their own steps): damp oscillation of max |f⊥|.
    if optimizer in ("Steepest descent", "Quick-min"):
        if max_norm > 1.25 * state.last_max:
            state.current_step = max(state.current_step * 0.5, 1.0e-6)
            state.time_step = max(state.time_step * 0.5, 1.0e-6)
            state.velocity[:] = 0.0
        elif max_norm < state.last_max:
            state.current_step = min(state.current_step * 1.03, state.step_size)
            state.time_step = min(state.time_step * 1.03, np.sqrt(state.step_size))
    # --- Store convergence history -----
    # Purpose: Save the current maximum force and return the constrained candidate geometry.
    # Reason: The next direct-step iteration needs this reference for adaptation.
    state.last_max = max_norm
    return trial


# --- Convergence ------------------------------------------------------------
# Purpose: Group the joined-string and force-tolerance test used by the driver.
# Reason: Growth completion alone is insufficient; all interior nodes must also be relaxed.
def is_converged(
    perpendicular: NDArray[np.float64], node_count: int, max_images: int, tolerance: float
) -> tuple[bool, float]:
    """The string has converged when the fragments have joined (all
    ``max_images`` nodes present) and every interior node satisfies
    |f⊥| < ``tolerance``. Returns (converged, max interior |f⊥|)."""

    # --- Measure and test convergence -----
    # Purpose: Compute the largest interior perpendicular-gradient norm and combine it with node completion.
    # Reason: A valid result requires both a fully joined string and a relaxed path.

    # Purpose: max_norm stores the largest remaining normal-gradient magnitude on an interior node.
    # Reason: It is both the convergence metric and a diagnostic returned to the caller.
    max_norm = float(np.max(np.linalg.norm(perpendicular[1:-1], axis=1), initial=0.0))
    return node_count >= max_images and max_norm < tolerance, max_norm


# --- Growth -----------------------------------------------------------------
# Purpose: Group the growth event model and the routine that advances converged fragment frontiers.
# Reason: The driver needs both updated geometry and per-addition metadata for callbacks.
@dataclass
class Growth:
    """One node added by ``grow_string``."""

    # Purpose: side identifies whether the reactant or product fragment grew.
    # Reason: The driver resets the corresponding fragment's overdue counter.
    side: int  # 0 = reactant fragment, 1 = product fragment

    # Purpose: before stores the complete string before this node addition.
    # Reason: Growth callbacks use it to compare the pre-growth and post-growth paths.
    before: NDArray[np.float64]  # string before this node was added

    # Purpose: after stores the reparametrized string after this node addition.
    # Reason: It is the geometry accepted for the next growth or evolution operation.
    after: NDArray[np.float64]  # string after (one node longer)

    # Purpose: new_index stores the added frontier node's index in after.
    # Reason: The callback needs to extract and identify the new coordinates.
    new_index: int  # index of the new frontier node in ``after``

    # Purpose: reactant_count stores the reactant fragment size after this addition.
    # Reason: A callback may be emitted after each of two additions in the same iteration.
    reactant_count: int  # fragment sizes after this addition

    # Purpose: product_count stores the product fragment size after this addition.
    # Reason: Together with reactant_count it records the exact intermediate growth state.
    product_count: int


def grow_string(
    nodes: NDArray[np.float64],
    perpendicular: NDArray[np.float64],
    reactant_count: int,
    product_count: int,
    max_images: int,
    growth_tolerance: float,
    overdue: tuple[bool, bool] = (False, False),
) -> tuple[NDArray[np.float64], int, int, list[Growth]]:
    """Grow each fragment whose frontier has converged (Eqs. 7 and 9).

    The indicator i(f⊥) of Eq. (7) is 1 when the frontier's perpendicular
    gradient satisfies |f⊥| <= ``growth_tolerance``; ``overdue[side]`` forces
    it to 1 (the driver's ``max_frontier_steps`` safeguard). A ready fragment
    gains one node, i.e. its end of the vacant gap moves inward by one
    spacing (a1 += ds or a2 -= ds), and ``reparametrize`` places the new
    frontier on the spline's interpolation into the gap. Nodes are added only
    while fewer than ``max_images`` exist; if both frontiers are ready for the
    last slot, the reactant side takes it.

    Returns the new nodes, the new fragment sizes and one ``Growth`` per node
    added (empty if neither frontier was ready). The string's gradients are
    stale afterwards: reparametrization moves the existing nodes too.
    """

    # --- Determine growth readiness -----
    # Purpose: Measure both frontier forces and mark fragments allowed or forced to grow.
    # Reason: Eq. 7 gates normal growth, while the overdue flags prevent indefinite stalling.

    # Purpose: frontier_norms stores the perpendicular-gradient magnitude at each fragment frontier.
    # Reason: Each value is compared with growth_tolerance for its side.
    frontier_norms = np.linalg.norm(perpendicular[[reactant_count - 1, reactant_count]], axis=1)

    # Purpose: side iterates over the reactant and product frontier identifiers.
    # Reason: Readiness must be computed independently for both fragments.
    # Purpose: ready stores whether each fragment currently satisfies either growth condition.
    # Reason: The subsequent loop uses it to skip fragments that still require relaxation.
    ready = [frontier_norms[side] <= growth_tolerance or overdue[side] for side in (0, 1)]

    # Purpose: growths accumulates one immutable description for every node added in this call.
    # Reason: The driver reports additions separately even when both fragments grow together.
    growths = []

    # --- Add ready frontier nodes -----
    # Purpose: Increase each eligible fragment count, reparametrize the path, and record the addition.
    # Reason: Moving the target frontier inward places the new node on the spline across the gap.

    # Purpose: side iterates over reactant first and product second.
    # Reason: This ordering gives the reactant side priority when only the last node slot remains.
    for side in (0, 1):
        if not ready[side] or reactant_count + product_count >= max_images:
            continue

        # --- Increase selected fragment -----
        # Purpose: Advance the node count for the eligible side by one.
        # Reason: Reparametrization interprets the new count as an inward frontier target.
        if side == 0:
            reactant_count += 1
        else:
            product_count += 1

        # Purpose: before stores the geometry preceding the current individual addition.
        # Reason: It preserves the correct callback baseline if both sides grow in one call.
        before = nodes
        nodes = reparametrize(nodes, reactant_count, product_count, max_images)

        # Purpose: new_index stores the new frontier's location in the reparametrized node array.
        # Reason: Reactant and product additions occupy different positions around the fragment boundary.
        new_index = reactant_count - 1 if side == 0 else reactant_count
        growths.append(Growth(side, before, nodes, new_index, reactant_count, product_count))

    # --- Return accumulated growth -----
    # Purpose: Return current geometry, updated fragment counts, and ordered addition records.
    # Reason: The driver must continue from the new state and notify callbacks for each addition.
    return nodes, reactant_count, product_count, growths


# --- Driver -----------------------------------------------------------------
# Purpose: Coordinate input validation, bidirectional growth, relaxation, convergence, and result construction.
# Reason: The public GSM entry point must manage the full algorithm and its reporting lifecycle.
def growing_string(
    energy: EnergyFunction,
    gradient: GradientFunction,
    reactant: ArrayLike,
    product: ArrayLike,
    *,
    num_images: int = 25,  # max_images: node count of the joined string
    bounds: tuple[tuple[float, float], ...] | None = None,
    max_iterations: int = 1000,  # evolution steps, growth and final relaxation together
    tolerance: float = 2.0e-3,  # convergence threshold on max interior |f⊥|
    growth_tolerance: float | None = None,  # frontier |f⊥| that allows growth; None = 10 * tolerance
    max_frontier_steps: int = 25,  # grow a frontier anyway after this many steps without growth
    step_size: float = 0.001,
    callback: GSMCallback | None = None,
    string_optimizer: str = GSM_STRING_OPTIMIZERS[0],
    ts_refinement: str = TS_REFINEMENTS[0],
) -> StringResult:
    """Estimate an MEP with the growing string method (Peters et al., 2004).

    Each fragment grows by one node whenever the perpendicular gradient at its
    frontier drops to ``growth_tolerance`` (the indicator function of Eq. 7).
    ``max_frontier_steps`` is a practical safeguard the paper leaves to an
    adaptive tolerance: a frontier that has not met the tolerance for that
    many steps grows regardless. After joining, the string is relaxed until
    max |f⊥| < ``tolerance``. If ``max_iterations`` runs out first, any
    missing nodes are filled in on the spline and ``converged`` is False.

    ``callback``, if given, receives a :class:`GSMStep` before and after each
    growth and at every iteration once the fragments have joined.

    ``string_optimizer`` (one of ``GSM_STRING_OPTIMIZERS``) evolves the string;
    ``ts_refinement`` (one of ``TS_REFINEMENTS``) refines the final TS.
    """

    # --- Input validation -------------------------------------------------
    # Purpose: Validate requested dimensions and choices, normalize endpoints, and set the growth threshold.
    # Reason: The driver relies on compatible one-dimensional endpoints and supported optimization modes.
    if num_images < 4:
        raise ValueError("num_images must be at least 4")
    _check_choice(string_optimizer, GSM_STRING_OPTIMIZERS, "string optimizer")

    # Purpose: reactant_coords stores the normalized floating-point reactant endpoint.
    # Reason: Array shape checks and string construction require a predictable representation.
    reactant_coords = np.asarray(reactant, dtype=float)

    # Purpose: product_coords stores the normalized floating-point product endpoint.
    # Reason: It must match reactant_coords and anchor the other end of the string.
    product_coords = np.asarray(product, dtype=float)
    if reactant_coords.shape != product_coords.shape or reactant_coords.ndim != 1:
        raise ValueError("reactant and product must be matching 1-D coordinates")
    if growth_tolerance is None:
        growth_tolerance = 10.0 * tolerance

    # --- Wrap evaluation functions -----
    # Purpose: Decorate energy and gradient callables so every downstream evaluation is counted.
    # Reason: Optimization, refinement, and reporting costs must all appear in the final metadata.

    # Purpose: energy stores the counted wrapper around the caller's energy function.
    # Reason: It remains in use through optimization, transition-state refinement, and final reporting.
    # Count every energy/gradient use (optimizer, TS refinement, reporting).
    energy = CountedFunction(energy)

    # Purpose: gradient stores the counted wrapper around the caller's gradient function.
    # Reason: It remains in use through optimization, transition-state refinement, and final reporting.
    gradient = CountedFunction(gradient)

    # --- Initial string: endpoints plus one frontier node per fragment -----
    # Purpose: Place one frontier beside each endpoint and initialize persistent driver state.
    # Reason: The paper begins with two automatically grown interior nodes before iterative growth.
    # With only two nodes the spline is the straight line, so the first two
    # frontiers are placed one spacing in from each end of it (the paper's
    # "first two interior nodes are automatically grown").
    # Purpose: reactant_count stores the number of nodes currently belonging to the reactant fragment.
    # Reason: It locates the left frontier, defines fragment slices, and contributes to joining detection.
    # Purpose: product_count stores the number of nodes currently belonging to the product fragment.
    # Reason: It locates the right fragment and combines with reactant_count to detect joining.
    # Nodes in the reactant / product fragment, each counting its endpoint.
    # Purpose: locate the two frontier nodes (reactant_count - 1 and
    # reactant_count), split the string into per-fragment line-search groups,
    # and detect joining (the counts sum to num_images).
    reactant_count, product_count = 2, 2
    # Purpose: nodes stores the current ordered coordinates of both string fragments.
    # Reason: Every growth, force, evolution, and final-result operation reads or replaces this geometry.
    # Current string, shape (reactant_count + product_count, dim): reactant
    # fragment first, then product fragment.
    # Purpose: the object being grown and evolved; returned as the final path.
    nodes = reparametrize(np.vstack((reactant_coords, product_coords)), reactant_count, product_count, num_images)
    # Purpose: total_node_additions_needed stores the full progress target for callback reporting.
    # Reason: It remains constant while the current addition count increases during growth.
    # Node additions needed to join.
    # Purpose: lets callbacks show growth progress (addition k of total_node_additions_needed).
    total_node_additions_needed = num_images - 2
    # Purpose: optimizer_state stores adaptive steps and momentum across evolution iterations.
    # Reason: Stateful optimizers improve using prior motion but must reset after growth.
    # Optimizer memory (step lengths, MD velocity, FIRE parameters).
    # Purpose: lets evolve_string adapt its step across iterations; reset
    # whenever the node count changes.
    optimizer_state = _OptimizerState(step_size)
    # Purpose: evolution_steps_since_growth stores each fragment's waiting time since its last addition.
    # Reason: It activates the max_frontier_steps safeguard independently on both sides.
    # Per fragment (reactant, product): evolution steps since it last grew.
    # Purpose: force a frontier to grow once it reaches max_frontier_steps,
    # even if it has not met growth_tolerance.
    evolution_steps_since_growth = [0, 0]
    # Purpose: converged stores whether the joined path currently satisfies the force tolerance.
    # Reason: It controls termination and is included in the final StringResult.
    # Whether max |f⊥| < tolerance on the fully grown string.
    # Purpose: stops the loop and is reported in the result.
    converged = False
    # Purpose: max_force_norm stores the most recent maximum interior perpendicular-gradient magnitude.
    # Reason: It is reported to callbacks and retained as the final convergence diagnostic.
    # Max |f⊥| over interior nodes at the last check.
    # Purpose: reported to callbacks each step and stored in the result.
    max_force_norm = np.inf
    # Purpose: completed_iteration stores the number of evolution/reparametrization cycles completed.
    # Reason: It enforces max_iterations and labels callback snapshots.
    # Completed evolution steps.
    # Purpose: enforces max_iterations and is reported in callbacks and the result.
    completed_iteration = 0

    # --- Emit callback snapshots -----
    # Purpose: Package a defensive copy of the current stage and cumulative evaluation counts.
    # Reason: Centralizing callback construction keeps stage metadata consistent and protects driver arrays.
    def report(stage: str, images, previous, new_node=None, max_perp=None, counts=None) -> None:
        # --- Skip absent callback -----
        # Purpose: Return immediately when the caller did not request progress reporting.
        # Reason: Avoiding snapshot construction and copies reduces overhead in the common no-callback case.
        if callback is None:
            return

        # Purpose: left and right store the fragment sizes associated with this exact snapshot.
        # Reason: Growth events may describe an intermediate count different from the current closure state.
        left, right = counts if counts is not None else (reactant_count, product_count)

        # Purpose: joined stores whether the snapshot contains the requested total number of nodes.
        # Reason: reactant_count should be hidden from callback users after the fragment boundary disappears.
        joined = left + right >= num_images

        # --- Build and dispatch snapshot -----
        # Purpose: Copy stage geometry and metadata into GSMStep and invoke the caller's callback.
        # Reason: Callbacks must not mutate arrays still owned by the optimization driver.
        callback(
            GSMStep(
                stage=stage,
                addition=left + right - 3,
                total_additions=total_node_additions_needed,
                iteration=completed_iteration + 1,
                images=images.copy(),
                previous_images=None if previous is None else previous.copy(),
                reactant_count=None if joined else left,
                new_node=new_node,
                max_perpendicular_gradient=max_perp,
                **evaluation_counts(energy, gradient),
            )
        )

    # --- Locate active frontiers -----
    # Purpose: Return the two fragment-boundary indices until the string has joined.
    # Reason: Frontier nodes require spline tangents during growth but ordinary tangents afterwards.
    def frontiers() -> tuple[int, ...]:
        # --- Test joined state -----
        # Purpose: Suppress special frontier handling once all requested nodes exist.
        # Reason: A joined string no longer contains a vacant gap or fragment boundary.
        # Frontier node indices while growing; none once the fragments joined.
        if reactant_count + product_count >= num_images:
            return ()

        # --- Return boundary indices -----
        # Purpose: Identify the last reactant node and first product node.
        # Reason: These are the two nodes adjacent to the vacant gap.
        return (reactant_count - 1, reactant_count)

    # --- Run growth and relaxation loop -----
    # Purpose: Repeatedly evaluate forces, grow eligible frontiers, evolve nodes, and restore parametrization.
    # Reason: Alternating these operations builds the path and then relaxes the joined string to convergence.
    while True:
        # --- Evaluate current convergence state -----
        # Purpose: Compute projected gradients and determine whether the current path is complete and relaxed.
        # Reason: Growth and evolution decisions both depend on current forces and joining status.

        # Purpose: perpendicular stores current normal gradients for all nodes.
        # Reason: It drives growth tests, convergence checks, and the next evolution step.
        # Purpose: tau stores current path tangents for all nodes.
        # Reason: Velocity-based evolution removes motion along these directions.
        perpendicular, tau = perpendicular_forces(nodes, energy, gradient, frontiers())

        # Purpose: joined stores whether the fragment counts have filled every requested node slot.
        # Reason: It selects between growth behavior and final whole-string relaxation.
        joined = reactant_count + product_count >= num_images
        converged, max_force_norm = is_converged(perpendicular, len(nodes), num_images, tolerance)

        # --- Stop completed or exhausted run -----
        # Purpose: Exit when convergence is reached or the evolution budget is consumed.
        # Reason: The driver must respect both the scientific criterion and the caller's iteration limit.
        if converged or completed_iteration >= max_iterations:
            break

        # --- Growth: check the frontiers (Eqs. 7 and 9) --------------------
        # Purpose: Add nodes to relaxed or overdue fragments while vacant slots remain.
        # Reason: GSM constructs the path progressively before applying joined-string relaxation.
        if not joined:
            # Purpose: overdue stores whether each fragment has waited the maximum allowed evolution steps.
            # Reason: It forces progress when a frontier does not meet growth_tolerance promptly.
            overdue = (
                evolution_steps_since_growth[0] >= max_frontier_steps,
                evolution_steps_since_growth[1] >= max_frontier_steps,
            )

            # Purpose: growths stores ordered descriptions of additions performed in this iteration.
            # Reason: The driver must reset side-specific counters and emit one callback per new node.
            nodes, reactant_count, product_count, growths = grow_string(
                nodes, perpendicular, reactant_count, product_count, num_images, growth_tolerance, overdue
            )

            # Purpose: growth iterates over each node addition made by grow_string.
            # Reason: Each addition has its own side, geometry, counts, and callback event.
            for growth in growths:
                evolution_steps_since_growth[growth.side] = 0
                report("grow", growth.after, growth.before, new_node=growth.after[growth.new_index].copy(),
                       counts=(growth.reactant_count, growth.product_count))

            # --- Refresh after growth -----
            # Purpose: Reset optimizer memory and recompute geometry-dependent forces after node insertion.
            # Reason: Reparametrization changes every node and invalidates prior gradients and velocities.
            if growths:
                optimizer_state.reset_after_growth()
                # The new nodes need gradients; reparametrization also moved the others.
                perpendicular, tau = perpendicular_forces(nodes, energy, gradient, frontiers())
                joined = reactant_count + product_count >= num_images
                max_force_norm = float(np.max(np.linalg.norm(perpendicular[1:-1], axis=1)))

        # --- Evolution and reparametrization ------------------------------
        # Purpose: Relax interior nodes along normal forces, then redistribute them at prescribed density.
        # Reason: Evolution improves the path while reparametrization prevents nodes from clustering or drifting.
        # While growing, each fragment gets its own line-search step (F_left, F_right).

        # Purpose: groups stores the node slices assigned independent line-search objectives.
        # Reason: Growing fragments relax separately, whereas a joined string uses one interior group.
        groups = [slice(1, -1)] if joined else [slice(1, reactant_count), slice(reactant_count, -1)]

        # Purpose: evolved stores the geometry after one optimizer step but before spacing correction.
        # Reason: It is reported separately and serves as reparametrization input.
        evolved = evolve_string(
            nodes, perpendicular, tau, energy, gradient, optimizer_state, string_optimizer, bounds, groups, frontiers()
        )
        report("evolve", evolved, nodes, max_perp=max_force_norm)
        nodes = reparametrize(evolved, reactant_count, product_count, num_images)
        report("reparametrize", nodes, evolved, max_perp=max_force_norm)

        # --- Advance iteration counters -----
        # Purpose: Record the completed evolution cycle and age unfinished fragment frontiers.
        # Reason: These counters enforce the global iteration budget and the growth safeguard.
        completed_iteration += 1
        if not joined:
            evolution_steps_since_growth[0] += 1
            evolution_steps_since_growth[1] += 1

    # --- Fill unfinished gap -----
    # Purpose: Produce a full-size output string if the iteration budget expires during growth.
    # Reason: Result construction expects a complete path even though it must remain marked unconverged.
    # Out of iterations before the fragments joined: fill the gap on the spline.
    if len(nodes) < num_images:
        nodes = reparametrize(nodes, num_images, 0, num_images)
        _, max_force_norm = is_converged(perpendicular_forces(nodes, energy, gradient)[0], num_images, num_images, tolerance)
        converged = False

    # --- Build transition-state result -----
    # Purpose: Evaluate the final path, select or refine its highest-energy image, and package diagnostics.
    # Reason: Callers need a standard StringResult containing both the path and transition-state estimate.
    # ``build_result_with_transition_state`` evaluates energies, picks the highest image as the TS
    # (optionally refining it to a nearby saddle point), and packages a
    # StringResult.
    # Purpose: result stores the completed GSM path, transition state, convergence data, and metadata.
    # Reason: It is enriched with optimizer and evaluation-count information before being returned.
    result = build_result_with_transition_state(
        nodes,
        energy,
        gradient,
        "GSM",
        converged,
        completed_iteration,
        max_force_norm,
        bounds,
        ts_refinement,
    )

    # --- Attach method metadata -----
    # Purpose: Record the chosen string optimizer and counted evaluation totals on the result.
    # Reason: Reproducibility and performance comparisons require both configuration and computational cost.
    result.metadata["string_optimizer"] = string_optimizer
    return record_evaluation_counts(result, energy, gradient)


# --- Public compatibility alias -----
# Purpose: Expose growing_string through the conventional run_gsm module-level name.
# Reason: Existing callers can invoke the algorithm without depending on its descriptive function name.
# Purpose: run_gsm stores the public alias referring to growing_string.
# Reason: The alias preserves a stable entry point without wrapping or changing behavior.
run_gsm = growing_string

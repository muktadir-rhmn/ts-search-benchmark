"""Freezing String Method (FSM) for two- or higher-dimensional potentials."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .string_utils import (
    FSM_NODE_OPTIMIZERS,
    TS_REFINEMENTS,
    CountedFunction,
    EnergyFunction,
    GradientFunction,
    StringResult,
    build_result_with_transition_state,
    evaluation_counts,
    record_evaluation_counts,
    minimize_in_perpendicular_hyperplane,
    redistribute,
    unit,
)


@dataclass
class FSMStep:
    """Snapshot of one FSM growth step, passed to ``freezing_string``'s callback."""

    step: int  # 0-based index of this image addition
    side: str  # "reactant" or "product": which front grew
    seed_image: NDArray[np.float64]  # starting point on the frontier-to-frontier line
    growth_direction: NDArray[np.float64]  # unit vector; the image may not move along it
    trust_radius: float  # max distance the image may move from its seed
    optimizer_trajectory: NDArray[np.float64]  # (k, dim) iterates, seed first, new image last
    new_image: NDArray[np.float64]  # the optimized, now frozen, image
    image_converged: bool
    reactant_side: NDArray[np.float64]  # frozen reactant front after this step
    product_side: NDArray[np.float64]  # frozen product front after this step
    # Running totals of energy/gradient use since the run started.
    energy_calls: int = 0
    gradient_calls: int = 0
    energy_evaluations: int = 0
    gradient_evaluations: int = 0


FSMCallback = Callable[[FSMStep], None]

# Smallest trust radius (in units of nominal spacing) tried when a new image
# would move its front away from the opposite one; see the growth loop.
MIN_TRUST_RADIUS_FACTOR = 0.25


def freezing_string(
    energy: EnergyFunction,  # potential energy V(q) to search on
    gradient: GradientFunction,  # dV/dq, used by the optimizer
    reactant: ArrayLike,  # 1-D coordinates of the reactant minimum (fixed endpoint)
    product: ArrayLike,  # 1-D coordinates of the product minimum (fixed endpoint)
    *,
    num_images: int = 25,  # images in the returned path; also sets nominal spacing
    bounds: tuple[tuple[float, float], ...] | None = None,  # per-coordinate (min, max) box for the optimizer
    optimization_steps: int = 120,  # max optimizer iterations per new image
    tolerance: float = 1.0e-3,  # optimizer convergence tolerance (SLSQP ftol = 0.01 * tolerance)
    trust_radius_factor: float = 5.0,  # initial trust radius, in units of nominal spacing
    max_additions: int | None = None,  # cap on images added; None means 4 * num_images
    callback: FSMCallback | None = None,  # called with an FSMStep after each image is frozen
    node_optimizer: str = FSM_NODE_OPTIMIZERS[0],  # freezing-step minimizer, see FSM_NODE_OPTIMIZERS
    ts_refinement: str = TS_REFINEMENTS[0],  # saddle search from the highest image, see TS_REFINEMENTS
) -> StringResult:
    """Estimate an MEP with the freezing string method.

    Two fronts grow from reactant and product. Each new image is locally
    minimized perpendicular to its growth tangent and is then frozen. The
    completed path is redistributed only for smooth plotting; frozen images are
    not globally relaxed. ``trust_radius_factor`` controls how far a new image
    may move normal to its initial growth direction, in units of nominal image
    spacing. If an optimized image ends up farther from the opposite frontier
    than its predecessor (it slid back down toward its own basin), it is
    re-optimized with half the trust radius, down to
    ``MIN_TRUST_RADIUS_FACTOR`` spacings, so the fronts cannot wander.

    Fronts keep growing (alternating sides) until the distance between the two
    frontier images is at most the nominal image spacing. ``max_additions``
    (default ``4 * num_images``) caps the number of new images in case the
    fronts fail to close; hitting the cap marks the result as not converged.

    ``callback``, if given, is called with an :class:`FSMStep` after every new
    image is frozen (e.g. to animate the run). Leave it as ``None`` to skip
    recording entirely.
    """

    # --- Input validation -------------------------------------------------
    if num_images < 3:
        raise ValueError("num_images must be at least 3")
    reactant_coords = np.asarray(reactant, dtype=float)
    product_coords = np.asarray(product, dtype=float)
    if reactant_coords.shape != product_coords.shape or reactant_coords.ndim != 1:
        raise ValueError("reactant and product must be matching 1-D coordinates")
    # Count every energy/gradient use (optimizer, TS refinement, reporting).
    energy = CountedFunction(energy)
    gradient = CountedFunction(gradient)

    # --- Set up the two growing fronts ------------------------------------
    # Each side is a list of frozen images, starting at its endpoint; the last
    # element is that side's current frontier image.
    reactant_side = [reactant_coords.copy()]
    product_side = [product_coords.copy()]
    # Target distance between consecutive images, based on the straight-line
    # reactant -> product distance. Used both as the growth step and as the
    # "fronts have met" threshold.
    nominal_spacing = np.linalg.norm(product_coords - reactant_coords) / (num_images - 1)
    if max_additions is None:
        max_additions = 4 * num_images
    image_converged_flags: list[bool] = []  # optimizer success flag per new image
    total_iterations = 0  # summed optimizer iterations, reported in the result
    images_added = 0
    fronts_met = False

    # --- Growth loop -------------------------------------------------------
    # Alternate sides: even additions extend the reactant front, odd ones the
    # product front. Each new image is placed toward the opposite frontier.
    while images_added < max_additions:
        # Determine which side to grow and the current frontiers.
        grow_reactant_side = images_added % 2 == 0
        growing_frontier = reactant_side[-1] if grow_reactant_side else product_side[-1]
        opposite_frontier = product_side[-1] if grow_reactant_side else reactant_side[-1]
        
        # Stop once the gap between frontiers is no larger than one spacing.
        frontier_gap = np.linalg.norm(opposite_frontier - growing_frontier)
        if frontier_gap <= nominal_spacing:
            fronts_met = True
            break
        growth_direction = unit(opposite_frontier - growing_frontier)
        # Near the end, split the remaining gap so no near-duplicate image is
        # placed right next to the opposite frontier. This prevents overshotting.
        step_length = min(nominal_spacing, 0.5 * frontier_gap)
        # Seed the new image on the straight line joining the two frontiers.
        seed_image = growing_frontier + step_length * growth_direction
        # Freezing step: minimize the energy in the hyperplane perpendicular to
        # ``growth_direction`` through ``seed_image`` (``node_optimizer``, SLSQP by default),
        # restricted to a trust radius around the seed. Once accepted, the
        # image is never moved again.
        trust_radius = trust_radius_factor * nominal_spacing
        while True:
            # Only record optimizer iterates when someone is listening.
            trajectory = None if callback is None else [seed_image.copy()]
            # Why only perpendicular to ``growth_direction``?
            # - The MEP is defined by a vanishing *perpendicular* gradient: at
            #   every point on it the force points along the path (ie, 
            #   the gradient has no component perpendicular to the path, 
            #  so all the force points along it). Minimizing
            #   in the hyperplane drives exactly that component to zero, so the
            #   image lands at the valley floor of its "slice" of the surface.
            # - Not along the growth direction: on the way up to the TS, the
            #   energy along the path increases, so minimizing along the
            #   tangent would slide the image straight back downhill onto its
            #   own frontier (or basin), undoing the step. Past the TS it would
            #   rush toward the opposite minimum. Either way the image would
            #   never sit on the uphill part of the path, where the TS is.
            # - Not along a mixed (oblique) direction: any direction with a
            #   tangent component inherits that downhill drift, so the image
            #   creeps backward and the spacing between images is lost.
            # Fixing the tangent coordinate (via ``step_length``) and relaxing
            # only the perpendicular ones separates "where along the path"
            # (our choice) from "where across the valley" (the optimizer's).
            #
            # Why a trust radius on top of the hyperplane constraint?
            # - The hyperplane is unbounded: it fixes only the tangent
            #   coordinate, so the image is free to travel arbitrarily far
            #   sideways. Far from the seed the hyperplane can cross a
            #   different valley (or the reactant/product basin itself), and
            #   the unconstrained minimum there is not on *this* path.
            # - The straight-line seed is a local guess; the perpendicular
            #   minimum we want is the nearby valley floor, not the global
            #   minimum of the whole slice. Bounding the step keeps the
            #   optimizer in that neighbourhood.
            # - Keeping consecutive images close preserves the string's
            #   continuity, so the next growth direction (frontier to
            #   frontier) remains a sensible estimate of the path tangent.
            new_image, image_converged, image_iterations = minimize_in_perpendicular_hyperplane(
                seed_image,
                growth_direction,
                energy,
                gradient,
                bounds,
                optimization_steps,
                # Stopping criterion for the perpendicular relaxation (passed to
                # SLSQP as ftol = 0.01 * tolerance): stop once the energy
                # change between iterations falls below it. This decides how
                # precisely the image settles onto the valley floor before it
                # is frozen; frozen images are never re-relaxed, so a loose
                # value leaves them off the MEP, while a tight one costs more
                # iterations (and may hit ``optimization_steps``, marking the
                # image unconverged).
                tolerance,
                trust_radius,
                trajectory,
                node_optimizer,
            )
            total_iterations += image_iterations
            # The perpendicular hyperplane can cut across a whole valley, so a
            # wide trust radius lets the image slide back into its own basin.
            # That widens the gap between fronts and makes them chase each
            # other for many extra steps. Accept the image only if it brings
            # the fronts closer; otherwise retry with a tighter trust radius.
            closes_gap = np.linalg.norm(opposite_frontier - new_image) < frontier_gap
            if closes_gap or trust_radius <= MIN_TRUST_RADIUS_FACTOR * nominal_spacing:
                break
            trust_radius *= 0.5
        image_converged_flags.append(image_converged)
        # Freeze the image by appending it to its front.
        if grow_reactant_side:
            reactant_side.append(new_image)
        else:
            product_side.append(new_image)
        if callback is not None:
            callback(
                FSMStep(
                    step=images_added,
                    side="reactant" if grow_reactant_side else "product",
                    seed_image=seed_image,
                    growth_direction=growth_direction,
                    trust_radius=float(trust_radius),
                    optimizer_trajectory=np.vstack(trajectory + [new_image]),
                    new_image=new_image,
                    image_converged=image_converged,
                    reactant_side=np.array(reactant_side),
                    product_side=np.array(product_side),
                    **evaluation_counts(energy, gradient),
                )
            )
        images_added += 1
    else:
        # while/else: runs only if the loop exhausted ``max_additions`` without
        # ``break``. Re-check whether the final addition happened to close the gap.
        fronts_met = np.linalg.norm(product_side[-1] - reactant_side[-1]) <= nominal_spacing

    # --- Assemble the path -------------------------------------------------
    # Join reactant front with the reversed product front so the path runs
    # reactant -> product, then resample to exactly ``num_images`` points
    # evenly spaced in arc length (linear interpolation; endpoints kept). This
    # is cosmetic: the resampled points are not re-optimized.
    frozen_images = np.vstack((reactant_side, np.asarray(product_side[::-1])))
    path_images = redistribute(frozen_images, num_images)

    # FSM convergence refers to the constrained local optimization of each new
    # node. Also report the final global residual as a transparent path-quality
    # indicator; unlike GSM, FSM does not subsequently relax frozen nodes.
    from .string_utils import perpendicular_gradients

    _, perpendicular_grads, _ = perpendicular_gradients(path_images, energy, gradient)
    # Endpoints are excluded: they are fixed minima, not part of the search.
    max_perpendicular_gradient = float(
        np.max(np.linalg.norm(perpendicular_grads[1:-1], axis=1), initial=0.0)
    )
    # ``build_result_with_transition_state`` evaluates energies, picks the highest image as the TS
    # (optionally refining it to a nearby saddle point), and packages a
    # StringResult. Converged = fronts closed AND every freeze step succeeded.
    result = build_result_with_transition_state(
        path_images,
        energy,
        gradient,
        "FSM",
        fronts_met and all(image_converged_flags),
        total_iterations,
        max_perpendicular_gradient,
        bounds,
        ts_refinement,
    )
    result.metadata["node_optimizer"] = node_optimizer
    return record_evaluation_counts(result, energy, gradient)


run_fsm = freezing_string

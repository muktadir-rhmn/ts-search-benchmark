"""Cost and accuracy tables for every FSM / GSM optimizer combination.

Run with ``python benchmark.py [num_images]``. For each surface it prints one
table for FSM (node optimizer x TS refinement) and one for GSM (string
optimizer x TS refinement) with the number of energy/gradient calls (and the
points they evaluated) and the error of the reported transition state against
the exact saddle point.
"""

from __future__ import annotations

import sys
import time

import numpy as np
from scipy.optimize import minimize_scalar, root

from surfaces import leps
from surfaces import muller_brown
from algorithms.fsm import freezing_string
from algorithms.gsm import growing_string
from algorithms.string_utils import FSM_NODE_OPTIMIZERS, GSM_STRING_OPTIMIZERS, TS_REFINEMENTS


def exact_saddle(gradient, guess) -> np.ndarray:
    """Tight root solve of grad E = 0 from a guess next to the known saddle."""

    solved = root(lambda q: np.asarray(gradient(q), dtype=float), guess, tol=1.0e-12)
    if np.linalg.norm(gradient(solved.x)) > 1.0e-8:
        raise RuntimeError(f"reference saddle solve failed: {solved.message}")
    return solved.x


def leps_saddle_guess() -> np.ndarray:
    # The collinear LEPS saddle is symmetric (r_AB = r_BC) and is the energy
    # minimum along that diagonal.
    r = minimize_scalar(lambda r: leps.energy([r, r]), bounds=(0.6, 1.5), method="bounded").x
    return np.array([r, r])


SURFACES = {
    "Muller-Brown": {
        "module": muller_brown,
        # The highest saddle on the MA -> MC -> MB path.
        "guess": np.array([-0.822, 0.624]),
    },
    "LEPS (collinear)": {
        "module": leps,
        "guess": leps_saddle_guess(),
    },
}


def run_one(algorithm, module, reference, num_images, **options) -> dict:
    started = time.perf_counter()
    try:
        result = algorithm(
            module.energy,
            module.gradient,
            module.DEFAULT_REACTANT,
            module.DEFAULT_PRODUCT,
            bounds=module.PLOT_BOUNDS,
            num_images=num_images,
            **options,
        )
    except Exception as error:  # report the failure in the table instead of aborting
        return {"error": f"{type(error).__name__}: {error}"}
    ts = result.transition_state
    return {
        # Cost. "calls" counts function invocations; "pts" counts the
        # geometries evaluated, because one call may evaluate many points:
        # energy([x, y]) is 1 call / 1 point, while energy(images) on a
        # (25, 2) string is 1 call / 25 points (see CountedFunction in
        # string_utils.py). Points are the fair cost measure when every
        # geometry is an expensive calculation (e.g. quantum chemistry);
        # GSM re-evaluates the whole string every iteration, so its point
        # counts are far higher than its call counts.
        "e_calls": result.energy_calls,
        "e_pts": result.energy_evaluations,
        "g_calls": result.gradient_calls,
        "g_pts": result.gradient_evaluations,
        # Accuracy against the exact saddle TS* from exact_saddle().
        # dist: Euclidean distance |TS - TS*| in coordinate space.
        "dist": float(np.linalg.norm(ts - reference)),
        # d_energy: E(reported TS) - E(TS*).
        #   ~0       -> the reported TS is the true saddle (machine precision).
        #   negative -> barrier underestimated: the TS lies slightly along the
        #               path, which falls away on both sides of the saddle
        #               (typical of the highest image with no refinement).
        #   positive -> barrier overestimated: the TS lies slightly across
        #               the valley, which rises on both sides of the saddle.
        "d_energy": result.transition_state_energy - float(module.energy(reference)),
        # grad_norm: |grad E| at the reported TS; zero at a true stationary point.
        "grad_norm": float(np.linalg.norm(module.gradient(ts))),
        "refined": result.metadata.get("transition_state_source") == "stationary point anchored to path",
        "converged": result.converged,
        "seconds": time.perf_counter() - started,
    }


HEADER = (
    f"{'optimizer':<22}{'TS refinement':<16}{'E calls':>9}{'E pts':>9}{'G calls':>9}{'G pts':>9}"
    f"{'|TS-TS*|':>11}{'E-E*':>11}{'|grad|':>10}  {'saddle?':<8}{'conv':<6}{'time/s':>7}"
)


def format_row(optimizer: str, refinement: str, row: dict) -> str:
    if "error" in row:
        return f"{optimizer:<22}{refinement:<16}  FAILED: {row['error']}"
    return (
        f"{optimizer:<22}{refinement:<16}{row['e_calls']:>9}{row['e_pts']:>9}{row['g_calls']:>9}{row['g_pts']:>9}"
        f"{row['dist']:>11.2e}{row['d_energy']:>11.2e}{row['grad_norm']:>10.1e}  "
        f"{'yes' if row['refined'] else 'no':<8}{'yes' if row['converged'] else 'no':<6}{row['seconds']:>7.2f}"
    )


def main() -> None:
    num_images = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    methods = (
        ("FSM", freezing_string, "node_optimizer", FSM_NODE_OPTIMIZERS),
        ("GSM", growing_string, "string_optimizer", GSM_STRING_OPTIMIZERS),
    )
    for surface_name, spec in SURFACES.items():
        module = spec["module"]
        reference = exact_saddle(module.gradient, spec["guess"])
        print(f"\n=== {surface_name}: exact TS* = {np.array2string(reference, precision=6)}, "
              f"E* = {module.energy(reference):.6f}, {num_images} images ===")
        for name, algorithm, keyword, optimizers in methods:
            print(f"\n{name}\n{HEADER}\n{'-' * len(HEADER)}")
            for optimizer in optimizers:
                for refinement in TS_REFINEMENTS:
                    row = run_one(algorithm, module, reference, num_images,
                                  **{keyword: optimizer, "ts_refinement": refinement})
                    print(format_row(optimizer, refinement, row), flush=True)
    print(
        "\nE/G calls = energy/gradient function invocations; pts = points evaluated "
        "(one vectorized call on a whole string counts every image).\n"
        "|TS-TS*| = distance of the reported TS from the exact saddle; E-E* = its energy error; "
        "|grad| = gradient norm at the reported TS.\n"
        "saddle? = the TS refinement was accepted (otherwise the highest path image is reported); "
        "conv = the string method's own convergence flag."
    )


if __name__ == "__main__":
    main()

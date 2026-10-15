"""Lightweight numerical checks; run with ``python -m unittest -v``."""

from __future__ import annotations

import unittest

import numpy as np

from surfaces import leps
from surfaces import muller_brown
from algorithms.fsm import freezing_string
from algorithms.gsm import growing_string
from algorithms.string_utils import FSM_NODE_OPTIMIZERS, GSM_STRING_OPTIMIZERS, TS_REFINEMENTS


def finite_difference(function, point, step=1.0e-6):
    result = np.zeros_like(point, dtype=float)
    for i in range(len(point)):
        delta = np.zeros_like(point, dtype=float)
        delta[i] = step
        result[i] = (function(point + delta) - function(point - delta)) / (2.0 * step)
    return result


class PotentialTests(unittest.TestCase):
    def test_muller_brown_gradient(self):
        point = np.array([-0.2, 0.8])
        np.testing.assert_allclose(muller_brown.gradient(point), finite_difference(muller_brown.energy, point), rtol=2e-6)

    def test_collinear_leps_gradient(self):
        point = np.array([1.1, 1.4])
        np.testing.assert_allclose(leps.gradient(point), finite_difference(leps.energy, point), rtol=2e-6)

    def test_general_and_collinear_leps_agree(self):
        point = np.array([1.2, 1.5])
        self.assertAlmostEqual(leps.energy(point), leps.leps_potential([1.2, 1.5, 2.7]))


class StringTests(unittest.TestCase):
    def test_algorithms_return_finite_paths(self):
        cases = (
            (muller_brown, 0.001),
            (leps, 0.002),
        )
        for potential, step in cases:
            for algorithm in (freezing_string, growing_string):
                with self.subTest(potential=potential.__name__, algorithm=algorithm.__name__):
                    kwargs = dict(num_images=11, bounds=potential.PLOT_BOUNDS)
                    if algorithm is growing_string:
                        kwargs["max_iterations"] = 80
                        kwargs["step_size"] = step
                    result = algorithm(
                        potential.energy, potential.gradient,
                        potential.DEFAULT_REACTANT, potential.DEFAULT_PRODUCT, **kwargs,
                    )
                    self.assertEqual(result.images.shape, (11, 2))
                    self.assertTrue(np.all(np.isfinite(result.energies)))
                    self.assertTrue(np.isfinite(result.transition_state_energy))
                    self.assertLess(
                        np.min(np.linalg.norm(result.images - result.transition_state, axis=1)),
                        1.0e-12,
                        "the reported TS must be an image on the returned pathway",
                    )
                    self.assertEqual(
                        result.transition_state_energy,
                        float(np.max(result.energies)),
                        "the reported TS must be the highest-energy path image",
                    )


    def test_every_optimizer_choice_finds_the_saddle(self):
        """All selectable optimizers reach the same TS on both surfaces."""

        saddles = {"surfaces.muller_brown": (-0.822002, 0.624313), "surfaces.leps": (0.925438, 0.925438)}
        for potential in (muller_brown, leps):
            args = (potential.energy, potential.gradient, potential.DEFAULT_REACTANT, potential.DEFAULT_PRODUCT)
            kwargs = dict(num_images=25, bounds=potential.PLOT_BOUNDS)
            runs = [("FSM", o, freezing_string(*args, node_optimizer=o, **kwargs)) for o in FSM_NODE_OPTIMIZERS]
            runs += [("GSM", o, growing_string(*args, string_optimizer=o, **kwargs)) for o in GSM_STRING_OPTIMIZERS]
            runs += [
                ("TS", o, growing_string(*args, string_optimizer="FIRE", ts_refinement=o, **kwargs))
                for o in TS_REFINEMENTS if o != "None"
            ]
            for stage, option, result in runs:
                with self.subTest(potential=potential.__name__, stage=stage, option=option):
                    self.assertTrue(result.converged)
                    self.assertEqual(result.metadata["transition_state_source"], "stationary point anchored to path")
                    np.testing.assert_allclose(result.transition_state, saddles[potential.__name__], atol=1e-5)

    def test_unknown_optimizer_is_rejected(self):
        with self.assertRaises(ValueError):
            growing_string(muller_brown.energy, muller_brown.gradient, muller_brown.DEFAULT_REACTANT,
                           muller_brown.DEFAULT_PRODUCT, string_optimizer="Adam")


if __name__ == "__main__":
    unittest.main()

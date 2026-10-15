# Freezing and Growing String Methods

This project demonstrates minimum-energy-path (MEP) and transition-state (TS)
estimation on the Müller–Brown surface and a collinear H + H2 LEPS surface.

## Install and run

```bash
python -m pip install -r requirements.txt
python test.py
```

In the GUI, choose a surface, enable FSM and/or GSM, choose an odd number of
images, and click **Run**. The colored line is the estimated MEP and the star is
the highest-energy path image (the TS estimate). Energy/barrier values appear at
bottom left. The GUI identifies itself as version **1.7.0** in the
plot title; if that version is absent, an older copy is being run.

With **Animate** ticked, choose **Play** to replay the iterations on a timer
(speed set by *Frame delay*) or **Step** to advance one iteration per click of
**Next ▸** (**◂ Back** goes back one). Switching between Play and Step while an
animation is on screen pauses or resumes it; **Stop** skips to the final paths.

For clarity at sharp FSM turns, the GUI draws a shape-preserving interpolation
through all discrete images and uses a hollow TS star. The colored TS node and
path remain visible through the marker. This interpolation is display-only; it
does not alter the path energies or string calculation.

Three combo boxes choose the optimizers (defaults are the first entries):

| Combo box | Options | Problem solved |
|---|---|---|
| FSM node optimizer | SLSQP, trust-constr, L-BFGS-B (subspace), Brent (1-D line) | minimize each new image in the hyperplane ⊥ its growth direction |
| GSM string optimizer | Quadratic line search (Peters et al.), Steepest descent, FIRE, Quick-min | relax every node's perpendicular gradient to zero |
| TS refinement | Root (hybr), Newton-Raphson, P-RFO, Dimer, None | locate the saddle starting from the highest image |

The same choices are the `node_optimizer`, `string_optimizer` and
`ts_refinement` keyword arguments of `freezing_string` / `growing_string`; the
option lists are `FSM_NODE_OPTIMIZERS`, `GSM_STRING_OPTIMIZERS` and
`TS_REFINEMENTS` in `algorithms/string_utils.py`. "Brent (1-D line)" exists only for 2-D
surfaces, where the perpendicular hyperplane is a line.

Run the numerical checks with:

```bash
python -m unittest -v
```

## API example

```python
from surfaces import muller_brown as mb
from algorithms.gsm import growing_string

result = growing_string(
    mb.energy,
    mb.gradient,
    mb.DEFAULT_REACTANT,
    mb.DEFAULT_PRODUCT,
    bounds=mb.PLOT_BOUNDS,
    num_images=25,
)
print(result.transition_state, result.transition_state_energy)
```

`leps.leps_potential()` accepts all three distances `(r_AB, r_BC, r_AC)`.
`leps.leps_collinear()` imposes `r_AC = r_AB + r_BC`; its aliases `energy` and
`gradient` are what the generic FSM/GSM routines and GUI use.

## Method notes

- FSM grows from both ends. Each new image is optimized only perpendicular to
  its growth direction inside a local trust region and then frozen. Its
  `converged` flag describes those local constrained optimizations; the reported
  `max_perpendicular_gradient` may remain appreciable because frozen nodes are
  deliberately not globally relaxed.
- GSM follows Peters et al., J. Chem. Phys. 120, 7877 (2004). Two fragments
  grow from the endpoints; a fragment gains a node when the perpendicular
  gradient at its frontier drops below `growth_tolerance` (or after
  `max_frontier_steps` steps). Every iteration evolves all interior nodes along
  the normal force and reparametrizes along a cubic spline: equal spacing in
  each fragment with a vacant gap between them, uniform once they join.
  Frontier nodes use spline tangents; other nodes use upwind tangents, which
  stay stable at sharp bends where spline tangents let kinks grow.
- After finding the MEP, both routines attempt a local stationary-point solve
  from the highest-energy image. A candidate is accepted only if it stays near
  the string, lies in bounds, and has a negative Hessian eigenvalue. An accepted
  saddle replaces the nearest interior image, so the reported and plotted TS is
  guaranteed to lie on the returned pathway.

These compact implementations are intended for teaching and small model
potentials, not production electronic-structure calculations.
# ts-search-benchmark

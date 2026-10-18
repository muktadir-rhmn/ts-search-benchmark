"""
Regression tests for pyGSM issue #47: molecular fragments whose atom indices
are interleaved (not grouped into contiguous per-fragment ranges).

These build PrimitiveInternalCoordinates directly (Topology.build_topology ->
PrimitiveInternalCoordinates.from_options), without any electronic structure
level of theory, following the lightweight construction pattern used in
test_basic_mecp.py.
"""

from collections import Counter

import numpy as np
import pytest

from pyGSM.coordinate_systems.primitive_internals import PrimitiveInternalCoordinates
from pyGSM.coordinate_systems.topology import Topology
from pyGSM.utilities import block_matrix, elements

ELEMENT_TABLE = elements.ElementData()


def _build_two_diatomics(perm):
    """Two carbon-carbon 'molecules' (bonded pairs), far apart from each
    other, with atoms placed according to `perm`: perm[new_idx] = base_idx,
    where the base (unpermuted) ordering is contiguous by fragment
    (molecule A = atoms 0,1; molecule B = atoms 2,3).
    """
    base_xyz = np.array([
        [0.0, 0.0, 0.0],
        [1.3, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [11.3, 0.0, 0.0],
    ])
    xyz = base_xyz[perm]
    atoms = [ELEMENT_TABLE.from_symbol('C') for _ in perm]
    return xyz, atoms


def _build_primitives(xyz, atoms):
    topology = Topology.build_topology(xyz, atoms, bondlistfile=None)
    return PrimitiveInternalCoordinates.from_options(
        xyz=xyz,
        atoms=atoms,
        connect=False,
        addtr=True,
        addcart=False,
        topology=topology,
    )


def _block_windows_partition_atoms(block_info, natoms):
    windows = sorted((info[0], info[1]) for info in block_info)
    covered = []
    for sa, ea in windows:
        covered += list(range(sa, ea))
    return covered == list(range(natoms))


def _no_duplicate_primitives(internals):
    counts = Counter(str(p) for p in internals)
    return all(v == 1 for v in counts.values())


# Interleaved: molecule A = atoms {0, 2}, molecule B = atoms {1, 3}.
INTERLEAVED_PERM = [0, 2, 1, 3]
CONTIGUOUS_PERM = [0, 1, 2, 3]


def test_get_hybrid_indices_no_crash_interleaved():
    """Originally reported crash: get_hybrid_indices raised a bare
    RuntimeError when fragment atom indices were interleaved."""
    xyz, atoms = _build_two_diatomics(INTERLEAVED_PERM)
    prims = _build_primitives(xyz, atoms)
    assert prims.hybrid_idx_start_stop == []


def test_no_duplicate_primitives_interleaved():
    """reorderPrimsByFrag used to match primitives to a fragment by a
    numeric (start, end) window; with interleaved fragments that window can
    overlap another fragment's window, silently duplicating primitives."""
    xyz, atoms = _build_two_diatomics(INTERLEAVED_PERM)
    prims = _build_primitives(xyz, atoms)
    assert _no_duplicate_primitives(prims.Internals)

    prims.reorderPrimsByFrag()
    assert _no_duplicate_primitives(prims.Internals)
    assert len(prims.Internals) == 14  # 7 primitives per diatomic * 2


def test_block_info_partitions_atoms_interleaved():
    """block_info's (start, end) windows must be non-overlapping and must
    tile [0, natoms) with no gaps, both after initial construction and
    after reorderPrimsByFrag (the live SE-GSM code path)."""
    xyz, atoms = _build_two_diatomics(INTERLEAVED_PERM)
    prims = _build_primitives(xyz, atoms)
    assert _block_windows_partition_atoms(prims.block_info, natoms=4)

    prims.reorderPrimsByFrag()
    assert _block_windows_partition_atoms(prims.block_info, natoms=4)


def test_copy_roundtrips_block_info():
    xyz, atoms = _build_two_diatomics(INTERLEAVED_PERM)
    prims = _build_primitives(xyz, atoms)
    copied = PrimitiveInternalCoordinates.copy(prims)
    assert copied.block_info == prims.block_info
    assert len(copied.Internals) == len(prims.Internals)


def test_wilsonB_permutation_equivalent():
    """The strongest correctness check: build the same physical molecule
    twice (fragment-contiguous vs. interleaved atom ordering) and verify
    that, after permuting back to a common atom ordering, primitive values
    and B-matrix rows match exactly. This validates that wilsonB's
    block-diagonal derivative assembly needs no special-casing once
    block_info windows are guaranteed to be true partitions."""
    xyz_c, atoms_c = _build_two_diatomics(CONTIGUOUS_PERM)
    xyz_i, atoms_i = _build_two_diatomics(INTERLEAVED_PERM)

    prims_c = _build_primitives(xyz_c, atoms_c)
    prims_i = _build_primitives(xyz_i, atoms_i)

    dense_c = block_matrix.full_matrix(prims_c.wilsonB(xyz_c))
    dense_i = block_matrix.full_matrix(prims_i.wilsonB(xyz_i))

    natoms = 4
    remapped_i = np.zeros_like(dense_i)
    for new_idx in range(natoms):
        old_idx = INTERLEAVED_PERM[new_idx]
        remapped_i[:, 3*old_idx:3*old_idx+3] = dense_i[:, 3*new_idx:3*new_idx+3]

    def key(prim, perm):
        return (type(prim).__name__, tuple(sorted(perm[a] for a in prim.atoms)))

    keys_c = [key(p, CONTIGUOUS_PERM) for p in prims_c.Internals]
    keys_i = [key(p, INTERLEAVED_PERM) for p in prims_i.Internals]
    assert sorted(keys_c) == sorted(keys_i)

    for row_i, k in enumerate(keys_i):
        row_c = keys_c.index(k)
        assert dense_c[row_c] == pytest.approx(remapped_i[row_i], abs=1e-8)

    values_c = {key(p, CONTIGUOUS_PERM): p.value(xyz_c) for p in prims_c.Internals}
    for p in prims_i.Internals:
        k = key(p, INTERLEAVED_PERM)
        assert p.value(xyz_i) == pytest.approx(values_c[k], abs=1e-8)


def test_three_fragments_deeply_interleaved():
    """Three fragments interleaved atom-by-atom (A={0,3}, B={1,4}, C={2,5})
    -- exercises chained window-merging, not just a single overlapping
    pair."""
    base_xyz = np.array([
        [0.0, 0.0, 0.0], [1.3, 0.0, 0.0],
        [20.0, 0.0, 0.0], [21.3, 0.0, 0.0],
        [40.0, 0.0, 0.0], [41.3, 0.0, 0.0],
    ])
    perm = [0, 2, 4, 1, 3, 5]
    xyz = base_xyz[perm]
    atoms = [ELEMENT_TABLE.from_symbol('C') for _ in perm]

    prims = _build_primitives(xyz, atoms)
    assert _no_duplicate_primitives(prims.Internals)
    assert _block_windows_partition_atoms(prims.block_info, natoms=6)
    assert len(prims.Internals) == 21  # 7 primitives per diatomic * 3

    prims.reorderPrimsByFrag()
    assert _no_duplicate_primitives(prims.Internals)
    assert _block_windows_partition_atoms(prims.block_info, natoms=6)

    # Must not crash (this is the hot path that previously broke on
    # interleaved multi-fragment blocks).
    dense = block_matrix.full_matrix(prims.wilsonB(xyz))
    assert dense.shape == (21, 18)

import numpy as np
from numpy.testing import assert_allclose

from pyzmat import ZmatUtils


def test_zmat_cartesian_round_trip(zmat_data):
    zmat, zmat_conn = zmat_data

    atoms = ZmatUtils.zmat_2_atoms(zmat, zmat_conn)
    recovered = ZmatUtils.atoms_2_zmat(atoms, zmat_conn)

    assert atoms.get_chemical_symbols() == [row[0] for row in zmat]
    for original, actual in zip(zmat, recovered):
        assert original[0] == actual[0]
        assert_allclose(
            [value for value in original[1:] if value is not None],
            [value for value in actual[1:] if value is not None],
            atol=1e-10,
        )


def test_analytical_b_matrix_matches_finite_difference(zmat_data):
    zmat, zmat_conn = zmat_data

    analytical = ZmatUtils.get_B_matrix(zmat, zmat_conn)
    finite_difference = ZmatUtils.get_fd_B_matrix(
        zmat, zmat_conn, db=1e-5, da=1e-3, dt=1e-3
    )

    assert analytical.shape == (9, 15)
    assert_allclose(analytical, finite_difference, atol=1e-8, rtol=1e-8)


def test_numba_curvature_tensor_matches_finite_difference(zmat_data):
    zmat, zmat_conn = zmat_data

    analytical = ZmatUtils.get_curvature_tensor_numba(zmat, zmat_conn)
    finite_difference = ZmatUtils.get_fd_curvature_tensor(
        zmat, zmat_conn, db=1e-4, da=1e-2, dt=1e-2
    )

    assert analytical.shape == (15, 9, 9)
    assert_allclose(analytical, analytical.transpose(0, 2, 1), atol=1e-12)
    assert_allclose(analytical, finite_difference, atol=1e-6, rtol=1e-6)


def test_fast_and_numba_curvature_implementations_agree(zmat_data):
    zmat, zmat_conn = zmat_data

    fast = ZmatUtils.get_curvature_tensor_fast(zmat, zmat_conn)
    numba = ZmatUtils.get_curvature_tensor_numba(zmat, zmat_conn)

    assert_allclose(fast, numba, atol=1e-11, rtol=1e-11)


def test_rmsd_is_invariant_to_rotation_and_translation(zmat_data):
    zmat, zmat_conn = zmat_data
    atoms = ZmatUtils.zmat_2_atoms(zmat, zmat_conn)
    transformed = atoms.copy()
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transformed.positions = transformed.positions @ rotation + [4.0, -2.0, 1.5]

    assert ZmatUtils.calculate_rmsd(atoms, transformed) < 1e-12

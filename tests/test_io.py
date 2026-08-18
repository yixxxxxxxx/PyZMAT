import numpy as np
from numpy.testing import assert_allclose
from ase.units import Bohr, Ha

from pyzmat import ParseUtils, PrintUtils, ZMatrix


def test_gaussian_input_write_parse_round_trip(molecule, tmp_path):
    output = tmp_path / "molecule.com"
    molecule.save_gaussian_com(output, title="round trip")

    loaded = ZMatrix.load_from_gaussian_input(output)

    assert loaded.zmat_conn == molecule.zmat_conn
    assert_allclose(
        np.asarray([row[1:] for row in loaded.zmat], dtype=object)[1:, 0].astype(float),
        np.asarray([row[1:] for row in molecule.zmat], dtype=object)[1:, 0].astype(float),
    )
    assert loaded.constraints.bonds == [(1, 1.42)]
    assert loaded.constraints.angles == [(2, 112.0)]
    assert loaded.constraints.dihedrals == [(3, 63.0)]


def test_orca_input_formatter_and_parser_round_trip(zmat_data, tmp_path):
    zmat, zmat_conn = zmat_data
    text = PrintUtils.print_orca_input(zmat, zmat_conn, geom_maxiter=0)
    output = tmp_path / "molecule.inp"
    output.write_text(text, encoding="utf-8")

    parsed_zmat, parsed_conn, constraints = ParseUtils.parse_orca_input(output)

    assert parsed_conn == zmat_conn
    assert constraints.bonds == constraints.angles == constraints.dihedrals == []
    for expected, actual in zip(zmat, parsed_zmat):
        assert expected[0] == actual[0]
        if expected[1] is not None:
            assert_allclose(actual[1], expected[1], atol=1e-6)
        if expected[2] is not None:
            assert_allclose(actual[2], expected[2], atol=1e-6)
        if expected[3] is not None:
            periodic_difference = (actual[3] - expected[3] + 180.0) % 360.0 - 180.0
            assert_allclose(periodic_difference, 0.0, atol=1e-6)


def test_json_round_trip_preserves_core_state(molecule, tmp_path):
    molecule.energy = -12.5
    molecule.forces = np.arange(9, dtype=float)
    molecule.hessian = np.eye(9)
    output = tmp_path / "molecule.json"

    molecule.dump_json(output)
    loaded = ZMatrix.load_json(output, load_hessian=True)

    assert loaded.zmat == molecule.zmat
    assert loaded.zmat_conn == molecule.zmat_conn
    assert loaded.constraints.bonds == molecule.constraints.bonds
    assert loaded.constraints.angles == molecule.constraints.angles
    assert loaded.constraints.dihedrals == molecule.constraints.dihedrals
    assert loaded.energy == molecule.energy
    assert_allclose(loaded.forces, molecule.forces)
    assert_allclose(loaded.hessian, molecule.hessian)


def test_pickle_round_trip(molecule, tmp_path):
    output = tmp_path / "molecule.pkl"

    molecule.save_pickle(output)
    loaded = ZMatrix.load_pickle(output)

    assert loaded.zmat == molecule.zmat
    assert loaded.zmat_conn == molecule.zmat_conn
    assert repr(loaded.constraints) == repr(molecule.constraints)


def test_parse_zmat_labels_and_references():
    text = """Z-matrix
C1
C2 C1
H1 C1 C2
O1 C2 C1 H1
"""

    assert ParseUtils.parse_zmat(text) == [
        ("C", None, None, None),
        ("C", 0, None, None),
        ("H", 0, 1, None),
        ("O", 1, 0, 2),
    ]


def test_orca_output_parser_selects_final_blocks(tmp_path):
    output = tmp_path / "orca.out"
    output.write_text(
        """====================
INPUT FILE
====================
| 1> * gzmt 0 1
| 2> C
| 3> C 1 1.42
| 4> H 1 1.09 2 112.0
| 5> O 1 1.36 2 108.0 3 63.0
| 6> *
====================
--------------------
CARTESIAN COORDINATES (ANGSTROEM)
--------------------
C 9.0 9.0 9.0
C 10.0 9.0 9.0

FINAL SINGLE POINT ENERGY -1.0
FINAL ENERGY EVALUATION AT THE STATIONARY POINT
--------------------
CARTESIAN COORDINATES (ANGSTROEM)
--------------------
C 0.0 0.0 0.0
C 1.42 0.0 0.0
H -0.407300 1.010983 0.0
O -0.420308 1.214732 0.563402

FINAL SINGLE POINT ENERGY -2.0
--------------------
CARTESIAN GRADIENT
--------------------
1 C : 0.1 0.2 0.3
2 C : 0.4 0.5 0.6
3 H : 0.7 0.8 0.9
4 O : 1.0 1.1 1.2
--------------------
""",
        encoding="utf-8",
    )

    zmat, conn, constraints, energy, forces = ParseUtils.parse_orca_output(output)

    assert conn == [
        ("C", None, None, None),
        ("C", 0, None, None),
        ("H", 0, 1, None),
        ("O", 0, 1, 2),
    ]
    assert constraints.bonds == constraints.angles == constraints.dihedrals == []
    assert_allclose(energy, -2.0 * Ha)
    assert_allclose(zmat[1][1], 1.42)
    factor = -Ha / Bohr
    expected_forces = np.asarray(
        [[0.1, -0.2, -0.3], [0.4, -0.5, -0.6],
         [0.7, -0.8, -0.9], [1.0, -1.1, -1.2]]
    ) * factor
    assert_allclose(forces, expected_forces, rtol=1e-7)

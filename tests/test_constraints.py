import pytest

from pyzmat import Constraints, ZMatrix


def test_constraints_normalise_values_and_apply_to_zmat(zmat_data):
    zmat, zmat_conn = zmat_data
    constraints = Constraints(
        bonds=[(1, 1.5)],
        angles=[(2, 248.0)],
        dihedrals=[(3, -30.0)],
    )

    molecule = ZMatrix(zmat, zmat_conn, constraints=constraints)

    assert molecule.zmat[1][1] == 1.5
    assert molecule.zmat[2][2] == 112.0
    assert molecule.zmat[3][3] == 330.0


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"bonds": "not-a-list"}, TypeError),
        ({"angles": [1]}, ValueError),
        ({"dihedrals": [2]}, ValueError),
        ({"bonds": [(1, "bad")]}, TypeError),
    ],
)
def test_constraints_reject_invalid_definitions(kwargs, error):
    with pytest.raises(error):
        Constraints(**kwargs)


def test_setting_and_clearing_constraints_updates_ase_constraints(molecule):
    molecule.constraints = Constraints(dihedrals=[(3, 45.0)])
    assert molecule.zmat[3][3] == 45.0
    assert len(molecule.ase_constraints.dihedrals) == 1

    molecule.clear_constraints()
    assert molecule.constraints.dihedrals == []

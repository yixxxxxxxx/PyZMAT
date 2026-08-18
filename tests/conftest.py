import pytest

from pyzmat import Constraints, ZMatrix


@pytest.fixture
def zmat_data():
    """A non-collinear molecule exercising every internal-coordinate type."""
    zmat = [
        ["C", None, None, None],
        ["C", 1.42, None, None],
        ["H", 1.09, 112.0, None],
        ["O", 1.36, 108.0, 63.0],
        ["H", 0.98, 104.5, 217.0],
    ]
    zmat_conn = [
        ("C", None, None, None),
        ("C", 0, None, None),
        ("H", 0, 1, None),
        ("O", 0, 1, 2),
        ("H", 3, 0, 1),
    ]
    return zmat, zmat_conn


@pytest.fixture
def molecule(zmat_data):
    zmat, zmat_conn = zmat_data
    constraints = Constraints(bonds=[1], angles=[2], dihedrals=[3])
    return ZMatrix(zmat, zmat_conn, constraints=constraints, name="fixture")

import copy
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scientific_support import molecule, energy_fd

pytestmark = pytest.mark.L1


def test_energy_force_hessian(zmat_data):
    m=molecule(zmat_data)
    original=copy.deepcopy(m.zmat)
    x=m.get_atoms().positions
    assert_allclose(m.get_energy(), (x*x).sum()+x.sum()**2+(x**4).sum(),rtol=1e-12,atol=1e-12)
    assert_allclose(m.get_forces_cart(), -(2*x+2*x.sum()+4*x**3).ravel(),rtol=1e-12,atol=1e-12)
    analytical=m.get_hessian()
    for step in (5e-5,1e-4,2e-4):
        f,h=energy_fd(m,step)
        assert_allclose(m.get_forces(),f,rtol=1e-6,atol=1e-6)
        assert_allclose(analytical,h,rtol=1e-5,atol=1e-4)
    assert_allclose(analytical,analytical.T,rtol=0,atol=1e-12)
    b=m._get_B_matrix()
    assert np.max(np.abs(analytical-b@m.get_hess_cart()@b.T))>1
    assert m.zmat==original


@pytest.mark.parametrize('shape',[(15,15),(1,15,15),(5,3,5,3)])
def test_cartesian_hessian_shape(zmat_data,shape):
    m=molecule(zmat_data)
    expected=m.get_hessian().copy()
    h=m.calculator.get_hessian(m.get_atoms())
    m.calculator.get_hessian=lambda atoms: h.reshape(shape)
    assert_allclose(m.get_hessian(),expected,rtol=1e-12,atol=1e-12)
    m.calculator.get_hessian=lambda atoms: np.zeros((14,14))
    with pytest.raises(ValueError,match='expected'):
        m.get_hess_cart()

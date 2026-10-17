import copy
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scientific_support import molecule

pytestmark=pytest.mark.L2


def test_three_hessian_methods_and_serialization(zmat_data,tmp_path):
    m=molecule(zmat_data)
    before=copy.deepcopy(m.zmat)
    reference=m.get_hessian()
    for scale in (.5,1.,2.):
        for actual in (m.get_geom_fd_hessian(1e-4*scale,1e-3*scale,1e-3*scale),
                       m.get_full_fd_hessian(1e-4*scale,1e-3*scale,1e-3*scale)):
            assert_allclose(actual,reference,atol=1e-4,rtol=1e-5)
            assert_allclose(actual,actual.T,atol=1e-5,rtol=1e-6)
    assert m.zmat==before
    m.energy=m.get_energy(); m.forces=m.get_forces()
    path=tmp_path/'result.json'; m.dump_json(path)
    loaded=type(m).load_json(path,load_hessian=True)
    assert_allclose(loaded.hessian,reference,atol=0,rtol=0)
    assert_allclose(loaded.forces,m.forces,atol=0,rtol=0)
    loaded.save_gaussian_fchk(tmp_path/'result.fchk')
    loaded.save_gaussian_log(tmp_path/'result.log')


@pytest.mark.parametrize('method',['get_full_fd_hessian','get_geom_fd_hessian'])
@pytest.mark.parametrize('steps',[(0,1e-3,1e-3),(-1,1,1),(1,float('nan'),1)])
def test_fd_invalid_steps(zmat_data,steps,method):
    with pytest.raises(ValueError,match='positive'):
        getattr(molecule(zmat_data),method)(*steps)


def test_bounded_ase_optimization_and_export(zmat_data,tmp_path,monkeypatch):
    from ase.calculators.calculator import Calculator,all_changes
    from pyzmat import Constraints,ZMatrix
    m=molecule(zmat_data)
    target=m.get_atoms().get_all_distances()
    class Springs(Calculator):
        implemented_properties=['energy','forces']
        def calculate(self,atoms=None,properties=('energy',),system_changes=all_changes):
            super().calculate(atoms,properties,system_changes)
            x=atoms.positions;f=np.zeros_like(x);e=0.
            for i in range(len(x)):
                for j in range(i):
                    delta=x[i]-x[j];distance=np.linalg.norm(delta);error=distance-target[i,j]
                    e+=.5*error**2;force=-error*delta/distance;f[i]+=force;f[j]-=force
            self.results={'energy':e,'forces':f}
    m.constraints=Constraints(bonds=[(1,1.42)],angles=[(2,112.)],dihedrals=[(3,63.)])
    fixed=copy.deepcopy(m.constraints)
    m.zmat[4][1]+=.08;m.calculator=Springs();m.name='springs'
    initial=m.get_energy()
    monkeypatch.chdir(tmp_path)
    m.optimise_ase(mode='bfgs',fmax=1e-5,max_steps=100)
    assert m.energy<initial
    for entries,column in [(fixed.bonds,1),(fixed.angles,2),(fixed.dihedrals,3)]:
        for index,value in entries: assert abs(m.zmat[index][column]-value)<1e-5
    m.dump_json(tmp_path/'optimized.json')
    loaded=ZMatrix.load_json(tmp_path/'optimized.json')
    assert_allclose(loaded.forces,m.forces,atol=0,rtol=0)
    m.save_gaussian_log(tmp_path/'optimized.log')
    assert 'Normal termination of MLIP.' in (tmp_path/'optimized.log').read_text()

"""Replay raw Gaussian derivatives, never cached transformed matrices."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from numpy.testing import assert_allclose
from pyzmat import ZMatrix

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('gaussian_validation',ROOT/'scripts/validate_gaussian_hessian.py')
validation=importlib.util.module_from_spec(spec); spec.loader.exec_module(validation)
FIXTURES=ROOT/'tests/fixtures/gaussian_hessians'


def replay(case,method='numba'):
    folder=FIXTURES/case
    cart=next((folder/'cart-2-z').glob('*.fchk'))
    internal=next((folder/'z').glob('*.fchk'))
    com=next((folder/'cart-2-z').glob('*.com'))
    names,constants=validation.parse_coordinate_order(com)
    assert_allclose(validation.read_fchk_array(cart,'Atomic numbers'),validation.read_fchk_array(internal,'Atomic numbers'),atol=0,rtol=0)
    assert_allclose(validation.read_fchk_array(cart,'Current cartesian coordinates'),validation.read_fchk_array(internal,'Current cartesian coordinates'),atol=0,rtol=0)
    assert_allclose(validation.read_fchk_array(cart,'Cartesian Gradient'),validation.read_fchk_array(internal,'Cartesian Gradient'),atol=1e-11,rtol=0)
    hx,fx,xyz=validation.extract_cartesian(cart)
    expected=validation.extract_zmat_hessian(internal,len(names))
    actual,rmsd=validation.transform_cartesian(hx,fx,xyz,com,method,(1e-4,1e-3,1e-3))
    actual=validation.convert_internal_to_bohr(actual,names)
    order=[names.index(n) for n in names if n not in constants]+[names.index(n) for n in constants]
    assert sorted(order)==list(range(len(names)))
    assert len(names)==len(fx)-6
    assert rmsd<1e-6
    return actual[np.ix_(order,order)],expected


@pytest.mark.L2
def test_roy_reference():
    actual,expected=replay('roy_new_lam_00001')
    assert_allclose(actual,expected,atol=2e-7,rtol=1e-5)


@pytest.mark.L3
@pytest.mark.parametrize('case',['roy_new_lam_00001','roy_new_lam_00050','vusdix_new_lam_00001'])
@pytest.mark.parametrize('method',['fast','numba','fd','slow'])
def test_archive_campaign(case,method):
    actual,expected=replay(case,method)
    if case == 'vusdix_new_lam_00001':
        folder=FIXTURES/case/'cart-2-z'
        bound=validation.propagated_rounding_bound(next(folder.glob('*.fchk')),next(folder.glob('*.com')))
        assert np.all(np.abs(actual-expected)<=2e-7+1e-5*np.abs(expected)+bound)
    else:
        assert_allclose(actual,expected,atol=2e-7,rtol=1e-5)


def test_archive_checksums():
    import hashlib
    for entry in json.loads((FIXTURES/'manifest.json').read_text()):
        data=(FIXTURES/entry['path']).read_bytes()
        assert hashlib.sha256(data).hexdigest()==entry['sha256']


@pytest.mark.L1
def test_frame_transform(zmat_data):
    from scientific_support import molecule
    m=molecule(zmat_data); xyz=m.get_atoms().positions
    rotation=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    transform=np.kron(np.eye(len(xyz)),rotation.T)
    fx=m.get_forces_cart(); hx=m.get_hess_cart()
    external_xyz=xyz@rotation+[3.,-2.,1.]
    recovered_h,recovered_f,rmsd=validation.rotate_cartesian_tensors(transform@hx@transform.T,transform@fx,external_xyz,xyz)
    assert rmsd<1e-12
    assert_allclose(recovered_f,fx,rtol=1e-9,atol=1e-9)
    assert_allclose(recovered_h,hx,rtol=1e-9,atol=1e-9)
    external_xyz[3,0]+=.1
    with pytest.raises(ValueError,match='RMSD'):
        validation.rotate_cartesian_tensors(hx,fx,external_xyz,xyz)


@pytest.mark.L2
def test_vusdix_file_precision_bound():
    actual,expected=replay('vusdix_new_lam_00001')
    folder=FIXTURES/'vusdix_new_lam_00001'/'cart-2-z'
    bound=validation.propagated_rounding_bound(next(folder.glob('*.fchk')),next(folder.glob('*.com')))
    assert not np.allclose(actual,expected,atol=2e-7,rtol=1e-5)
    allowed=2e-7+1e-5*np.abs(expected)+bound
    assert np.all(np.abs(actual-expected)<=allowed)
    corrupted=actual.copy();corrupted[0,5]+=.01
    assert np.any(np.abs(corrupted-expected)>allowed)


@pytest.mark.L2
@pytest.mark.parametrize('case,code',[('roy_new_lam_00001',0),('vusdix_new_lam_00001',1)])
def test_archive_cli_propagates_numerical_failure(case,code,tmp_path,monkeypatch):
    import sys
    folder=FIXTURES/case
    monkeypatch.setattr(sys,'argv',['validate','--cart-fchk',str(next((folder/'cart-2-z').glob('*.fchk'))),'--zmat-fchk',str(next((folder/'z').glob('*.fchk'))),'--com',str(next((folder/'cart-2-z').glob('*.com'))),'--output-dir',str(tmp_path/'output')])
    assert validation.main()==code
    report=json.loads((tmp_path/'output/comparison.json').read_text())
    assert report['allclose']==(code==0)

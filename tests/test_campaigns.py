"""Explicit allocated-node campaigns with local, provenance-pinned models."""
import copy
import os
from pathlib import Path
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scientific_support import molecule,energy_fd

pytestmark=pytest.mark.L3


def model_path():
    if not os.environ.get('PBS_JOBID'): pytest.skip('Requires PBS allocation')
    path=os.environ.get('LAM_MODEL')
    if not path: pytest.skip('Set LAM_MODEL to local MACE checkpoint')
    assert Path(path).is_file(),'Configured checkpoint is missing'
    return path


def test_real_model_hessian_methods(zmat_data):
    from mace.calculators import MACECalculator
    m=molecule(zmat_data)
    m.calculator=MACECalculator(model_paths=model_path(),device='cpu',default_dtype='float64')
    h=m.get_hessian()
    for scale in (.5,1.,2.):
        assert_allclose(m.get_full_fd_hessian(1e-4*scale,1e-3*scale,1e-3*scale),h,atol=1e-3,rtol=1e-5)
        assert_allclose(m.get_geom_fd_hessian(1e-4*scale,1e-3*scale,1e-3*scale),h,atol=1e-3,rtol=1e-5)
    f,_=energy_fd(m,1e-4)
    assert_allclose(m.get_forces(),f,atol=1e-5,rtol=1e-5)


@pytest.mark.parametrize('system',['roy_new_lam_00001','vusdix_new_lam_00001'])
@pytest.mark.parametrize('batch',[1,2,8])
def test_archive_cuda_batches(system,batch):
    import torch
    from pyzmat import ZMatrix,ZmatUtils
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    folder=Path(__file__).parent/'fixtures/gaussian_hessians'/system/'cart-2-z'
    m=ZMatrix.load_from_gaussian_input(next(folder.glob('*.com')))
    q=[v if j==1 else np.deg2rad(v) for row in m.zmat for j,v in enumerate(row[1:],1) if v is not None]
    q=torch.tensor([q]*batch,dtype=torch.float64)
    q[:,-1]+=torch.arange(batch)*.01
    expected=ZmatUtils.get_coordinate_derivatives_torch(q,m.zmat_conn)
    actual=ZmatUtils.get_coordinate_derivatives_torch(q.flip(0).cuda(),m.zmat_conn)
    for cpu,gpu in zip(expected,actual):
        assert_allclose(gpu.flip(0).cpu(),cpu,atol=1e-8,rtol=1e-8)

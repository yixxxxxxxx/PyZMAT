"""Check the optional batched backend against independent existing routines."""

import copy

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyzmat import ZmatUtils

torch = pytest.importorskip("torch")
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def pack(zmat):
    return [value if kind == 1 else np.deg2rad(value)
            for row in zmat for kind, value in enumerate(row[1:], 1)
            if value is not None]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("third_ref", [0, 1])
def test_batch_matches_numba_and_b(zmat_data, device, dtype, third_ref):
    zmat, conn = copy.deepcopy(zmat_data)
    conn[2] = ("H", third_ref, 1 - third_ref, None)
    changed = copy.deepcopy(zmat)
    changed[1][1] += 0.1
    changed[3][2] += 15.0
    changed[4][3] -= 120.0
    batch = [zmat, changed]
    q = torch.tensor([pack(z) for z in batch], dtype=dtype, device=device)
    # A non-contiguous tensor with more than one batch axis.
    q = q.repeat(2, 1, 1).transpose(0, 1)
    with torch.inference_mode():
        xyz, b, k = ZmatUtils.get_coordinate_derivatives_torch(q, conn)
    assert xyz.shape == (2, 2, 5, 3)
    assert b.shape == (2, 2, 9, 15)
    assert k.shape == (2, 2, 15, 9, 9)
    assert k.dtype == dtype and k.device == q.device
    tol = 3e-5 if dtype == torch.float32 else 1e-10
    for i, z in enumerate(batch):
        expected = (ZmatUtils.zmat_2_atoms(z, conn).positions,
                    ZmatUtils.get_B_matrix(z, conn),
                    ZmatUtils.get_curvature_tensor_numba(z, conn))
        for actual, reference in zip((xyz, b, k), expected):
            for j in range(2):
                assert_allclose(actual[i, j].cpu(), reference, atol=tol, rtol=tol)
    assert_allclose(k.cpu(), k.transpose(-1, -2).cpu(), atol=tol, rtol=tol)


@pytest.mark.parametrize("n", [3, 5])
def test_single_matches_finite_difference(zmat_data, n):
    zmat, conn = zmat_data
    zmat, conn = zmat[:n], conn[:n]
    q = torch.tensor(pack(zmat), dtype=torch.float64)
    k = ZmatUtils.get_curvature_tensor_torch(q, conn, validate=False)
    fd = ZmatUtils.get_fd_curvature_tensor(zmat, conn, 1e-4, 1e-2, 1e-2)
    assert_allclose(k, fd, atol=1e-6, rtol=1e-6)


def test_invalid_inputs(zmat_data):
    zmat, conn = zmat_data
    q = torch.tensor(pack(zmat), dtype=torch.float64)
    with pytest.raises(ValueError, match="shape"):
        ZmatUtils.get_curvature_tensor_torch(q[:-1], conn)
    with pytest.raises(TypeError, match="dtype"):
        ZmatUtils.get_curvature_tensor_torch(q.long(), conn)
    bad_conn = copy.deepcopy(conn)
    bad_conn[3] = ("O", 0, 1, 3)
    with pytest.raises(ValueError, match="earlier"):
        ZmatUtils.get_curvature_tensor_torch(q, bad_conn)
    for index, value in [(0, 0.), (0, -1.), (2, 0.), (4, float("nan"))]:
        bad_q = q.repeat(2, 1)
        bad_q[1, index] = value
        with pytest.raises(ValueError, match="non-degenerate"):
            ZmatUtils.get_curvature_tensor_torch(bad_q, conn)


def test_empty_batch(zmat_data):
    _, conn = zmat_data
    k = ZmatUtils.get_curvature_tensor_torch(torch.empty(0, 9), conn)
    assert k.shape == (0, 15, 9, 9)


@pytest.mark.parametrize("seed", [12, 37])
def test_deeper_branched_topology(zmat_data, seed):
    zmat, conn = copy.deepcopy(zmat_data)
    rng = np.random.default_rng(seed)
    for i in range(5, 10):
        zmat.append(["C", rng.uniform(0.9, 1.6), rng.uniform(65., 135.),
                     rng.uniform(-180., 180.)])
        conn.append(("C", *rng.choice(i, size=3, replace=False)))
    q = torch.tensor(pack(zmat), dtype=torch.float64)
    xyz, b, k = ZmatUtils.get_coordinate_derivatives_torch(q, conn)
    assert_allclose(xyz, ZmatUtils.zmat_2_atoms(zmat, conn).positions, atol=1e-11)
    assert_allclose(b, ZmatUtils.get_B_matrix(zmat, conn), atol=1e-10)
    assert_allclose(k, ZmatUtils.get_curvature_tensor_numba(zmat, conn), atol=1e-9)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_cuda_matches_cpu(zmat_data):
    zmat, conn = zmat_data
    q = torch.tensor([pack(zmat)] * 4, dtype=torch.float64)
    with torch.inference_mode():
        cpu = ZmatUtils.get_curvature_tensor_torch(q, conn)
        gpu = ZmatUtils.get_curvature_tensor_torch(q.cuda(), conn)
    assert_allclose(gpu.cpu(), cpu, atol=1e-10, rtol=1e-10)

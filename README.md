# PyZMAT

Author: Yixuan Huang

ASE wrapper for workflows in internal coordinates/Z-Matrix.

Built as a personal tool as a part of my research in ML-CSP under the HAIEFF framework [citations]. 

## Main capabilities

* Building molecules in Cartesian coordinates (ASE Atoms object) from Z-matrices.
* Calculating internal coordinates and forming Z-matrices from Cartesian coordinates.
* Analytical first and second derivatives of Cartesian coordinates with respect to internal coordinates.
* Analytical transformation of force vectors and Hessian matrices from Cartesian to internal coordinates based on the above derivatives.
* Executing a selection of ASE minimisation routines using ML force fields.

## Installation

(placeholder)

PyZMAT should work independently if you only wish to use its coordinate transform capabilities. Please install this package in a working MACE or AIMNet2 conda environment if you wish to use ML-FFs. 

## Batched coordinate derivatives on a GPU

The optional PyTorch backend computes Cartesian positions, B and the curvature
tensor K entirely on the input tensor's device, including inside
`torch.inference_mode()`. Install with `pip install -e '.[torch]'` in an environment
with a suitable PyTorch/CUDA installation. The existing NumPy/Numba APIs and
`ZMatrix.get_hessian()` retain their CPU behavior.

Pack each geometry's internal coordinates as
`[r1, r2, angle2, r3, angle3, dihedral3, ...]`, omitting the unused entries.
**Angles must be in radians**, while the original list-based Z-matrix API uses
degrees. All geometries in a call share the usual `zmat_conn` list and atom order.
For example, given a list `zmats` of conformers and their shared `zmat_conn`:

```python
import math
import torch
from pyzmat import ZmatUtils

device = "cuda" if torch.cuda.is_available() else "cpu"
q = torch.tensor([
    [value if kind == 1 else math.radians(value)
     for row in zmat for kind, value in enumerate(row[1:], start=1)
     if value is not None]
    for zmat in zmats
], dtype=torch.float64, device=device)

with torch.inference_mode():
    xyz, B, K = ZmatUtils.get_coordinate_derivatives_torch(q, zmat_conn)
    # Or, when only K is needed:
    # K = ZmatUtils.get_curvature_tensor_torch(q, zmat_conn)
```

For `q` shaped `(batch, M)`, where `M = 3*N - 6`, the outputs are
`xyz: (batch, N, 3)`, `B: (batch, M, 3*N)`, and `K: (batch, 3*N, M, M)`.
Unbatched inputs and multiple leading batch dimensions are also supported.
`B[s, i] = dr_i/dq_s` and `K[i, s, t] = d²r_i/(dq_s dq_t)` match the
existing routines, including their fixed Cartesian frame. A model's Cartesian
forces and Hessians must be expressed in this same frame and atom order.
Their batched transformation is:

```python
# F_cart: (..., 3*N), H_cart: (..., 3*N, 3*N), on the same device/dtype
F_internal = torch.einsum("...si,...i->...s", B, F_cart)
H_internal = B @ H_cart @ B.transpose(-1, -2) - torch.einsum(
    "...ist,...i->...st", K, F_cart
)
```

Use float64 for close agreement with the CPU implementation; float32 is also
supported. At least three atoms are required. Different topologies or atom counts
need separate batches. Validation rejects invalid connectivity, nonfinite values,
nonpositive bond lengths and degenerate reference frames. It synchronizes the GPU
once per call; `validate=False` avoids this synchronization for trusted numeric
inputs, but singular geometries can then produce NaNs/Infs.

Atoms are constructed sequentially, with batch and derivative dimensions
vectorized. GPU speedups depend on molecule size, batch size and hardware;
small inputs may be faster with Numba. Dense K alone occupies
`batch * 3*N * (3*N-6)**2 * element_size` bytes, with additional intermediate
storage. Split large batches into chunks to fit memory.

## Acknowledgements

Benjamin Tan for calculating the analytical first derivatives [citation].

## Optional ORCA force parsing

`ZMatrix.load_from_orca_output(path, parse_forces=False)` loads the same
geometry, connectivity, constraints and energy while skipping force parsing
and the CPU Wilson B transformation. Both `forces` and `forces_cart` are None.
The underlying `ParseUtils.parse_orca_output` accepts the same keyword and
returns None in its force slot. The default remains `parse_forces=True`.

## Testing

See [the four-tier test guide](tests/README.md) for kernel, structure, workflow
and allocated campaign checks, including the curated Gaussian Hessian fixtures.

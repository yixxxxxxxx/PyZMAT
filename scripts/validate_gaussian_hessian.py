#!/usr/bin/env python3
"""Extract, transform, reorder, and compare Gaussian Cartesian/Z-matrix Hessians.

The Cartesian FCHK supplies a 3N gradient and packed 3N x 3N Hessian.  The
Z-matrix FCHK supplies a packed (3N-6) x (3N-6) internal Hessian.  PyZMAT's
coordinate order follows the atom-definition block, whereas Gaussian writes
coordinates declared under ``constants:`` after all variables.  Consequently
both axes of the transformed Hessian are permuted before comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path

# PyZMAT enables Numba's on-disk cache.  Installed/editable source trees can be
# read-only on clusters, so choose a writable cache before importing PyZMAT.
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "pyzmat-numba-cache"))

import numpy as np
from pyzmat import ZmatUtils, ZMatrix


ANGSTROM_TO_BOHR = 1.889726125


def read_fchk_array(path: Path, field_name: str) -> np.ndarray:
    """Read a real array from a Gaussian formatted checkpoint file."""
    lines = path.read_text().splitlines()
    for line_number, line in enumerate(lines):
        if not line.startswith(field_name):
            continue
        match = re.search(r"N=\s*(\d+)", line)
        if match is None:
            raise ValueError(f"No array size on {field_name!r} line in {path}")
        size = int(match.group(1))
        values: list[float] = []
        for data_line in lines[line_number + 1 :]:
            try:
                values.extend(float(x.replace("D", "E")) for x in data_line.split())
            except ValueError as exc:
                raise ValueError(
                    f"{field_name!r} in {path} ended after {len(values)}/{size} values"
                ) from exc
            if len(values) >= size:
                return np.asarray(values[:size], dtype=float)
        raise ValueError(f"{field_name!r} in {path} has only {len(values)}/{size} values")
    raise ValueError(f"Field {field_name!r} was not found in {path}")


def unpack_symmetric(packed: np.ndarray, size: int, field_name: str) -> np.ndarray:
    """Expand Gaussian's row-wise packed lower triangle."""
    expected = size * (size + 1) // 2
    if packed.size != expected:
        raise ValueError(
            f"{field_name} has {packed.size} values, expected {expected} "
            f"for a {size} x {size} matrix"
        )
    matrix = np.empty((size, size), dtype=float)
    lower = np.tril_indices(size)
    matrix[lower] = packed
    matrix[(lower[1], lower[0])] = packed
    return matrix


def extract_cartesian(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gradient = read_fchk_array(path, "Cartesian Gradient")
    if gradient.size % 3:
        raise ValueError(f"Cartesian Gradient length {gradient.size} is not 3N")
    packed = read_fchk_array(path, "Cartesian Force Constants")
    hessian = unpack_symmetric(packed, gradient.size, "Cartesian Force Constants")
    xyz = read_fchk_array(path, "Current cartesian coordinates").reshape(-1, 3)
    if xyz.size != gradient.size:
        raise ValueError("Cartesian coordinates and gradient have inconsistent sizes")
    return hessian, -gradient, xyz / ANGSTROM_TO_BOHR


def extract_zmat_hessian(path: Path, expected_size: int) -> np.ndarray:
    packed = read_fchk_array(path, "Internal Force Constants")
    return unpack_symmetric(packed, expected_size, "Internal Force Constants")


def parse_coordinate_order(com_path: Path) -> tuple[list[str], list[str]]:
    """Return natural Z-matrix coordinate names and names marked constant."""
    lines = com_path.read_text().splitlines()
    section = None
    constants: list[str] = []
    for line in lines:
        stripped = line.strip()
        heading = stripped.lower()
        if heading in {"variables:", "constants:"}:
            section = heading[:-1]
            continue
        if section == "constants" and stripped:
            constants.append(stripped.split()[0])

    # Coordinate identifiers occur in columns 3, 5, and 7 of atom definitions.
    atom_start = next(
        (i + 1 for i, line in enumerate(lines) if re.fullmatch(r"\s*-?\d+\s+\d+\s*", line)),
        None,
    )
    if atom_start is None:
        raise ValueError(f"Could not locate charge/multiplicity line in {com_path}")
    coordinates: list[str] = []
    for line in lines[atom_start:]:
        stripped = line.strip()
        if not stripped or stripped.lower() in {"variables:", "constants:"}:
            break
        tokens = stripped.split()
        coordinates.extend(tokens[i] for i in (2, 4, 6) if i < len(tokens))

    missing = [name for name in constants if name not in coordinates]
    if missing:
        raise ValueError(f"Constant coordinates absent from Z-matrix: {missing}")
    return coordinates, constants


def rotate_cartesian_tensors(
    hessian: np.ndarray,
    forces: np.ndarray,
    gaussian_xyz: np.ndarray,
    pyzmat_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Rotate Gaussian tensors into the Cartesian frame constructed by PyZMAT."""
    if gaussian_xyz.shape != pyzmat_xyz.shape:
        raise ValueError(f"Geometry shapes differ: {gaussian_xyz.shape} vs {pyzmat_xyz.shape}")
    gaussian_centered = gaussian_xyz - gaussian_xyz.mean(axis=0)
    pyzmat_centered = pyzmat_xyz - pyzmat_xyz.mean(axis=0)
    u, _, vt = np.linalg.svd(pyzmat_centered.T @ gaussian_centered)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    aligned = pyzmat_centered @ rotation
    rmsd = float(np.sqrt(np.mean(np.sum((aligned - gaussian_centered) ** 2, axis=1))))
    if rmsd > 1.0e-6:
        raise ValueError(f"Gaussian/PyZMAT geometry alignment RMSD is {rmsd:.6e} Angstrom")

    natoms = len(gaussian_xyz)
    rotated_forces = forces.reshape(natoms, 3) @ rotation.T
    blocks = hessian.reshape(natoms, 3, natoms, 3)
    rotated_hessian = np.einsum("pa,iajb,qb->ipjq", rotation, blocks, rotation)
    return rotated_hessian.reshape(3 * natoms, 3 * natoms), rotated_forces.ravel(), rmsd


def transform_cartesian(
    cart_hessian: np.ndarray,
    cart_forces: np.ndarray,
    gaussian_xyz: np.ndarray,
    com_path: Path,
    k_method: str,
    fd_steps: tuple[float, float, float],
) -> tuple[np.ndarray, float]:
    """Apply the same Cartesian-to-Z-matrix transformation as cart-2-z/transform.py."""
    molecule = ZMatrix.load_from_gaussian_input(str(com_path))
    zmat, connectivity = molecule.zmat, molecule.zmat_conn

    # PyZMAT differentiates coordinates expressed in Angstrom/radian.
    hessian_a = cart_hessian * ANGSTROM_TO_BOHR**2
    forces_a = cart_forces * ANGSTROM_TO_BOHR
    hessian_a, forces_a, rmsd = rotate_cartesian_tensors(
        hessian_a, forces_a, gaussian_xyz, molecule.get_atoms().get_positions()
    )
    b_matrix = ZmatUtils.get_B_matrix(zmat, connectivity)
    if k_method == "fast":
        curvature = ZmatUtils.get_curvature_tensor_fast(zmat, connectivity)
    elif k_method == "numba":
        curvature = ZmatUtils.get_curvature_tensor_numba(zmat, connectivity)
    elif k_method == "slow":
        curvature = ZmatUtils.get_curvature_tensor(zmat, connectivity)
    elif k_method == "fd":
        curvature = ZmatUtils.get_fd_curvature_tensor(zmat, connectivity, *fd_steps)
    else:  # Guard direct callers as well as the argparse CLI.
        raise ValueError(f"Unknown K-tensor method: {k_method!r}")
    transformed = b_matrix @ hessian_a @ b_matrix.T
    transformed -= np.einsum("isp,i->sp", curvature, forces_a)
    return transformed, rmsd


def convert_internal_to_bohr(hessian: np.ndarray, names: list[str]) -> np.ndarray:
    """Convert bond axes from Angstrom to bohr; angular axes remain radians."""
    scale = np.asarray(
        [1.0 / ANGSTROM_TO_BOHR if name.lower().startswith(("bnd", "r")) else 1.0 for name in names]
    )
    return scale[:, None] * hessian * scale[None, :]


def comparison(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    delta = candidate - reference
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "relative_frobenius": float(np.linalg.norm(delta) / np.linalg.norm(reference)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cart-fchk", type=Path, default=Path("cart-2-z/uniform_lam_00001.fchk"))
    parser.add_argument("--zmat-fchk", type=Path, default=Path("z/uniform_lam_00001_zmat.fchk"))
    parser.add_argument("--com", type=Path, default=Path("cart-2-z/new_lam_00001.com"))
    parser.add_argument(
        "--k-method",
        choices=("fast", "numba", "slow", "fd"),
        default="numba",
        help="K-tensor implementation to use (default: %(default)s)",
    )
    parser.add_argument(
        "--fd-steps",
        type=float,
        nargs=3,
        metavar=("BOND", "ANGLE", "DIHEDRAL"),
        default=(1.0e-4, 1.0e-3, 1.0e-3),
        help="finite-difference steps used by --k-method fd (Angstrom, degrees, degrees)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="output directory (default: consolidated_output_<k-method>)",
    )
    parser.add_argument("--rtol", type=float, default=1.0e-5)
    parser.add_argument("--atol", type=float, default=2.0e-7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or Path(f"consolidated_output_{args.k_method}")
    output = output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    coordinate_names, constants = parse_coordinate_order(args.com)
    cart_hessian, cart_forces, gaussian_xyz = extract_cartesian(args.cart_fchk)
    expected_internal = cart_forces.size - 6
    if len(coordinate_names) != expected_internal:
        raise ValueError(
            f"COM defines {len(coordinate_names)} coordinates; expected 3N-6 = {expected_internal}"
        )
    zmat_hessian = extract_zmat_hessian(args.zmat_fchk, expected_internal)
    transformed_a, alignment_rmsd = transform_cartesian(
        cart_hessian,
        cart_forces,
        gaussian_xyz,
        args.com,
        args.k_method,
        tuple(args.fd_steps),
    )
    transformed_bohr = convert_internal_to_bohr(transformed_a, coordinate_names)

    constant_set = set(constants)
    ordered_names = [name for name in coordinate_names if name not in constant_set] + constants
    permutation = [coordinate_names.index(name) for name in ordered_names]
    transformed_reordered = transformed_bohr[np.ix_(permutation, permutation)]

    np.save(output / "cartesian_forces.npy", cart_forces)
    np.save(output / "cartesian_hessian.npy", cart_hessian)
    np.save(output / "zmat_hessian.npy", zmat_hessian)
    np.save(output / "transformed_hessian_natural_order.npy", transformed_bohr)
    np.save(output / "transformed_hessian.npy", transformed_reordered)

    raw_stats = comparison(transformed_bohr, zmat_hessian)
    final_stats = comparison(transformed_reordered, zmat_hessian)
    passed = np.allclose(transformed_reordered, zmat_hessian, rtol=args.rtol, atol=args.atol)
    report = {
        "cartesian_forces_shape": list(cart_forces.shape),
        "cartesian_hessian_shape": list(cart_hessian.shape),
        "zmat_hessian_shape": list(zmat_hessian.shape),
        "alignment_rmsd_angstrom": alignment_rmsd,
        "k_tensor_method": args.k_method,
        "fd_steps": args.fd_steps if args.k_method == "fd" else None,
        "natural_coordinate_order": coordinate_names,
        "constant_coordinates": constants,
        "gaussian_coordinate_order": ordered_names,
        "permutation": permutation,
        "before_constant_reordering": raw_stats,
        "after_constant_reordering": final_stats,
        "rtol": args.rtol,
        "atol": args.atol,
        "allclose": bool(passed),
    }
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"Cartesian forces:  {cart_forces.shape}")
    print(f"Cartesian Hessian: {cart_hessian.shape}")
    print(f"Z-matrix Hessian:  {zmat_hessian.shape}")
    print(f"Frame alignment RMSD: {alignment_rmsd:.3e} Angstrom")
    print(f"Constants moved to end: {constants or 'none'}")
    print(f"Coordinate permutation: {permutation}")
    print("Before constant reordering: " + ", ".join(f"{k}={v:.6e}" for k, v in raw_stats.items()))
    print("After constant reordering:  " + ", ".join(f"{k}={v:.6e}" for k, v in final_stats.items()))
    print(f"np.allclose(rtol={args.rtol:g}, atol={args.atol:g}): {passed}")
    print(f"Saved arrays in {output}")
    return 0 if passed else 1




def fchk_rounding_bounds(path, field_name):
    """Half a last-written decimal unit for each FCHK real token."""
    lines=Path(path).read_text().splitlines()
    for index,line in enumerate(lines):
        if line.startswith(field_name):
            size=int(line.split('N=')[1]); tokens=[]
            for line in lines[index+1:]:
                tokens.extend(line.split())
                if len(tokens)>=size: break
            if len(tokens)!=size: raise ValueError('Invalid FCHK array length')
            bounds=[]
            for token in tokens:
                mantissa,exponent=re.split('[DEde]',token)
                decimals=len(mantissa.split('.')[1]) if '.' in mantissa else 0
                bounds.append(.5*10.**(int(exponent)-decimals))
            return np.array(bounds)
    raise ValueError('Missing FCHK field: '+field_name)


def propagated_rounding_bound(cart_path, com_path):
    """Absolute interval bound for rounding Cartesian H/F before transformation.

    |delta(B H B.T - K F)| <= |B R| deltaH |B R|.T + |K| |R| deltaF.
    Geometry/frame truncation remains covered by the original absolute floor.
    This bounds file quantization, not an adjustable numerical tolerance.
    """
    h,f,x=extract_cartesian(cart_path)
    mol=ZMatrix.load_from_gaussian_input(com_path)
    y=mol.get_atoms().positions
    u,_,vt=np.linalg.svd((y-y.mean(0)).T@(x-x.mean(0)))
    rotation=u@vt
    if np.linalg.det(rotation)<0:
        u[:,-1]*=-1; rotation=u@vt
    block=np.kron(np.eye(len(x)),rotation)
    dh=unpack_symmetric(fchk_rounding_bounds(cart_path,'Cartesian Force Constants'),len(f),'rounding')
    df=fchk_rounding_bounds(cart_path,'Cartesian Gradient')
    b=np.abs(mol._get_B_matrix()@block)*ANGSTROM_TO_BOHR
    k=np.abs(mol._get_K_tensor())
    bound=b@dh@b.T+np.einsum('ist,i->st',k,np.abs(block)@df*ANGSTROM_TO_BOHR)
    names,constants=parse_coordinate_order(com_path)
    order=[names.index(n) for n in names if n not in constants]+[names.index(n) for n in constants]
    return convert_internal_to_bohr(bound,names)[np.ix_(order,order)]

if __name__ == "__main__":
    raise SystemExit(main())

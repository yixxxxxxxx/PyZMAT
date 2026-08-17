#!/usr/bin/env python3
"""Compare PyZMAT internal-coordinate Hessian implementations.

The Gaussian input must contain a Z-matrix followed by Variables:/Constants:
blocks in the format accepted by ``ZMatrix.load_from_gaussian_input``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

from pyzmat import ZMatrix


METHODS = ("analytical", "geometry-fd", "full-fd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate internal-coordinate Hessians using PyZMAT's analytical, "
            "geometry finite-difference, and full finite-difference methods."
        )
    )
    parser.add_argument("input", type=Path, help="Gaussian Z-matrix input (.com)")
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=None,
        help="Output directory (default: <input-stem>_hessian_comparison)",
    )
    parser.add_argument(
        "--model", choices=("mace-off", "mace-omol", "aimnet2"),
        default="mace-off", help="Calculator backend (default: mace-off)",
    )
    parser.add_argument(
        "--model-size", choices=("small", "medium", "large"), default="large",
        help="MACE-OFF model size (default: large)",
    )
    parser.add_argument("--gpu", action="store_true", help="Use the GPU backend")
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=list(METHODS),
        help="Methods to evaluate (default: all)",
    )
    parser.add_argument(
        "--db", type=float, default=1.0e-4,
        help="Bond finite-difference step in angstrom (default: 1e-4)",
    )
    parser.add_argument(
        "--da", type=float, default=1.0e-3,
        help="Angle finite-difference step in degrees (default: 1e-3)",
    )
    parser.add_argument(
        "--dt", type=float, default=1.0e-3,
        help="Dihedral finite-difference step in degrees (default: 1e-3)",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.input.is_file():
        raise FileNotFoundError(f"Gaussian input does not exist: {args.input}")
    if args.input.suffix.lower() not in {".com", ".gjf"}:
        raise ValueError("Input must have a .com or .gjf extension")
    for name in ("db", "da", "dt"):
        if getattr(args, name) <= 0.0:
            raise ValueError(f"--{name} must be positive")
    if len(set(args.methods)) < 2:
        raise ValueError("Select at least two methods to calculate comparison metrics")


def matrix_metrics(matrix: np.ndarray) -> dict[str, float | list[int]]:
    scale = max(float(np.linalg.norm(matrix, ord="fro")), np.finfo(float).eps)
    return {
        "shape": list(matrix.shape),
        "frobenius_norm": float(np.linalg.norm(matrix, ord="fro")),
        "max_abs_element": float(np.max(np.abs(matrix))),
        "symmetry_max_abs": float(np.max(np.abs(matrix - matrix.T))),
        "symmetry_relative_frobenius": float(
            np.linalg.norm(matrix - matrix.T, ord="fro") / scale
        ),
    }


def comparison_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    if a.shape != b.shape:
        raise ValueError(f"Cannot compare Hessians with shapes {a.shape} and {b.shape}")
    diff = a - b
    abs_diff = np.abs(diff)
    reference_scale = max(float(np.linalg.norm(b, ord="fro")), np.finfo(float).eps)
    denominator = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1.0e-12)
    return {
        "mae": float(np.mean(abs_diff)),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "max_abs_error": float(np.max(abs_diff)),
        "frobenius_error": float(np.linalg.norm(diff, ord="fro")),
        "relative_frobenius_error": float(
            np.linalg.norm(diff, ord="fro") / reference_scale
        ),
        "max_elementwise_relative_error": float(np.max(abs_diff / denominator)),
    }


def evaluate(
    molecule: ZMatrix, method: str, db: float, da: float, dt: float
) -> np.ndarray:
    if method == "analytical":
        result = molecule.get_hessian()
    elif method == "geometry-fd":
        result = molecule.get_geom_fd_hessian(db, da, dt)
    elif method == "full-fd":
        result = molecule.get_full_fd_hessian(db, da, dt)
    else:
        raise ValueError(f"Unknown Hessian method: {method}")

    matrix = np.asarray(result, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{method} returned a non-square shape: {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise FloatingPointError(f"{method} returned NaN or infinite values")
    return matrix


def print_summary(report: dict) -> None:
    print("\nMethod summary")
    print(f"{'method':<14} {'seconds':>11} {'||H||F':>14} {'symmetry max':>14}")
    for name, values in report["methods"].items():
        print(
            f"{name:<14} {values['seconds']:>11.3f} "
            f"{values['frobenius_norm']:>14.6e} "
            f"{values['symmetry_max_abs']:>14.6e}"
        )

    print("\nPairwise comparisons (second method is the relative-error reference)")
    print(f"{'pair':<31} {'MAE':>12} {'RMSE':>12} {'max abs':>12} {'rel ||.||F':>12}")
    for pair, values in report["comparisons"].items():
        print(
            f"{pair:<31} {values['mae']:>12.4e} {values['rmse']:>12.4e} "
            f"{values['max_abs_error']:>12.4e} "
            f"{values['relative_frobenius_error']:>12.4e}"
        )


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        output_dir = args.output_dir or Path(
            f"{args.input.stem}_hessian_comparison"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        molecule = ZMatrix.load_from_gaussian_input(str(args.input))
        molecule.attach_calculator(
            args.model, model_size=args.model_size, gpu=args.gpu
        )

        report: dict = {
            "input": str(args.input.resolve()),
            "model": args.model,
            "model_size": args.model_size,
            "gpu": args.gpu,
            "finite_difference_steps": {
                "bond_angstrom": args.db,
                "angle_degrees": args.da,
                "dihedral_degrees": args.dt,
            },
            "methods": {},
            "comparisons": {},
        }
        matrices = {}
        for method in dict.fromkeys(args.methods):
            print(f"Evaluating {method} Hessian...", flush=True)
            started = time.perf_counter()
            matrix = evaluate(molecule, method, args.db, args.da, args.dt)
            elapsed = time.perf_counter() - started
            matrices[method] = matrix
            np.save(output_dir / f"hessian_{method}.npy", matrix)
            np.savetxt(
                output_dir / f"hessian_{method}.txt",
                matrix,
                fmt="%.12e",
            )
            report["methods"][method] = {
                "seconds": elapsed,
                **matrix_metrics(matrix),
            }

        for first, second in combinations(matrices, 2):
            key = f"{first} vs {second}"
            report["comparisons"][key] = comparison_metrics(
                matrices[first], matrices[second]
            )

        with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")

        print_summary(report)
        print(f"\nMatrices and metrics written to {output_dir.resolve()}")
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

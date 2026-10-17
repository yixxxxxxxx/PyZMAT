# Four tier scientific tests

Run from the repository root with Python 3.10 or later and the package's NumPy,
ASE and pytest dependencies. Torch, MACE, Numba and GPU support are optional
capabilities checked separately. On HX1 use `/gpfs/home/yh6324/prod/bin/python`;
the old `~/load_conda.sh` initialization is not installed.

```bash
PYTHONPATH=src python tools/run_tiers.py --validate
PYTHONPATH=src python tools/run_tiers.py --level L0 --list
PYTHONPATH=src python tools/run_tiers.py --level L0 --out /fresh/evidence/L0 --require-all
PYTHONPATH=src python tools/run_tiers.py --level L1 --out /fresh/evidence/L1 --require-all
PYTHONPATH=src python tools/run_tiers.py --level L2 --out /fresh/evidence/L2 --require-all
```

`tests/meta.json` indexes case directories. Each contains a contract, fixture
checksums and reference policy. The standalone version-2 schema adds `pyzmat`
and `ml-lams` package identities without modifying CSPImperial's registry.
References remain `prepared`; this runner never overwrites or automatically
promotes scientific baselines. Mathematical tests use independent analytic or
finite-difference oracles, while archive tests use original Gaussian fields.

L0 covers geometry, constraints, input/output, CPU derivatives, CLI failure
status and contract validation. L1 exercises a full polynomial energy-to-force
and Hessian calculation, Cartesian shape normalization and external-frame
rotation. L2 compares three Hessian methods, executes a bounded ASE optimization
with constraints and serialization, replays ROY and checks VUSDIX file precision.
L3 replays all archive methods, checks a real local MACE model, and compares
Torch CPU/CUDA batches for both larger molecules.

The runner sets one numerical thread, enforces wall and per-process CPU limits,
checks peak RSS against the case budget and preserves logs/JUnit/timing and
scratch artifacts. Work happens in a short node-local temporary directory so
Unix sockets and callback staging work; evidence is copied to the requested
fresh directory. GNU `time -v` is used when installed; otherwise Linux `wait4`
records wall/CPU/peak RSS explicitly. Memory is a measured failure gate, not an
address-space limit (Torch maps large shared libraries). A whole process group
is killed on wall timeout. CPU limits apply per process; they are not an HPC
allocation-wide accounting substitute.

`results.json` records source-file hashes (including dirty/new files), revision,
interpreter, packages, model hashes, command and results. PASS requires all
selected tests to pass. Optional missing dependencies yield SKIP with reasons;
`--require-all` returns 2 for incomplete coverage and 1 for failures. Case
checksums are verified before execution. Registering a test in the wrong tier
cannot turn an empty selection into PASS. Existing directories are refused.
L0/L1 budgets include startup; slow cases must move tiers explicitly. These
runs make no performance claim; use same-node single-pass paired campaigns and
scheduler accounting for such claims.

## Allocated campaigns

```bash
export LAM_MODEL=/absolute/path/to/local/float64/checkpoint
PYTHONPATH=src python tools/run_tiers.py --level L3 --out /fresh/evidence/L3 --require-all
```

L3 requires `PBS_JOBID`. CUDA cases require a working GPU/Torch stack; model
cases require the local file above and MACE. No model is downloaded by these
campaign tests. GPU batching checks use 1, 2 and 8 distinct conformers and verify
reversed order. The existing CPU/CUDA derivative cases are selected at L3 too.
Plain `python -m pytest tests` skips L3 even on a GPU. For debugging a selected
tier directly, set `TIER_LEVEL=L1` (resource limits and checksum verification
still require the runner).

[Raw Gaussian fixtures](fixtures/gaussian_hessians/README.md) document the
stale archive outputs excluded from use and the independently derived VUSDIX
file-quantization bound. The original archive remains unchanged. The strict
VUSDIX historical comparison still fails and is explicitly tested as such;
the separate precision-aware contract passes without changing old thresholds.

`python scripts/compare_hessians.py input.com --checkpoint /path/model
--output-dir /fresh/comparison` now exits nonzero on numerical mismatch and
records input/model hashes and tolerances. Outputs must be fresh. The
`validate_gaussian_hessian.py` CLI also returns nonzero on its strict comparison
failure; its default retains the original tolerances. The campaign's explicit
quantization test is separately implemented in `tests/test_archive.py`.

## Scope corrections to the initial plan

The original plans are at `~/organisation/tiered-tests/`. The PyZMAT fixture
set is curated into this repository, so basic replay needs no archive or
Gaussian installation. OptKing is an optional alternate optimizer and is not
certified by the ASE gate; its previously missing lazy import is fixed. No
OptKing installation or Gaussian regeneration is claimed. Performance baselines
and cross-toolchain baseline promotion remain separate acceptance work.

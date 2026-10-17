# Raw Gaussian Hessian references

These 18 files (16,461,571 bytes) are byte-preserving selections from
`tests/hess_test.tar`, not regenerated electronic calculations. `manifest.json`
records the original member path and SHA-256 of every selected file. Reproduce
into a new directory with `python tools/curate_hessians.py tests/hess_test.tar
--out /fresh/directory`. No cached transformed arrays, notebooks, binary
checkpoints or stale comparison reports are used.

ROY 00001 and 00050 agree with fresh PyZMAT transformations at the historical
`atol=2e-7, rtol=1e-5` bound in Hartree/bohr/radian blocks. VUSDIX has 46 atoms
and 132 internal coordinates; the historical 75-coordinate outputs were stale
ROY data and are excluded. Raw VUSDIX still fails the original pointwise bound.

FCHK writes Cartesian derivatives with limited decimal precision. The new
VUSDIX precision test propagates half a last-written decimal unit per input
through `|B R| deltaH |B R|^T + |K| |R| deltaF`, including angstrom/bohr scaling
and both-axis coordinate permutation. It requires each error to be within this
computed bound plus the unchanged historical absolute/relative floor. The
original strict failure is asserted separately; a 0.01 corruption must fail the
precision-aware gate. This is a distinct T3 quantization contract, not a new
accepted golden matrix or an unexplained increase in tolerance.

The Z-matrix calculations deliberately use `Opt=(Z-Matrix,CalcAll,MaxCycles=1)`.
Their logs can terminate at the optimization cycle limit. These fixtures test
fixed-geometry derivatives, not optimization convergence. Compare the input,
Cartesian frame, atom count, units and coordinate order before using arrays.
No optimization success claim is made from these files.

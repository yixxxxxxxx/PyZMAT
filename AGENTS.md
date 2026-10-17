# Workspace environment

For Python development and testing, prefer the project Conda environment when
its initialization script is installed:

```bash
source ~/load_conda.sh
conda activate MACE_lora
```

On HX1 the script and Conda are absent. Use the verified scientific interpreter
`/gpfs/home/yh6324/prod/bin/python` with `PYTHONPATH=src` instead. Do not resolve
its executable symlink: doing so loses the virtual environment's packages.
Set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
NUMBA_NUM_THREADS=1` for bounded CPU checks. Shell state does not persist between
commands. See `tests/README.md` for tier selection and allocated campaigns.

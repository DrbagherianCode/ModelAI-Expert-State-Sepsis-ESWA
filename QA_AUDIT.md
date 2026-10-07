# v1.1.0 portability and upstream-construction QA audit

Date: 6 October 2026

## Raw-to-core smoke test

`01_build_core_arrays.py` was executed on the first 100 patients of Set A and first 100 patients of Set B (7,593 hourly rows). The generated arrays were compared row-for-row against the canonical full-study arrays at the corresponding patient rows:

- `ALL_Xb.npy`: exact match, including NaN locations and values.
- `ALL_Xs.npy`: exact match.
- `ALL_y.npy`: exact match.

## Trajectory smoke test

`02_build_trajectory_v2.py` was executed on the same 200-patient QA output in four patient-boundary-safe blocks. Comparison with the corresponding canonical full-study rows showed:

- `ALL_Xtraj_v2.npy`: exact match, including NaN locations/values.
- `ALL_Xforward_v2.npy`: exact match.

## Split reconstruction

The patient-disjoint split construction in `01_build_core_arrays.py` reproduces the same train/validation/test patient memberships used in the study (70/15/15, seed 42, stratified by site and patient-positive status). The archive uses uncompressed `np.savez` to preserve the canonical split representation.

## Path portability audit

All executable Python scripts and `run_all.py` were searched for hard-coded `/mnt/data` paths. None are present. Paths are provided through `--set-a`, `--set-b`, `--input-dir`, `--array-dir`, `--prediction-dir`, `--work-dir`, `--results-dir`, or resolved relative to the repository root.

## Syntax audit

All Python scripts in `scripts/` and `run_all.py` pass `python -m py_compile` under Python 3.13.5.

## Canonical full-study verification

`CANONICAL_ARRAY_SHA256.txt` records the exact SHA-256 values of the generated arrays used by the manuscript. After a full raw-data preprocessing run, `scripts/09_verify_reproduction.py` can verify byte-level identity before model fitting.

## v1.1.1 editorial reproducibility correction
- Added mandatory automatic canonical-array verification to `run_all.py` immediately after controls and before training.
- Added final reference-result comparison at workflow completion.
- Static path audit remains clean: no executable Python script contains `/mnt/data`.
- This change does not alter scientific results or analysis definitions.

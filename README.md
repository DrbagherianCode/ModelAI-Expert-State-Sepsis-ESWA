# ModelAI Expert-State Sepsis Reproducibility

[![Version](https://img.shields.io/badge/version-1.1.1-blue)](#versioning)
[![Python](https://img.shields.io/badge/python-3.13%20tested-blue)](#environment)
[![Dataset](https://img.shields.io/badge/data-PhysioNet%2FCinC%202019-red)](https://physionet.org/content/challenge-2019/1.0.0/)
[![Reproducibility](https://img.shields.io/badge/reproducibility-end--to--end-brightgreen)](#one-command-reproduction)
[![DOI](https://img.shields.io/badge/DOI-pending-lightgrey)](#archival-doi)

End-to-end reproducibility repository for:

**Auditable Expert State-Machine Representations for Early Sepsis Prediction:  
An Exploratory Evaluation Against Generic Temporal Controls**

Author: **Abbas Bagherian Kasgari**  
Affiliation: **ModelAI Ltd, London, United Kingdom**

## Overview

This repository reconstructs the reported analysis directly from the public
PhysioNet/Computing in Cardiology Challenge 2019 raw PSV files. It builds the
raw/missingness baseline, expert alerts and state machines, longitudinal
trajectory representations, forward-scenario features, GenericTemporal27,
Noise27, fixed patient split, model fits, calibration, utility, bootstrap,
cross-site analyses, repeated-split sensitivity analyses, and integrity checks.

The final trajectory analysis is explicitly **exploratory** because earlier
versions of the original held-out split had already been inspected. The primary
endpoint remains AUPRC. The associated manuscript does **not** claim unique
predictive superiority of expert semantics over the strong generic temporal
comparator.

## Public dataset

**Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in
Cardiology Challenge 2019, v1.0.0**

- PhysioNet: https://physionet.org/content/challenge-2019/1.0.0/
- Dataset DOI: https://doi.org/10.13026/v64v-d857

Raw PhysioNet data are intentionally **not redistributed** here.

The pipeline accepts either extracted directories of `.psv` files or the
official Set A / Set B ZIP archives.

## Repository structure

```text
.
├── run_all.py
├── scripts/
│   ├── 00_download_physionet.sh
│   ├── 01_build_core_arrays.py
│   ├── 02_build_trajectory_v2.py
│   ├── 03_build_controls.py
│   ├── 04_train_all_arms.py
│   ├── 05_calibration_utility.py
│   ├── 06_bootstrap_primary.py
│   ├── 07_crosssite.py
│   ├── 08_repeated_splits.py
│   └── 09_verify_reproduction.py
├── specs/
├── results_reference/
├── CANONICAL_ARRAY_SHA256.txt
├── SHA256SUMS.txt
├── FILE_MANIFEST.json
├── PROVENANCE.md
├── QA_AUDIT.md
├── SOURCE_DATA.md
├── CITATION.cff
├── requirements.txt
├── environment.yml
├── Dockerfile
├── VERSION
└── LICENSE
```

## Environment

Tested reference environment:

- Python 3.13.5
- NumPy 2.3.5
- pandas 2.2.3
- scikit-learn 1.8.0
- LightGBM 4.6.0

Create an isolated environment:

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## One-command reproduction

From the repository root:

```bash
python run_all.py \
  --set-a /path/to/training_setA \
  --set-b /path/to/training_setB \
  --work-dir work \
  --results-dir results_reproduced
```

ZIP inputs are also accepted:

```bash
python run_all.py \
  --set-a /path/to/training_setA.zip \
  --set-b /path/to/training_setB.zip \
  --work-dir work \
  --results-dir results_reproduced
```

No executable Python script depends on `/mnt/data` or another
machine-specific absolute analysis path.

## Reproduction stages

1. `01_build_core_arrays.py`
   - `ALL_Xb.npy`
   - `ALL_Xs.npy`
   - `ALL_y.npy`
   - `ALL_pid.npy`
   - `patient_manifest.csv`
   - `split_seed42.npz`
2. `02_build_trajectory_v2.py`
   - `ALL_Xtraj_v2.npy`
   - `ALL_Xforward_v2.npy`
3. `03_build_controls.py`
   - `ALL_Xgeneric27.npy`
   - `ALL_Xnoise27.npy`
4. `09_verify_reproduction.py`
   - mandatory canonical SHA-256 gate before model fitting
5. `04_train_all_arms.py`
   - six analysis arms, seeds 11, 23, 42, 73, 101
6. `05_calibration_utility.py`
   - validation-only Platt calibration, ECE, Brier, Challenge utility,
     alert burden, lead time
7. `06_bootstrap_primary.py`
   - 2,000 patient-cluster bootstrap replicates
8. `07_crosssite.py`
   - five-seed A→B and B→A internal transportability analysis
9. `08_repeated_splits.py`
   - three prespecified repeated patient-split sensitivity analyses
10. `09_verify_reproduction.py`
   - final array verification and reference-result comparison

## Canonical integrity

Expected generated-array SHA-256 values are stored in:

`CANONICAL_ARRAY_SHA256.txt`

The complete distribution manifest is:

`SHA256SUMS.txt`

`run_all.py` invokes verification before model fitting and again after
analysis completion.

## Quick QA smoke test

```bash
python scripts/01_build_core_arrays.py \
  --set-a /path/to/training_setA.zip \
  --set-b /path/to/training_setB.zip \
  --output-dir qa_work \
  --max-patients-a 100 \
  --max-patients-b 100

python scripts/02_build_trajectory_v2.py \
  --input-dir qa_work \
  --patients-per-block 50
```

The `--max-patients-*` options are for QA only and must be omitted for full
manuscript reproduction.

## Citation

Please use `CITATION.cff` and cite the associated manuscript when using this
repository.

## Archival DOI

The exact frozen archive for the reviewed manuscript is:

`ModelAI_ESWA_EndToEnd_Reproducibility_v1.1.1.zip`

The repository release should be tagged **v1.1.1**, and that exact ZIP should
be attached to the GitHub Release and deposited unchanged in Zenodo/OSF.

**Permanent DOI: pending repository deposit.**

After a DOI is minted, update the paper's Data and Code Availability section
and the GitHub repository description/release notes. Do not invent a DOI.

## Versioning

- `v1.1.1` — current frozen reproducibility release.
- Earlier `v1.1.0` and `v1.0.0` packages are obsolete and should not be used.

Any change to executable code or the frozen specifications after publication
must receive a new version identifier; do not silently overwrite `v1.1.1`.

## License and intellectual property

See `LICENSE`.

The repository is public for research transparency and reproducibility.
PhysioNet source data remain governed by their original terms. ModelAI-related
patent and other intellectual-property rights are not waived by publication of
this repository.

## Reproducibility boundaries

This repository reproduces the reported benchmark analyses. It does not claim
prospective clinical validation, independent external validation, treatment
effect, or deployment safety.

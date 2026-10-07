# GitHub + Zenodo Release Guide

## Recommended GitHub repository name

`ModelAI-Expert-State-Sepsis-ESWA`

Recommended description:

> End-to-end reproducibility code for auditable expert state-machine representations in early sepsis prediction using PhysioNet/CinC 2019.

## 1. Create the GitHub repository

Create a **public** repository with the name above.

Do not initialize it with a separate README, license, or `.gitignore`; this
package already contains them.

## 2. Upload this repository

### Option A — GitHub website

1. Open the new repository.
2. Choose **Add file → Upload files**.
3. Upload the extracted contents of this GitHub-ready ZIP.
4. Commit with:
   `Initial public reproducibility release v1.1.1`

### Option B — Git command line

```bash
git init
git add .
git commit -m "Initial public reproducibility release v1.1.1"
git branch -M main
git remote add origin https://github.com/<YOUR-USERNAME>/ModelAI-Expert-State-Sepsis-ESWA.git
git push -u origin main
```

## 3. Create the frozen GitHub Release

Create tag:

`v1.1.1`

Release title:

`ModelAI ESWA End-to-End Reproducibility v1.1.1`

Attach the **exact unchanged** file:

`ModelAI_ESWA_EndToEnd_Reproducibility_v1.1.1.zip`

Do not re-zip it before attaching.

Expected SHA-256:

`8b78661f675125963d5212c40bc4c696e3e722cb11af98d2bd325d3fecde187a`

Suggested release note:

> Frozen end-to-end reproducibility release for “Auditable Expert State-Machine Representations for Early Sepsis Prediction.” Starts from public PhysioNet/CinC 2019 PSV files and reconstructs all analysis arrays, controls, model fits, calibration, utility, bootstrap, cross-site, and repeated-split analyses. Includes automatic canonical-array integrity verification before training. Raw PhysioNet data are not redistributed.

## 4. Connect GitHub to Zenodo

1. Sign in to Zenodo using your GitHub account.
2. Open the GitHub integration page.
3. Enable archiving for `ModelAI-Expert-State-Sepsis-ESWA`.
4. Create/publish the GitHub `v1.1.1` release.
5. Zenodo should archive the release and mint a DOI.
6. Verify the archived files and metadata before using the DOI in the paper.

Important: the journal reviewer requested a permanent repository identifier for
the exact frozen release. Keep the exact release ZIP unchanged.

## 5. Update the manuscript only after the DOI exists

Replace:

`Repository DOI/URL: [TO BE INSERTED AFTER DEPOSIT OF THE EXACT v1.1.1 ARCHIVE].`

with, for example:

`Code repository: https://github.com/<YOUR-USERNAME>/ModelAI-Expert-State-Sepsis-ESWA. Archived reproducibility release v1.1.1: https://doi.org/<ZENODO-DOI>.`

Do not use the example DOI syntax until Zenodo actually mints the identifier.

## 6. Recommended GitHub topics

- `sepsis`
- `machine-learning`
- `expert-systems`
- `finite-state-machine`
- `clinical-ai`
- `physionet`
- `reproducibility`
- `temporal-data`
- `lightgbm`
- `healthcare-ai`

## 7. Repository settings

Recommended:
- Visibility: Public
- Issues: optional
- Discussions: off unless you plan to support users
- Wiki: off
- Releases: on
- Branch protection for `main`: optional after publication

## 8. What NOT to upload

Do not upload:
- `training_setA` / `training_setB`
- raw `.psv` files
- generated `.npy` / `.npz` arrays
- credentials or `.env` files
- manuscript reviewer correspondence
- patent documents
- unpublished ModelAI source code unrelated to this paper

The `.gitignore` in this package is configured accordingly.

#!/usr/bin/env bash
set -euo pipefail
OUT=${1:-data_raw}
mkdir -p "$OUT"
# Public PhysioNet/CinC 2019 source data. No credentialed access is required for this challenge release.
wget -r -N -c -np -nH --cut-dirs=3 -R 'index.html*' \
  -P "$OUT" \
  https://physionet.org/files/challenge-2019/1.0.0/training/
echo "Downloaded under $OUT. Locate training_setA and training_setB and pass them to 01_build_core_arrays.py."

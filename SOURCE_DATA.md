# Source data requirements

The archive does not redistribute the public PhysioNet data.

Required source: **Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019, version 1.0.0**.

Official page: https://physionet.org/content/challenge-2019/1.0.0/

Canonical source composition used by the manuscript:

- training Set A: 20,336 patient PSV files
- training Set B: 20,000 patient PSV files
- total: 40,336 patients
- generated hourly rows: 1,552,210

Each PSV file must contain the official 41 columns ending in `SepsisLabel`. The raw builder validates column count while parsing.

The pipeline accepts either extracted source directories or ZIP archives. RAR files are not required and are intentionally not a package dependency.

#!/usr/bin/env python3
"""One-command end-to-end reproduction from raw public PhysioNet files to manuscript analyses."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

def run(cmd):
    print('\n>>>',' '.join(map(str,cmd)),flush=True); subprocess.run(list(map(str,cmd)),check=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--set-a',required=True,help='training_setA directory or ZIP')
    ap.add_argument('--set-b',required=True,help='training_setB directory or ZIP')
    ap.add_argument('--work-dir',default='work',help='Generated arrays/intermediates')
    ap.add_argument('--results-dir',default='results_reproduced',help='Model predictions/tables')
    ap.add_argument('--skip-crosssite',action='store_true')
    ap.add_argument('--skip-resplits',action='store_true')
    args=ap.parse_args()
    root=Path(__file__).resolve().parent; scripts=root/'scripts'; work=Path(args.work_dir).resolve(); results=Path(args.results_dir).resolve(); work.mkdir(parents=True,exist_ok=True); results.mkdir(parents=True,exist_ok=True)
    py=sys.executable
    run([py,scripts/'01_build_core_arrays.py','--set-a',Path(args.set_a).resolve(),'--set-b',Path(args.set_b).resolve(),'--output-dir',work])
    run([py,scripts/'02_build_trajectory_v2.py','--input-dir',work])
    run([py,scripts/'03_build_controls.py','--input-dir',work])
    # Mandatory pre-training integrity gate: verify every canonical ALL_* array and split hash
    # before any model fitting or metric comparison is allowed to proceed.
    run([py,scripts/'09_verify_reproduction.py','--array-dir',work])
    run([py,scripts/'04_train_all_arms.py','--input-dir',work,'--output-dir',results])
    run([py,scripts/'05_calibration_utility.py','--array-dir',work,'--prediction-dir',results,'--output-dir',results])
    run([py,scripts/'06_bootstrap_primary.py','--array-dir',work,'--prediction-dir',results,'--output-dir',results,'--n-bootstrap','2000'])
    if not args.skip_crosssite: run([py,scripts/'07_crosssite.py','--array-dir',work,'--output-dir',results,'--bootstrap','500'])
    if not args.skip_resplits: run([py,scripts/'08_repeated_splits.py','--array-dir',work,'--output-dir',results])
    # Final verification also compares the reproduced multiseed table with the archived reference results.
    run([py,scripts/'09_verify_reproduction.py','--array-dir',work,'--results-dir',results,'--reference-dir',root/'results_reference'])
    print('\nEND-TO-END REPRODUCTION COMPLETE')
    print('Arrays:',work); print('Results:',results)
if __name__=='__main__': main()

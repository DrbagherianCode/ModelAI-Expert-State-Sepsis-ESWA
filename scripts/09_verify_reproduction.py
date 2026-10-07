#!/usr/bin/env python3
"""Verify generated canonical array hashes and compare reproduced result tables to archived references."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import pandas as pd

CANONICAL={
'ALL_Xb.npy':'989faa256a7d195cba68a3d0482e28e3afa12b6f30f9e84abc9f018d4e52d2a5',
'ALL_Xs.npy':'abaebd540e8e91f15e8cf038b92f52ab07357d332d42e85bb758141b45ec1d77',
'ALL_Xtraj_v2.npy':'5a39d35cdcefe6aae96acc5481f203a98c772166ca1650f76fe822221924bcd0',
'ALL_Xforward_v2.npy':'a23ddfc91bcdfae5ae258c969b87c5ad4e60170ad66389e474eca39d725f3f28',
'ALL_y.npy':'aaa2c7502c7dddcb6012ad0b16edf2d680b14e1a73dc0192cfded353cbf9015c',
'ALL_pid.npy':'79da5f0fbae3beff08e9253c5d0aee410854652ef262e6b863bc51f52810344f',
'ALL_Xgeneric27.npy':'1d602b41ae066cbd142e1c553cf33bfba2875ead53235178a0924ae8e862d1df',
'ALL_Xnoise27.npy':'faf67a92f13b5a9fd34dcc52260ee1d16f1c1fe4ab79090f99799bab9c0c5d4c',
'split_seed42.npz':'f8af46ebf6cf6935681e988c394986661b905afffb3a86247e6c92155e437932'}

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--array-dir',required=True); ap.add_argument('--results-dir',default=None); ap.add_argument('--reference-dir',default=None); args=ap.parse_args()
    arr=Path(args.array_dir); ok=True
    print('Canonical array verification:')
    for name,expected in CANONICAL.items():
        p=arr/name
        if not p.exists(): print('MISSING',name); ok=False; continue
        got=sha256(p); match=got==expected; ok &= match; print(('OK' if match else 'MISMATCH'),name,got)
    if args.results_dir and args.reference_dir:
        got=Path(args.results_dir)/'ModelAI_Revision_Multiseed_AllArms.csv'; ref=Path(args.reference_dir)/'ModelAI_Revision_Multiseed_AllArms.csv'
        if got.exists() and ref.exists():
            a=pd.read_csv(got).sort_values(['arm','seed']).reset_index(drop=True); b=pd.read_csv(ref).sort_values(['arm','seed']).reset_index(drop=True)
            cols=['AUPRC','AUROC','Brier_raw']; print('Result-table comparison tolerance: 1e-7.')
            for c in cols:
                if c in a and c in b:
                    diff=float((a[c]-b[c]).abs().max()); print(c,'max_abs_diff',diff); ok &= diff<1e-7
    if not ok: raise SystemExit(2)
    print('VERIFICATION PASSED')
if __name__=='__main__': main()

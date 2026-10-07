#!/usr/bin/env python3
"""2,000 patient-bootstrap replicates for primary/secondary ensemble contrasts."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score

ARMS=['Baseline','CompactTrajectory20','GenericTemporal27']; SEEDS=[11,23,42,73,101]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--array-dir',required=True); ap.add_argument('--prediction-dir',required=True); ap.add_argument('--output-dir',required=True)
    ap.add_argument('--n-bootstrap',type=int,default=2000); ap.add_argument('--seed',type=int,default=20261006); args=ap.parse_args()
    arr=Path(args.array_dir); prd=Path(args.prediction_dir); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    yall=np.load(arr/'ALL_y.npy',mmap_mode='r'); pidall=np.load(arr/'ALL_pid.npy',mmap_mode='r'); s=np.load(arr/'split_seed42.npz'); te=s['test']; ite=np.where(np.isin(pidall,te))[0]
    y=np.asarray(yall[ite]); pid=np.asarray(pidall[ite]); preds={a:np.mean(np.vstack([np.load(prd/f'rev_{a}_{sd}_test.npy') for sd in SEEDS]),axis=0).astype(float) for a in ARMS}
    _,inv=np.unique(pid,return_inverse=True); n_pat=int(inv.max()+1); rng=np.random.default_rng(args.seed); A=np.zeros((args.n_bootstrap,6),float)
    for b in range(args.n_bootstrap):
        counts=rng.multinomial(n_pat,np.full(n_pat,1.0/n_pat)); w=counts[inv]
        ap={a:average_precision_score(y,preds[a],sample_weight=w) for a in ARMS}; auc={a:roc_auc_score(y,preds[a],sample_weight=w) for a in ARMS}
        A[b]=[ap['CompactTrajectory20']-ap['Baseline'],auc['CompactTrajectory20']-auc['Baseline'],ap['GenericTemporal27']-ap['Baseline'],auc['GenericTemporal27']-auc['Baseline'],ap['CompactTrajectory20']-ap['GenericTemporal27'],auc['CompactTrajectory20']-auc['GenericTemporal27']]
        if (b+1)%200==0: print('bootstrap',b+1)
    names=['Compact_vs_Base_dAUPRC','Compact_vs_Base_dAUROC','Generic_vs_Base_dAUPRC','Generic_vs_Base_dAUROC','Compact_vs_Generic_dAUPRC','Compact_vs_Generic_dAUROC']
    res={}
    for j,nm in enumerate(names):
        x=A[:,j]; res[nm]={'mean':float(x.mean()),'median':float(np.median(x)),'ci95':[float(np.quantile(x,.025)),float(np.quantile(x,.975))],'p_gt0':float(np.mean(x>0))}
    np.save(out/'ModelAI_Revision_Bootstrap2000_raw.npy',A); (out/'ModelAI_Revision_Bootstrap2000.json').write_text(json.dumps(res,indent=2),encoding='utf-8'); print(json.dumps(res,indent=2))
if __name__=='__main__': main()

#!/usr/bin/env python3
"""Repeated patient-level split sensitivity for Baseline, CompactTrajectory20 and GenericTemporal27."""
from __future__ import annotations
import argparse, gc
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score,roc_auc_score

SPLITS=[31415,27182,16180]; COMPACT=[6,13,20,27,34,41,48,55]+list(range(56,68))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--array-dir',required=True); ap.add_argument('--output-dir',required=True); args=ap.parse_args(); arr=Path(args.array_dir); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    Xb=np.load(arr/'ALL_Xb.npy',mmap_mode='r'); Xs=np.load(arr/'ALL_Xs.npy',mmap_mode='r'); Xt=np.load(arr/'ALL_Xtraj_v2.npy',mmap_mode='r'); Xg=np.load(arr/'ALL_Xgeneric27.npy',mmap_mode='r'); y=np.load(arr/'ALL_y.npy',mmap_mode='r'); pid=np.load(arr/'ALL_pid.npy',mmap_mode='r')
    up,starts=np.unique(pid,return_index=True); ends=np.r_[starts[1:],len(pid)]; pospat=np.array([int(np.asarray(y[a:b]).max()) for a,b in zip(starts,ends)]); meta=pd.read_csv(arr/'patient_manifest.csv').set_index('patient_idx'); site=np.array([0 if meta.loc[int(p),'site']=='A' else 1 for p in up]); strata=np.array([f'{a}_{b}' for a,b in zip(site,pospat)]); idxmap={int(p):i for i,p in enumerate(up)}; rows=[]
    for splitseed in SPLITS:
        tr,tmp=train_test_split(up,test_size=.30,random_state=splitseed,stratify=strata); stmp=np.array([strata[idxmap[int(p)]] for p in tmp]); va,te=train_test_split(tmp,test_size=.5,random_state=splitseed,stratify=stmp)
        itr=np.where(np.isin(pid,tr))[0]; ite=np.where(np.isin(pid,te))[0]; spw=float((len(itr)-y[itr].sum())/y[itr].sum())
        for arm in ['Baseline','CompactTrajectory20','GenericTemporal27']:
            trp=[np.asarray(Xb[itr])]; tep=[np.asarray(Xb[ite])]
            if arm=='CompactTrajectory20': trp += [np.asarray(Xs[itr,40:47]),np.asarray(Xt[itr][:,COMPACT])]; tep += [np.asarray(Xs[ite,40:47]),np.asarray(Xt[ite][:,COMPACT])]
            elif arm=='GenericTemporal27': trp.append(np.asarray(Xg[itr])); tep.append(np.asarray(Xg[ite]))
            Xtr=trp[0] if len(trp)==1 else np.concatenate(trp,1); Xte=tep[0] if len(tep)==1 else np.concatenate(tep,1)
            m=lgb.LGBMClassifier(objective='binary',n_estimators=100,learning_rate=.05,num_leaves=31,min_child_samples=80,colsample_bytree=1.0,subsample=.9,subsample_freq=1,reg_lambda=1.0,random_state=42,n_jobs=-1,verbosity=-1,scale_pos_weight=spw)
            m.fit(Xtr,y[itr]); p=m.predict_proba(Xte)[:,1]; r={'split_seed':splitseed,'arm':arm,'n_train_patients':len(tr),'n_test_patients':len(te),'AUPRC':float(average_precision_score(y[ite],p)),'AUROC':float(roc_auc_score(y[ite],p))}; rows.append(r); print(r)
            del Xtr,Xte,trp,tep,m,p; gc.collect()
    pd.DataFrame(rows).to_csv(out/'ModelAI_Revision_RepeatedSplit_Sensitivity.csv',index=False)
if __name__=='__main__': main()

#!/usr/bin/env python3
"""Five-seed A->B and B->A internal transportability analysis plus patient bootstrap."""
from __future__ import annotations
import argparse, gc, json
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss

SEEDS=[11,23,42,73,101]; ARMS=['Baseline','CompactTrajectory20','GenericTemporal27']; COMPACT=[6,13,20,27,34,41,48,55]+list(range(56,68))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--array-dir',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--bootstrap',type=int,default=500); args=ap.parse_args()
    arr=Path(args.array_dir); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    Xb=np.load(arr/'ALL_Xb.npy',mmap_mode='r'); Xs=np.load(arr/'ALL_Xs.npy',mmap_mode='r'); Xt=np.load(arr/'ALL_Xtraj_v2.npy',mmap_mode='r'); Xg=np.load(arr/'ALL_Xgeneric27.npy',mmap_mode='r'); y=np.load(arr/'ALL_y.npy',mmap_mode='r'); pid=np.load(arr/'ALL_pid.npy',mmap_mode='r')
    meta=pd.read_csv(arr/'patient_manifest.csv'); A_pat=meta.loc[meta.site=='A','patient_idx'].to_numpy(); B_pat=meta.loc[meta.site=='B','patient_idx'].to_numpy()
    rows=[]
    for direction in ['AtoB','BtoA']:
        if direction=='AtoB': itr=np.where(np.isin(pid,A_pat))[0]; ite=np.where(np.isin(pid,B_pat))[0]
        else: itr=np.where(np.isin(pid,B_pat))[0]; ite=np.where(np.isin(pid,A_pat))[0]
        spw=float((len(itr)-y[itr].sum())/y[itr].sum()); preds={a:[] for a in ARMS}
        for arm in ARMS:
            tr=[np.asarray(Xb[itr])]; te=[np.asarray(Xb[ite])]
            if arm=='CompactTrajectory20': tr += [np.asarray(Xs[itr,40:47]),np.asarray(Xt[itr][:,COMPACT])]; te += [np.asarray(Xs[ite,40:47]),np.asarray(Xt[ite][:,COMPACT])]
            elif arm=='GenericTemporal27': tr += [np.asarray(Xg[itr])]; te += [np.asarray(Xg[ite])]
            Xtr=tr[0] if len(tr)==1 else np.concatenate(tr,1); Xte=te[0] if len(te)==1 else np.concatenate(te,1)
            for seed in SEEDS:
                m=lgb.LGBMClassifier(objective='binary',n_estimators=100,learning_rate=.05,num_leaves=31,min_child_samples=80,colsample_bytree=1.0,subsample=.9,subsample_freq=1,reg_lambda=1.0,random_state=seed,n_jobs=-1,verbosity=-1,scale_pos_weight=spw)
                m.fit(Xtr,y[itr]); p=m.predict_proba(Xte)[:,1].astype(np.float32); preds[arm].append(p)
                r={'direction':direction,'seed':seed,'arm':arm,'AUPRC':float(average_precision_score(y[ite],p)),'AUROC':float(roc_auc_score(y[ite],p)),'Brier_raw':float(brier_score_loss(y[ite],p))}; rows.append(r); print(r)
                del m; gc.collect()
            del Xtr,Xte,tr,te; gc.collect()
        ens={a:np.mean(np.vstack(preds[a]),axis=0) for a in ARMS}; testpid=np.asarray(pid[ite]); yt=np.asarray(y[ite]); _,inv=np.unique(testpid,return_inverse=True); npat=int(inv.max()+1); rng=np.random.default_rng(20261006+(0 if direction=='AtoB' else 1000))
        B=np.zeros((args.bootstrap,4))
        for b in range(args.bootstrap):
            counts=rng.multinomial(npat,np.full(npat,1/npat)); w=counts[inv]
            aps={a:average_precision_score(yt,ens[a],sample_weight=w) for a in ARMS}; aucs={a:roc_auc_score(yt,ens[a],sample_weight=w) for a in ARMS}
            B[b]=[aps['CompactTrajectory20']-aps['Baseline'],aucs['CompactTrajectory20']-aucs['Baseline'],aps['GenericTemporal27']-aps['Baseline'],aucs['GenericTemporal27']-aucs['Baseline']]
        names=['Compact_vs_Base_dAUPRC','Compact_vs_Base_dAUROC','Generic_vs_Base_dAUPRC','Generic_vs_Base_dAUROC']; br={}
        for j,n in enumerate(names): br[n]={'mean':float(B[:,j].mean()),'ci95':[float(np.quantile(B[:,j],.025)),float(np.quantile(B[:,j],.975))]}
        summary={'ensemble':{a:{'AUPRC':float(average_precision_score(yt,ens[a])),'AUROC':float(roc_auc_score(yt,ens[a]))} for a in ARMS},f'bootstrap{args.bootstrap}':br}
        (out/f'ModelAI_Revision_CrossSite_{direction}_Ensemble_Bootstrap{args.bootstrap}.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    pd.DataFrame(rows).to_csv(out/'ModelAI_Revision_CrossSite_Multiseed.csv',index=False)
if __name__=='__main__': main()

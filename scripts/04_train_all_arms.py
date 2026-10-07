#!/usr/bin/env python3
"""Train all manuscript arms with identical LightGBM settings and save validation/test predictions."""
from __future__ import annotations
import argparse, gc, json
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss

ARMS=['Baseline','CurrentStates','CompactTrajectory20','GenericTemporal27','Noise27','NonSOFAExpert9']
SEEDS=[11,23,42,73,101]
COMPACT=[6,13,20,27,34,41,48,55]+list(range(56,68))
NONSOFA=[41,43,45,48,50,52,55,57,59]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input-dir',required=True); ap.add_argument('--output-dir',required=True)
    ap.add_argument('--arms',default=','.join(ARMS)); ap.add_argument('--seeds',default=','.join(map(str,SEEDS))); args=ap.parse_args()
    inp=Path(args.input_dir); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    arms=args.arms.split(','); seeds=[int(x) for x in args.seeds.split(',')]
    Xb=np.load(inp/'ALL_Xb.npy',mmap_mode='r'); Xs=np.load(inp/'ALL_Xs.npy',mmap_mode='r'); Xt=np.load(inp/'ALL_Xtraj_v2.npy',mmap_mode='r')
    y=np.load(inp/'ALL_y.npy',mmap_mode='r'); pid=np.load(inp/'ALL_pid.npy',mmap_mode='r'); spl=np.load(inp/'split_seed42.npz')
    Xg=np.load(inp/'ALL_Xgeneric27.npy',mmap_mode='r'); Xn=np.load(inp/'ALL_Xnoise27.npy',mmap_mode='r')
    tr,va,te=spl['train'],spl['val'],spl['test']; itr=np.where(np.isin(pid,tr))[0]; iva=np.where(np.isin(pid,va))[0]; ite=np.where(np.isin(pid,te))[0]
    pos=y[itr].sum(); spw=float((len(itr)-pos)/pos)
    def parts(arm,idx):
        pp=[np.asarray(Xb[idx])]
        if arm=='CurrentStates': pp.append(np.asarray(Xs[idx,40:47]))
        elif arm=='CompactTrajectory20': pp.extend([np.asarray(Xs[idx,40:47]),np.asarray(Xt[idx][:,COMPACT])])
        elif arm=='GenericTemporal27': pp.append(np.asarray(Xg[idx]))
        elif arm=='Noise27': pp.append(np.asarray(Xn[idx]))
        elif arm=='NonSOFAExpert9': pp.append(np.asarray(Xs[idx][:,NONSOFA]))
        elif arm!='Baseline': raise ValueError(arm)
        return pp[0] if len(pp)==1 else np.concatenate(pp,axis=1)
    rows=[]; valrows=[]
    for arm in arms:
        Xtr=parts(arm,itr); Xv=parts(arm,iva); Xte=parts(arm,ite)
        for seed in seeds:
            params=dict(objective='binary',n_estimators=100,learning_rate=.05,num_leaves=31,min_child_samples=80,
                        colsample_bytree=1.0,subsample=.9,subsample_freq=1,reg_lambda=1.0,random_state=seed,n_jobs=-1,
                        verbosity=-1,scale_pos_weight=spw)
            m=lgb.LGBMClassifier(**params); m.fit(Xtr,y[itr]); pv=m.predict_proba(Xv)[:,1].astype(np.float32); pt=m.predict_proba(Xte)[:,1].astype(np.float32)
            r={'seed':seed,'arm':arm,'n_features':int(Xtr.shape[1]),'val_AUPRC':float(average_precision_score(y[iva],pv)),
               'val_AUROC':float(roc_auc_score(y[iva],pv)),'test_AUPRC':float(average_precision_score(y[ite],pt)),
               'test_AUROC':float(roc_auc_score(y[ite],pt)),'test_Brier_raw':float(brier_score_loss(y[ite],pt))}
            rows.append({'arm':arm,'seed':seed,'AUPRC':r['test_AUPRC'],'AUROC':r['test_AUROC'],'Brier_raw':r['test_Brier_raw']}); valrows.append({'arm':arm,'seed':seed,'AUPRC':r['val_AUPRC'],'AUROC':r['val_AUROC']}); np.save(out/f'rev_{arm}_{seed}_val.npy',pv); np.save(out/f'rev_{arm}_{seed}_test.npy',pt); print(r,flush=True)
            del m,pv,pt; gc.collect()
        del Xtr,Xv,Xte; gc.collect()
    df=pd.DataFrame(rows); df.to_csv(out/'ModelAI_Revision_Multiseed_AllArms.csv',index=False); pd.DataFrame(valrows).to_csv(out/'ModelAI_Revision_Validation_Multiseed.csv',index=False)
    meta={'arms':arms,'seeds':seeds,'scale_pos_weight':spw,'compact_indices':COMPACT,'nonsofa_indices':NONSOFA}
    (out/'training_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
if __name__=='__main__': main()

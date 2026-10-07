#!/usr/bin/env python3
"""Build GenericTemporal27 and Noise27 controls from canonical core arrays."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from numpy.lib.format import open_memmap

DYN=['HR','O2Sat','Temp','SBP','MAP','DBP','Resp','EtCO2','BaseExcess','HCO3','FiO2','pH','PaCO2','SaO2','AST','BUN','Alkalinephos','Calcium','Chloride','Creatinine','Bilirubin_direct','Glucose','Lactate','Magnesium','Phosphate','Potassium','Bilirubin_total','TroponinI','Hct','Hgb','PTT','WBC','Fibrinogen','Platelets']

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input-dir',required=True); ap.add_argument('--output-dir',default=None); args=ap.parse_args()
    inp=Path(args.input_dir); outdir=Path(args.output_dir or args.input_dir); outdir.mkdir(parents=True,exist_ok=True)
    X=np.load(inp/'ALL_Xb.npy',mmap_mode='r'); pid=np.load(inp/'ALL_pid.npy',mmap_mode='r'); s=np.load(inp/'split_seed42.npz'); tr=s['train']
    trainmask=np.isin(pid,tr)
    dens=[]
    for i,c in enumerate(DYN): dens.append((float(np.isfinite(X[trainmask,i]).mean()),c,i))
    sel=sorted(dens,reverse=True)[:9]; idx=[x[2] for x in sel]; names=[x[1] for x in sel]
    N=len(pid); gout=open_memmap(outdir/'ALL_Xgeneric27.npy',mode='w+',dtype=np.float32,shape=(N,27)); gout[:]=np.nan
    starts=np.r_[0,np.where(pid[1:]!=pid[:-1])[0]+1]; ends=np.r_[starts[1:],N]
    for pi,(a,b) in enumerate(zip(starts,ends)):
        vals=np.asarray(X[a:b,idx],dtype=np.float64); n=b-a; f=np.full((n,27),np.nan,dtype=np.float32)
        for j in range(9):
            v=vals[:,j]
            for t in range(n):
                lo=max(0,t-5); w=v[lo:t+1]; obs=np.isfinite(w)
                if not obs.any(): continue
                ww=w[obs]; base=3*j; f[t,base]=ww.mean(); f[t,base+1]=ww.max()-ww.min()
                if obs.sum()>=2:
                    tt=np.arange(lo,t+1,dtype=float)[obs]; xm=tt.mean(); ym=ww.mean(); denom=((tt-xm)**2).sum()
                    if denom>0: f[t,base+2]=((tt-xm)*(ww-ym)).sum()/denom
        gout[a:b]=f
        if (pi+1)%5000==0: print(f'GenericTemporal27: {pi+1:,}/{len(starts):,} patients')
    gout.flush()
    rng=np.random.default_rng(20261006)
    nout=open_memmap(outdir/'ALL_Xnoise27.npy',mode='w+',dtype=np.float32,shape=(N,27))
    # Generate in one call to reproduce the canonical RNG stream exactly.
    nout[:]=rng.normal(0,1,size=(N,27)).astype(np.float32); nout.flush()
    meta={'selection_rule':'Top 9 dynamic variables by non-missing observation density among training patients only; no labels used.',
          'window_hours':6,'features_per_variable':['rolling observed mean','rolling observed range (max-min)','least-squares slope over observed points'],
          'selected_variables':[{'name':c,'column_index':int(i),'training_observation_density':d} for d,c,i in sel],
          'ordered_feature_names':[f'{c}_{suf}' for _,c,_ in sel for suf in ['mean6h','range6h','slope6h']],
          'noise27':{'generator':'numpy.random.default_rng','seed':20261006,'distribution':'Normal(0,1)','dtype':'float32'}}
    (outdir/'ModelAI_GenericTemporal27_Spec_generated.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('Selected:',names); print('Done.')
if __name__=='__main__': main()

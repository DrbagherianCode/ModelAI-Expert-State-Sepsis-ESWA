#!/usr/bin/env python3
"""Build the canonical 68-feature trajectory bank and 18-feature forward-scenario bank.

This script consumes ALL_Xs.npy and ALL_pid.npy produced by 01_build_core_arrays.py.
It is patient-boundary-safe and processes patient blocks to limit memory usage.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from numpy.lib.format import open_memmap

RANK_MAP=np.array([0,1,2,3,4,2,5],dtype=np.float32)

def build_block(Xs,pid):
    S=np.asarray(Xs[:,40:47],dtype=np.int8)
    dwell=np.asarray(Xs[:,47:54],dtype=np.float32)
    trans6=np.asarray(Xs[:,54:61],dtype=np.float32)
    rk=RANK_MAP[S]; N=len(pid)
    def lag_safe(a,k):
        out=np.empty_like(a,dtype=np.float32); valid=np.zeros(N,dtype=bool)
        if N<=k:
            out[:]=a
            return out,valid
        out[:k]=a[:k]
        v=(pid[k:]==pid[:-k]); valid[k:]=v
        out[k:]=a[k:]
        tmp=out[k:]; tmp[v]=a[:-k][v]; out[k:]=tmp
        return out,valid
    lags={}; valids={}
    for k in (1,2,3,4,5,6,7,8,9,10,11): lags[k],valids[k]=lag_safe(rk,k)
    parts=[lags[1],lags[3],lags[6], rk-lags[1],rk-lags[3],rk-lags[6]]
    sum6=rk.copy(); cnt=np.ones((N,1),np.float32); max6=rk.copy()
    for k in range(1,6):
        v=valids[k][:,None].astype(np.float32); sum6+=lags[k]*v; cnt+=v; max6=np.maximum(max6,np.where(v>0,lags[k],-np.inf))
    mean6=sum6/cnt; parts += [max6,mean6]
    T=np.concatenate(parts,axis=1).astype(np.float32) # 56
    agg=np.zeros((N,12),np.float32)
    agg[:,0]=(rk[:,:6]>=2).sum(1); agg[:,1]=(rk[:,:6]>=3).sum(1); agg[:,2]=(rk[:,:6]>=4).sum(1)
    agg[:,3]=np.clip(rk[:,:6]-lags[3][:,:6],0,None).sum(1)
    agg[:,4]=np.clip(rk[:,:6]-lags[6][:,:6],0,None).sum(1)
    agg[:,5]=np.clip(lags[3][:,:6]-rk[:,:6],0,None).sum(1)
    burden=rk[:,:6].sum(1).astype(np.float32)
    def rolling_mean_scalar(x,w):
        total=x.copy(); count=np.ones(N,np.float32)
        for k in range(1,w):
            lg,_=lag_safe(x[:,None],k); v=valids[k].astype(np.float32); total+=lg[:,0]*v; count+=v
        return total/count
    agg[:,6]=rolling_mean_scalar(burden,6); agg[:,7]=rolling_mean_scalar(burden,12)
    wors=((rk[:,:6]>lags[1][:,:6]) & valids[1][:,None]); agg[:,8]=wors.sum(1)
    wc=agg[:,8]; maxwc=wc.copy()
    for k in range(1,6):
        lg,_=lag_safe(wc[:,None],k); maxwc=np.maximum(maxwc,lg[:,0]*valids[k])
    agg[:,9]=maxwc
    new=((rk[:,:6]>=2)&(lags[1][:,:6]<2)&valids[1][:,None]).sum(1).astype(np.float32); newsum=new.copy()
    for k in range(1,6):
        lg,_=lag_safe(new[:,None],k); newsum += lg[:,0]*valids[k]
    agg[:,10]=newsum; agg[:,11]=trans6[:,:6].sum(1)
    traj=np.concatenate([T,agg],axis=1).astype(np.float32)
    fwd=np.zeros((N,18),np.float32)
    recent_worse=(rk-lags[3])>0; pressure=recent_worse | ((dwell>=3)&(rk>=2))
    proj=rk+pressure.astype(np.float32); proj[:,:6]=np.clip(proj[:,:6],0,4); proj[:,6]=np.clip(proj[:,6],0,5)
    fwd[:,:7]=proj; fwd[:,7:14]=pressure.astype(np.float32)
    fwd[:,14]=(proj[:,:6]>=3).sum(1); fwd[:,15]=(proj[:,:6]>=4).sum(1)
    fwd[:,16]=((proj[:,:6]>=3).sum(1)>=3); fwd[:,17]=(S[:,:6]==0).sum(1)
    return traj,fwd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input-dir',required=True); ap.add_argument('--output-dir',default=None); ap.add_argument('--patients-per-block',type=int,default=1500)
    args=ap.parse_args(); inp=Path(args.input_dir); out=Path(args.output_dir or args.input_dir); out.mkdir(parents=True,exist_ok=True)
    Xs=np.load(inp/'ALL_Xs.npy',mmap_mode='r'); pid=np.load(inp/'ALL_pid.npy',mmap_mode='r'); N=len(pid)
    traj=open_memmap(out/'ALL_Xtraj_v2.npy',mode='w+',dtype=np.float32,shape=(N,68))
    fwd=open_memmap(out/'ALL_Xforward_v2.npy',mode='w+',dtype=np.float32,shape=(N,18))
    # patient row boundaries
    starts=np.r_[0,np.where(pid[1:]!=pid[:-1])[0]+1]; ends=np.r_[starts[1:],N]
    P=len(starts); bsz=args.patients_per_block
    for p0 in range(0,P,bsz):
        p1=min(P,p0+bsz); r0=int(starts[p0]); r1=int(ends[p1-1])
        tb,fb=build_block(np.asarray(Xs[r0:r1]),np.asarray(pid[r0:r1]))
        traj[r0:r1]=tb; fwd[r0:r1]=fb
        print(f'patients {p0:,}-{p1:,} / {P:,}; rows {r0:,}-{r1:,}')
    traj.flush(); fwd.flush()
    meta={'trajectory_shape':[int(N),68],'forward_shape':[int(N),18],
          'compact_trajectory20_indices':[6,13,20,27,34,41,48,55]+list(range(56,68)),
          'rank_map':{'UNKNOWN':0,'STABLE':1,'AT_RISK_OR_ISOLATED':2,'DYSFUNCTION_OR_MULTISYSTEM':3,'SEVERE_OR_PROGRESSIVE':4,'RECOVERING':2,'HIGH_RISK':5}}
    (out/'trajectory_v2_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('Done:',out/'ALL_Xtraj_v2.npy',out/'ALL_Xforward_v2.npy')
if __name__=='__main__': main()

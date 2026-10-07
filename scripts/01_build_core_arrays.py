#!/usr/bin/env python3
"""Build the canonical ModelAI Sepsis-2019 baseline and symbolic arrays from public raw PSV files.

Input
-----
Either directories containing *.psv files or ZIP archives for training_setA and training_setB.

Output
------
ALL_Xb.npy   : (N,85) float32 baseline [40 raw + 15 obs + 15 time-since + 15 delta]
ALL_Xs.npy   : (N,65) float32 symbolic [20 alert active + 20 alert eval + 7 state + 7 dwell + 7 transitions6h + 4 summaries]
ALL_y.npy    : (N,) uint8 hourly SepsisLabel
ALL_pid.npy  : (N,) int32 canonical patient index (A first, then B)
patient_manifest.csv
split_seed42.npz : patient-disjoint 70/15/15 stratified split used in the paper

No outcome label is used to construct predictors. All temporal features are patient-boundary-safe and causal.
"""
from __future__ import annotations
import argparse, io, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from sklearn.model_selection import train_test_split

COLS=['HR','O2Sat','Temp','SBP','MAP','DBP','Resp','EtCO2','BaseExcess','HCO3','FiO2','pH','PaCO2','SaO2','AST','BUN','Alkalinephos','Calcium','Chloride','Creatinine','Bilirubin_direct','Glucose','Lactate','Magnesium','Phosphate','Potassium','Bilirubin_total','TroponinI','Hct','Hgb','PTT','WBC','Fibrinogen','Platelets','Age','Gender','Unit1','Unit2','HospAdmTime','ICULOS','SepsisLabel']
IDX={c:i for i,c in enumerate(COLS)}
KEY=['HR','O2Sat','Temp','SBP','MAP','Resp','HCO3','FiO2','pH','PaCO2','Creatinine','Lactate','Bilirubin_total','WBC','Platelets']
BASE_NAMES=COLS[:-1]+[f'{c}_obs' for c in KEY]+[f'{c}_tsince' for c in KEY]+[f'{c}_delta' for c in KEY]
ALERT_NAMES=[f'A{i:02d}' for i in range(1,21)]
SYM_NAMES=[f'{a}_active' for a in ALERT_NAMES]+[f'{a}_eval' for a in ALERT_NAMES]
SYM_NAMES += [f'SM{i}_state' for i in range(1,8)]+[f'SM{i}_dwell' for i in range(1,8)]+[f'SM{i}_transitions6h' for i in range(1,8)]
SYM_NAMES += ['domains_atrisk','domains_dysfunction','worsening_transitions6h','trajectory_direction']

class PSVSource:
    def __init__(self, path: Path):
        self.path=Path(path)
        self.z=None
        if self.path.is_dir():
            self.mode='dir'
            self.names=sorted([p for p in self.path.rglob('*.psv')])
        elif self.path.suffix.lower()=='.zip':
            self.mode='zip'; self.z=zipfile.ZipFile(self.path)
            self.names=sorted([n for n in self.z.namelist() if n.lower().endswith('.psv')])
        else:
            raise ValueError(f'Unsupported input {self.path}. Use an extracted directory or ZIP archive.')
        if not self.names:
            raise ValueError(f'No .psv files found in {self.path}')
    def name(self, item):
        return item.name if self.mode=='dir' else Path(item).name
    def read_bytes(self,item):
        return item.read_bytes() if self.mode=='dir' else self.z.read(item)
    def count_rows(self,item):
        b=self.read_bytes(item)
        # one header line + non-empty data rows
        return max(0, len([x for x in b.splitlines() if x.strip()])-1)
    def parse(self,item):
        b=self.read_bytes(item)
        body=b.split(b'\n',1)[1].strip().replace(b'\r',b'').replace(b'\n',b'|')
        arr=np.fromstring(body.decode('ascii'),sep='|',dtype=np.float64)
        if arr.size % len(COLS):
            raise ValueError(f'Malformed PSV {self.name(item)}: {arr.size} values not divisible by {len(COLS)}')
        return arr.reshape(-1,len(COLS))
    def close(self):
        if self.z is not None: self.z.close()

def rollsum(x,w):
    x=x.astype(np.int16,copy=False); cs=np.cumsum(x,dtype=np.int32); out=cs.copy()
    if len(x)>w: out[w:]=cs[w:]-cs[:-w]
    return out

def tsince_delta(v):
    n=len(v); ii=np.arange(n); obs=np.isfinite(v)
    last=np.maximum.accumulate(np.where(obs,ii,-1))
    ts=np.where(last>=0,ii-last,np.nan).astype(np.float32)
    ts=np.minimum(ts,48,where=np.isfinite(ts),out=ts)
    prev=np.r_[-1,last[:-1]]; d=np.full(n,np.nan,dtype=np.float32)
    good=obs & (prev>=0); d[good]=(v[good]-v[prev[good]]).astype(np.float32)
    return obs.astype(np.float32),ts,d

def alerts(a):
    n=len(a); A=np.zeros((n,20),np.float32); E=np.zeros((n,20),np.float32)
    def val(c): return a[:,IDX[c]]
    def simple(j, cols, cond):
        ev=np.ones(n,bool)
        for c in cols: ev &= np.isfinite(val(c))
        E[:,j]=ev; A[:,j]=ev & cond
    simple(0,['SBP'],val('SBP')<=100)
    simple(1,['MAP'],val('MAP')<70)
    simple(2,['MAP'],val('MAP')<65)
    mp=val('MAP'); ev=np.isfinite(mp); prevmax=np.full(n,np.nan)
    for i in range(n):
        p=mp[max(0,i-3):i]; p=p[np.isfinite(p)]
        if len(p): prevmax[i]=p.max()
    e=ev & np.isfinite(prevmax); E[:,3]=e; A[:,3]=e & (mp<75) & ((prevmax-mp)>=10)
    simple(4,['Resp'],val('Resp')>=22)
    simple(5,['O2Sat'],val('O2Sat')<92)
    simple(6,['FiO2','O2Sat'],(val('FiO2')>=0.40)&(val('O2Sat')<94))
    simple(7,['Temp'],(val('Temp')>38)|(val('Temp')<36))
    simple(8,['WBC'],(val('WBC')>12)|(val('WBC')<4))
    simple(9,['Lactate'],val('Lactate')>2)
    lac=val('Lactate')
    for i in range(n):
        if np.isfinite(lac[i]):
            candidates=[j for j in range(i-1,max(-1,i-7),-1) if np.isfinite(lac[j])]
            if candidates:
                j=candidates[0]; E[i,10]=1; A[i,10]=(lac[i]-lac[j]>=0.5)
    simple(11,['Creatinine'],val('Creatinine')>=1.2)
    simple(12,['Creatinine'],val('Creatinine')>=2.0)
    cr=val('Creatinine')
    for i in range(n):
        if np.isfinite(cr[i]):
            p=cr[max(0,i-24):i]; p=p[np.isfinite(p)]
            if len(p): E[i,13]=1; A[i,13]=(cr[i]-p.min()>=0.3)
    simple(14,['Bilirubin_total'],val('Bilirubin_total')>=1.2)
    simple(15,['Bilirubin_total'],val('Bilirubin_total')>=2.0)
    simple(16,['Platelets'],val('Platelets')<150)
    simple(17,['Platelets'],val('Platelets')<100)
    simple(18,['pH','HCO3'],(val('pH')<7.35)&(val('HCO3')<22))
    temp_ev=np.isfinite(val('Temp')); temp_a=temp_ev&((val('Temp')>38)|(val('Temp')<36))
    hr_ev=np.isfinite(val('HR')); hr_a=hr_ev&(val('HR')>90)
    rr_ev=np.isfinite(val('Resp'))|np.isfinite(val('PaCO2'))
    rr_a=rr_ev&(((val('Resp')>20)&np.isfinite(val('Resp')))|((val('PaCO2')<32)&np.isfinite(val('PaCO2'))))
    w_ev=np.isfinite(val('WBC')); w_a=w_ev&((val('WBC')>12)|(val('WBC')<4))
    ne=temp_ev.astype(int)+hr_ev.astype(int)+rr_ev.astype(int)+w_ev.astype(int)
    na=temp_a.astype(int)+hr_a.astype(int)+rr_a.astype(int)+w_a.astype(int)
    E[:,19]=(ne>=2); A[:,19]=(ne>=2)&(na>=2)
    return A,E

def make_states(a,A,E):
    n=len(a); S=np.zeros((n,7),np.int8); r3=lambda j: rollsum(A[:,j]>0,3)
    ev=E[:,[0,1,2,3,9]].sum(1)>0; S[:,0]=np.where(ev,1,0)
    S[(A[:,0]+A[:,1]+A[:,3])>0,0]=2
    S[(r3(1)>=2)|(r3(2)>=2)|((A[:,0]>0)&(A[:,3]>0)),0]=3
    S[(A[:,2]>0)&(A[:,9]>0),0]=4
    ev=E[:,[4,5,6,18]].sum(1)>0; S[:,1]=np.where(ev,1,0)
    S[(A[:,4]+A[:,5])>0,1]=2
    S[(A[:,6]>0)|(r3(4)>=2)|(r3(5)>=2),1]=3
    S[((A[:,6]>0)&(A[:,5]>0))|((A[:,18]>0)&(A[:,4]>0)),1]=4
    cr=a[:,IDX['Creatinine']]; ev=E[:,[11,12,13]].sum(1)>0; S[:,2]=np.where(ev,1,0)
    S[A[:,11]>0,2]=2; S[(A[:,12]>0)|(A[:,13]>0),2]=3; S[np.isfinite(cr)&(cr>=3.5),2]=4
    ev=E[:,[7,8,19]].sum(1)>0; S[:,3]=np.where(ev,1,0)
    S[(A[:,7]+A[:,8])>0,3]=2; S[(A[:,19]>0)|(r3(7)>=2)|(r3(8)>=2),3]=3
    bt=a[:,IDX['Bilirubin_total']]; pl=a[:,IDX['Platelets']]; ev=E[:,[14,15,16,17]].sum(1)>0; S[:,4]=np.where(ev,1,0)
    S[(A[:,14]+A[:,16])>0,4]=2; S[(A[:,15]+A[:,17])>0,4]=3
    S[(np.isfinite(bt)&(bt>=6))|(np.isfinite(pl)&(pl<50)),4]=4
    lac=a[:,IDX['Lactate']]; ev=E[:,[9,10,18]].sum(1)>0; S[:,5]=np.where(ev,1,0)
    S[(A[:,9]+A[:,18])>0,5]=2; S[(A[:,10]>0)|((A[:,9]>0)&(A[:,18]>0)),5]=3; S[np.isfinite(lac)&(lac>=4),5]=4
    for j in range(6):
        prev=np.r_[0,S[:-1,j]]; rec=np.isin(prev,[3,4]) & np.isin(S[:,j],[1,2]); S[rec,j]=5
    ar=np.sum(np.isin(S[:,:6],[2,3,4]),axis=1); dy=np.sum(np.isin(S[:,:6],[3,4]),axis=1)
    S[:,6]=1; S[np.all(S[:,:6]==0,axis=1),6]=0; S[ar>=1,6]=2; S[ar>=2,6]=3; S[dy>=2,6]=4
    S[(dy>=3)|((dy>=2)&((A[:,2]>0)|(A[:,9]>0))),6]=6
    prev=np.r_[0,S[:-1,6]]; rec=np.isin(prev,[4,6])&np.isin(S[:,6],[1,2,3]); S[rec,6]=5
    D=np.ones((n,7),np.float32); T=np.zeros((n,7),np.float32); W=np.zeros(n,np.float32)
    rank=np.array([0,1,2,3,4,2,5],dtype=np.int8)
    for j in range(7):
        ch=np.r_[True,S[1:,j]!=S[:-1,j]]; starts=np.maximum.accumulate(np.where(ch,np.arange(n),0)); D[:,j]=np.arange(n)-starts+1
        tr=ch.astype(np.int8); tr[0]=0; T[:,j]=rollsum(tr,6)
        prevj=np.r_[S[0,j],S[:-1,j]]; w=(rank[S[:,j]]>rank[prevj]).astype(np.int8); w[0]=0; W+=rollsum(w,6)
    direction=np.zeros(n,np.float32); direction[W>0]=1; direction[S[:,6]==5]=-1
    return S.astype(np.float32),D,T,np.c_[ar,dy,W,direction].astype(np.float32)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--set-a',required=True,help='training_setA directory or ZIP')
    ap.add_argument('--set-b',required=True,help='training_setB directory or ZIP')
    ap.add_argument('--output-dir',required=True)
    ap.add_argument('--max-patients-a',type=int,default=None,help='QA/testing only; omit for canonical full run')
    ap.add_argument('--max-patients-b',type=int,default=None,help='QA/testing only; omit for canonical full run')
    args=ap.parse_args(); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    sources=[]
    for site,path,limit in [('A',Path(args.set_a),args.max_patients_a),('B',Path(args.set_b),args.max_patients_b)]:
        src=PSVSource(path); names=src.names[:limit] if limit else src.names; sources.append((site,src,names))
    counts=[]; total_rows=0; total_pat=0
    print('Pass 1/2: counting rows...')
    for site,src,names in sources:
        for item in names:
            n=src.count_rows(item); counts.append((site,item,n,total_pat)); total_rows+=n; total_pat+=1
    print(f'Patients={total_pat:,}; rows={total_rows:,}')
    Xb=open_memmap(out/'ALL_Xb.npy',mode='w+',dtype=np.float32,shape=(total_rows,len(BASE_NAMES)))
    Xb[:]=np.nan
    Xs=open_memmap(out/'ALL_Xs.npy',mode='w+',dtype=np.float32,shape=(total_rows,len(SYM_NAMES))); Xs[:]=0
    y=open_memmap(out/'ALL_y.npy',mode='w+',dtype=np.uint8,shape=(total_rows,)); y[:]=0
    pid=open_memmap(out/'ALL_pid.npy',mode='w+',dtype=np.int32,shape=(total_rows,)); pid[:]=0
    manifest=[]; row=0; pidx=0
    print('Pass 2/2: constructing baseline + expert symbolic arrays...')
    for site,src,names in sources:
        for item in names:
            a=src.parse(item); n=len(a); sl=slice(row,row+n)
            Xb[sl,:40]=a[:,:40].astype(np.float32); cache={c:tsince_delta(a[:,IDX[c]]) for c in KEY}; off=40
            for c in KEY: Xb[sl,off]=cache[c][0]; off+=1
            for c in KEY: Xb[sl,off]=cache[c][1]; off+=1
            for c in KEY: Xb[sl,off]=cache[c][2]; off+=1
            A,E=alerts(a); S,D,T,SUM=make_states(a,A,E)
            Xs[sl,:20]=A; Xs[sl,20:40]=E; o=40
            Xs[sl,o:o+7]=S; o+=7; Xs[sl,o:o+7]=D; o+=7; Xs[sl,o:o+7]=T; o+=7; Xs[sl,o:o+4]=SUM
            yy=a[:,IDX['SepsisLabel']].astype(np.uint8); y[sl]=yy; pid[sl]=pidx
            manifest.append((pidx,site,src.name(item),int(yy.max()),n,row,row+n))
            row+=n; pidx+=1
            if pidx%5000==0: print(f'  processed {pidx:,} patients / {row:,} rows')
    for _,src,_ in sources: src.close()
    Xb.flush(); Xs.flush(); y.flush(); pid.flush()
    meta=pd.DataFrame(manifest,columns=['patient_idx','site','file','patient_positive','n_hours','row_start','row_end'])
    meta.to_csv(out/'patient_manifest.csv',index=False)
    strata=meta['site'].astype(str)+'_'+meta['patient_positive'].astype(str)
    tr,tmp=train_test_split(meta['patient_idx'].to_numpy(),test_size=.30,random_state=42,stratify=strata)
    tmp_meta=meta.set_index('patient_idx').loc[tmp].reset_index(); strata_tmp=tmp_meta['site'].astype(str)+'_'+tmp_meta['patient_positive'].astype(str)
    va,te=train_test_split(tmp,test_size=.50,random_state=42,stratify=strata_tmp)
    np.savez(out/'split_seed42.npz',train=tr,val=va,test=te)
    meta_json={
      'n_patients':int(total_pat),'n_rows':int(total_rows),'n_features_baseline':len(BASE_NAMES),'n_features_symbolic':len(SYM_NAMES),
      'site_A_patients':int((meta.site=='A').sum()),'site_B_patients':int((meta.site=='B').sum()),
      'patient_positive_count':int(meta.patient_positive.sum()),'hourly_positive_count':int(np.asarray(y).sum()),
      'baseline_feature_names':BASE_NAMES,'symbolic_feature_names':SYM_NAMES,
      'split_seed':42,'split':'70/15/15 patient-disjoint, stratified by site and patient-positive status'
    }
    (out/'core_array_metadata.json').write_text(json.dumps(meta_json,indent=2),encoding='utf-8')
    print('Done. Canonical outputs written to',out)

if __name__=='__main__': main()

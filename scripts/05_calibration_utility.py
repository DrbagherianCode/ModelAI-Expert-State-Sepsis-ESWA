#!/usr/bin/env python3
"""Validation-only Platt calibration, Challenge utility and locked operating-point metrics."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss,confusion_matrix
from sklearn.linear_model import LogisticRegression

SEEDS=[11,23,42,73,101]
ARMS=['Baseline','CurrentStates','CompactTrajectory20','GenericTemporal27','Noise27','NonSOFAExpert9']

def logit(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
def platt_fit(p,y):
    lr=LogisticRegression(C=1e6,solver='lbfgs',max_iter=1000); lr.fit(logit(p).reshape(-1,1),y); return lr
def platt_apply(lr,p): return lr.predict_proba(logit(p).reshape(-1,1))[:,1]
def cal_intercept_slope(p,y):
    lr=LogisticRegression(C=1e6,solver='lbfgs',max_iter=1000); lr.fit(logit(p).reshape(-1,1),y); return float(lr.intercept_[0]),float(lr.coef_[0,0])
def ece_quantile(p,y,bins=10):
    q=np.quantile(p,np.linspace(0,1,bins+1)); q[0]-=1e-12; q[-1]+=1e-12; e=0.; tab=[]
    for i in range(bins):
        m=(p>q[i])&(p<=q[i+1])
        if not m.sum(): continue
        pred=float(p[m].mean()); obs=float(y[m].mean()); e+=float(m.mean())*abs(pred-obs); tab.append((i,int(m.sum()),pred,obs))
    return float(e),tab

def utility_arrays(labels,pids):
    n=len(labels); delta=np.zeros(n,float); u0=np.zeros(n,float); best=np.zeros(n,float)
    dt_early=-12; dt_opt=-6; dt_late=3.; max_tp=1.; min_fn=-2.; u_fp=-.05; u_tn=0.
    m1=max_tp/(dt_opt-dt_early); b1=-m1*dt_early; m2=-max_tp/(dt_late-dt_opt); b2=-m2*dt_late; m3=min_fn/(dt_late-dt_opt); b3=-m3*dt_opt
    starts=np.r_[0,np.where(pids[1:]!=pids[:-1])[0]+1]; ends=np.r_[starts[1:],n]
    for a,b in zip(starts,ends):
        lab=labels[a:b]; L=b-a; issep=bool(lab.any()); tsep=(int(np.argmax(lab))-dt_opt) if issep else np.inf
        for tt in range(L):
            pred1=pred0=0.
            if tt<=tsep+dt_late:
                if issep:
                    if tt<=tsep+dt_opt: pred1=max(m1*(tt-tsep)+b1,u_fp); pred0=0.
                    elif tt<=tsep+dt_late: pred1=m2*(tt-tsep)+b2; pred0=m3*(tt-tsep)+b3
                else: pred1=u_fp; pred0=u_tn
            u0[a+tt]=pred0; delta[a+tt]=pred1-pred0
            if issep and (max(0,int(tsep+dt_early))<=tt<=min(int(tsep+dt_late),L-1)): best[a+tt]=pred1
            else: best[a+tt]=pred0 if issep else 0.
    return delta,float(best.sum()-u0.sum())
def choose_threshold(p,delta):
    order=np.argsort(-p); ps=p[order]; cs=np.cumsum(delta[order]); k=int(np.argmax(cs)); return float(ps[k])
def patient_operating(labels,pids,pred):
    starts=np.r_[0,np.where(pids[1:]!=pids[:-1])[0]+1]; ends=np.r_[starts[1:],len(labels)]
    septic=detected=alert_pat=alert_septic=nonsep=nonsep_alert=nonsep_hours=false_hours=0; leads=[]
    for a,b in zip(starts,ends):
        lab=labels[a:b]; pr=pred[a:b]; issep=bool(lab.any()); anyalert=bool(pr.any())
        alert_pat+=int(anyalert)
        if issep:
            septic+=1; alert_septic+=int(anyalert); tsep=int(np.argmax(lab))+6; lo=max(0,tsep-12); hi=min(len(lab)-1,tsep+3); ids=np.where(pr[lo:hi+1])[0]
            if len(ids): detected+=1; leads.append(tsep-(lo+int(ids[0])))
        else:
            nonsep+=1; nonsep_hours+=len(lab); false_hours+=int(pr.sum()); nonsep_alert+=int(anyalert)
    return {'event_sensitivity_beneficial_window':detected/septic if septic else np.nan,'patient_any_alert_PPV':alert_septic/alert_pat if alert_pat else np.nan,
            'nonseptic_patient_specificity_no_alert':1-nonsep_alert/nonsep if nonsep else np.nan,
            'false_alerts_per_100_nonseptic_patient_hours':100*false_hours/nonsep_hours if nonsep_hours else np.nan,
            'beneficial_window_lead_median_h':float(np.median(leads)) if leads else np.nan}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--array-dir',required=True); ap.add_argument('--prediction-dir',required=True); ap.add_argument('--output-dir',required=True); args=ap.parse_args()
    arr=Path(args.array_dir); prd=Path(args.prediction_dir); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    y=np.load(arr/'ALL_y.npy',mmap_mode='r'); pid=np.load(arr/'ALL_pid.npy',mmap_mode='r'); s=np.load(arr/'split_seed42.npz'); va=s['val']; te=s['test']
    iva=np.where(np.isin(pid,va))[0]; ite=np.where(np.isin(pid,te))[0]; yv=np.asarray(y[iva]); yt=np.asarray(y[ite]); pidv=np.asarray(pid[iva]); pidt=np.asarray(pid[ite])
    dv,denv=utility_arrays(yv,pidv); dt,dent=utility_arrays(yt,pidt)
    rows=[]; calbins={}
    for arm in ARMS:
        pv=np.mean(np.vstack([np.load(prd/f'rev_{arm}_{sd}_val.npy') for sd in SEEDS]),axis=0).astype(float)
        pt=np.mean(np.vstack([np.load(prd/f'rev_{arm}_{sd}_test.npy') for sd in SEEDS]),axis=0).astype(float)
        lr=platt_fit(pv,yv); pcv=platt_apply(lr,pv); pct=platt_apply(lr,pt); ci,cs=cal_intercept_slope(pct,yt); ece,bins=ece_quantile(pct,yt,10); calbins[arm]=bins
        thr=choose_threshold(pcv,dv); bv=pcv>=thr; bt=pct>=thr; tn,fp,fn,tp=confusion_matrix(yt,bt,labels=[0,1]).ravel()
        row={'arm':arm,'ensemble_AUPRC':average_precision_score(yt,pt),'ensemble_AUROC':roc_auc_score(yt,pt),'raw_Brier':brier_score_loss(yt,pt),
             'calibrated_Brier':brier_score_loss(yt,pct),'calibration_intercept_test':ci,'calibration_slope_test':cs,'ECE10_test':ece,
             'platt_intercept_from_val':float(lr.intercept_[0]),'platt_slope_from_val':float(lr.coef_[0,0]),'locked_threshold_calibrated':thr,
             'validation_utility':float(dv[bv].sum()/denv),'test_utility':float(dt[bt].sum()/dent),'hourly_sensitivity':tp/(tp+fn),'hourly_specificity':tn/(tn+fp),
             'hourly_PPV':tp/(tp+fp) if tp+fp else np.nan,'hourly_NPV':tn/(tn+fn)}; row.update(patient_operating(yt,pidt,bt)); rows.append(row); print(arm,row)
    pd.DataFrame(rows).to_csv(out/'ModelAI_Revision_Calibration_Utility_OperatingMetrics.csv',index=False)
    (out/'ModelAI_Revision_CalibrationBins.json').write_text(json.dumps(calbins,indent=2),encoding='utf-8')
if __name__=='__main__': main()

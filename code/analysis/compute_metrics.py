"""Compute class presence, counting error, and report-level descriptive statistics."""
from pathlib import Path
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[2]
O=R/'results/recomputed';O.mkdir(parents=True,exist_ok=True)
def ratio(a,b):return a/b if b else np.nan

def main():
    d=pd.read_csv(R/'results/evaluation/perception/perception_detail.csv')
    rows=[]
    for g,gd in d.groupby('group'):
        cr=[]
        for c,cd in gd.groupby('class'):
            y=cd['true_count'].to_numpy()>0;p=cd['predicted_count'].to_numpy()>0
            tp=int((y&p).sum());fp=int((~y&p).sum());fn=int((y&~p).sum());tn=int((~y&~p).sum())
            r={'group':g,'class':c,'n':len(cd),'TP':tp,'FP':fp,'FN':fn,'TN':tn,'Precision':ratio(tp,tp+fp),'Recall':ratio(tp,tp+fn),'F1':ratio(2*tp,2*tp+fp+fn),'MAE':(cd['true_count']-cd['predicted_count']).abs().mean()}
            rows.append(r);cr.append(r)
        rows.append({'group':g,'class':'Macro','n':len(gd),**{k:np.mean([x[k] for x in cr]) for k in ['Precision','Recall','F1','MAE']}})
    pd.DataFrame(rows).to_csv(O/'perception_metrics.csv',index=False)
    raw=pd.read_csv(R/'results/evaluation/report_quality/ratings_long.csv');rep=raw.drop_duplicates('report_id').copy()
    for metric,col in [('basis_correctness_rate','correct_basis_count'),('scenario_matching_rate','matched_basis_count')]:rep[metric]=rep[col].div(rep['basis_count'].where(rep['basis_count']>0))
    rows=[]
    for g,gd in rep.groupby('group'):
        for metric in ['basis_correctness_rate','scenario_matching_rate','perception_seconds','alignment_seconds','retrieval_seconds','generation_seconds','total_seconds','total_tokens','api_cost_cny']:
            v=gd[metric].dropna();rows.append({'group':g,'metric':metric,'n':len(v),'mean':v.mean(),'sd':v.std(),'median':v.median(),'q1':v.quantile(.25),'q3':v.quantile(.75)})
        for metric in ['reliability','logical_coherence']:
            v=raw.loc[raw['group']==g,metric].dropna();rows.append({'group':g,'metric':metric,'n':len(v),'mean':v.mean(),'sd':v.std(),'median':v.median(),'q1':v.quantile(.25),'q3':v.quantile(.75)})
    pd.DataFrame(rows).to_csv(O/'descriptive_statistics.csv',index=False)
    print('Saved offline recomputations to',O)
if __name__=='__main__':main()

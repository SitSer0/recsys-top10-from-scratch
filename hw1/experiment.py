"""Validation-only selection, then one final test pass per frozen configuration."""
import gc, json, time
from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd
from models import ItemKNN, IALS, EASE, FM, RatingRecencyALS
from data_utils import TrainingData
from evaluation import evaluate, paired_bootstrap, recommend

CLASSES={'ItemKNN':ItemKNN,'iALS':IALS,'EASE':EASE,'FM':FM,'CustomALS':RatingRecencyALS}
GRIDS={
 'ItemKNN':[dict(neighbors=k,shrink=s) for k,s in [(30,10.),(100,30.)]],
 'iALS':[dict(factors=k,reg=r,alpha=20.,iterations=8,seed=42) for k,r in product([16,32],[1.,10.])],
 'EASE':[dict(reg=r) for r in [100.,500.]],
 'FM':[dict(factors=k,lr=.01,reg=r,epochs=8,negatives=2,metadata=True,seed=42) for k,r in product([16,32],[1e-5,1e-4])],
}

def tune(train,validation,shape,t1,output='results'):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    data=TrainingData(train,shape,t1)
    rows=[];best={}
    for name in ['ItemKNN','iALS','EASE','FM','CustomALS']:
        configs=GRIDS.get(name)
        if name=='CustomALS':
            # Controlled comparison: exactly the chosen baseline optimizer and confidence.
            configs=[dict(**best['iALS'],half_life=h) for h in [365.,1460.]]
        best_score=-np.inf
        for params in configs:
            start=time.perf_counter()
            model=CLASSES[name](**params).fit(data)
            fit_seconds=time.perf_counter()-start
            metrics,_=evaluate(model,validation,data.seen)
            row=dict(model=name,params=json.dumps(params,sort_keys=True),fit_seconds=fit_seconds,**metrics)
            rows.append(row)
            print(name,params,metrics, 'fit seconds',round(fit_seconds,2),flush=True)
            if metrics['NDCG@10']>best_score:
                best_score=metrics['NDCG@10'];best[name]=params
            del model;gc.collect()
            pd.DataFrame(rows).to_csv(output/'validation.csv',index=False)
    (output/'selected.json').write_text(json.dumps(best,indent=2,sort_keys=True))
    return pd.DataFrame(rows),best

def final_evaluation(train,validation,test,shape,t2,best,output='results'):
    output=Path(output)
    history=pd.concat([train,validation],ignore_index=True)
    data=TrainingData(history,shape,t2)
    rows=[];details={};examples=[]
    configurations=[(name,CLASSES[name],params) for name,params in best.items()]
    configurations.append(('FM_ID_only',FM,dict(best['FM'],metadata=False)))
    for name,cls,params in configurations:
        start=time.perf_counter();model=cls(**params).fit(data)
        elapsed=time.perf_counter()-start
        metrics,per_user=evaluate(model,test,data.seen)
        details[name]=per_user
        per_user.to_csv(output/f'per_user_{name}.csv',index=False)
        rows.append(dict(model=name,params=json.dumps(params,sort_keys=True),fit_seconds=elapsed,**metrics))
        # Same first three eligible users for every model. IDs only, no review text.
        users=sorted(per_user.u)[:3]
        for u,items in zip(users,recommend(model,users,data.seen)):
            for rank,item in enumerate(items,1):
                examples.append(dict(model=name,u=int(u),rank=rank,i=int(item)))
        print('TEST',name,metrics,flush=True)
        del model;gc.collect()
    results=pd.DataFrame(rows)
    results.to_csv(output/'test.csv',index=False)
    pd.DataFrame(examples).to_csv(output/'recommendations.csv',index=False)
    intervals={
      'CustomALS minus iALS':paired_bootstrap(details['CustomALS'],details['iALS']),
      'FM minus FM_ID_only':paired_bootstrap(details['FM'],details['FM_ID_only'])}
    (output/'paired_intervals.json').write_text(json.dumps(intervals,indent=2))
    return results,intervals

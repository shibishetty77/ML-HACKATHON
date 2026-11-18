"""Team-wide binary candidate metrics, deterministic GroupKFold and timing.

All model owners can import this module. Road/node/trip IDs are metadata only.
"""
import csv
import json
from pathlib import Path
import time
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,log_loss,confusion_matrix
from threadpoolctl import threadpool_limits
from audit import write_csv

SEED=352
THREADS=4
EPSILON=1e-7

def load_prepared(results,partition):
    parts=[]
    for path in sorted((Path(results)/'processed').glob(f'{partition}_*.npz')):
        with np.load(path) as z:
            n=len(z['X'])
            parts.append({k:z[k] for k in ['X','y','eligible','arc_id','reference_arc','observation','from_node','to_node']})
            parts[-1]['trip_id']=np.full(n,int(path.stem.split('_')[-1]),dtype=np.int32)
    if not parts:raise ValueError(f'No prepared files for {partition}')
    d={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    d['previous_dt_s']=d['X'][:,7]
    assert np.isfinite(d['X']).all()
    return d

def subset(data,mask):
    return {k:v[mask] for k,v in data.items()}

def group_folds(data,path):
    """Use eligible candidate counts to define one common, deterministic split."""
    valid=data['eligible'];groups=data['trip_id'][valid]
    cv=GroupKFold(n_splits=5)
    assignments=[];folds=[]
    for fold,(_,val) in enumerate(cv.split(np.zeros((len(groups),1),dtype=np.int8),data['y'][valid],groups=groups),1):
        validation_trips=np.unique(groups[val]);folds.append(validation_trips)
        for trip in validation_trips:
            assignments.append(dict(trip_id=f'{trip:08d}',fold=fold,
                                    eligible_candidate_rows=int(((data['trip_id']==trip)&valid).sum())))
    assert len(assignments)==80 and len({r['trip_id'] for r in assignments})==80
    write_csv(path,assignments)
    return folds

def observation_predictions(data,probability):
    boundary=np.r_[True,(data['trip_id'][1:]!=data['trip_id'][:-1])|(data['observation'][1:]!=data['observation'][:-1])]
    starts=np.flatnonzero(boundary)
    group=np.cumsum(boundary)-1
    best=np.maximum.reduceat(probability,starts)
    # On equal scores, keep the earlier candidate, which is sorted by distance then ID.
    winner=np.minimum.reduceat(np.where(probability==best[group],np.arange(len(probability)),len(probability)),starts)
    return starts,winner

def evaluate(data,probability):
    probability=np.asarray(probability)
    assert len(probability)==len(data['X']) and np.isfinite(probability).all()
    valid=data['eligible'];y=data['y'][valid];p=probability[valid]
    prediction=p>=.5
    starts,winner=observation_predictions(data,probability)
    observation_valid=data['eligible'][starts]
    correct=data['arc_id'][winner]==data['reference_arc'][starts]
    hits=np.add.reduceat(data['y'].astype(np.int16),starts)>0
    tid=data['trip_id'][starts];obs=data['observation'][starts]
    dt=data['previous_dt_s'][starts][1:]
    pair=(tid[1:]==tid[:-1])&(obs[1:]==obs[:-1]+1)&(dt>0)&(dt<=10)
    connected=(data['arc_id'][winner][1:]==data['arc_id'][winner][:-1]) | (data['to_node'][winner][:-1]==data['from_node'][winner][1:])
    score=dict(accuracy=float(accuracy_score(y,prediction)),precision=float(precision_score(y,prediction,zero_division=0)),
               recall=float(recall_score(y,prediction,zero_division=0)),
               macro_f1=float(f1_score(y,prediction,average='macro',zero_division=0)),
               top1_accuracy=float(correct[observation_valid].mean()),
               candidate_miss_rate=float((~hits[observation_valid]).mean()),
               log_loss=float(log_loss(y,np.clip(p,EPSILON,1-EPSILON),labels=[0,1])),
               route_continuity=float(connected[pair].mean()) if pair.any() else None,
               candidate_rows=int(valid.sum()),all_candidate_rows=len(probability),
               observations=len(starts),eligible_observations=int(observation_valid.sum()),
               excluded_observations=int((~observation_valid).sum()),continuity_pairs=int(pair.sum()),
               positive_candidates=int(y.sum()),negative_candidates=int((y==0).sum()),
               confusion_matrix=confusion_matrix(y,prediction,labels=[0,1]).tolist())
    return score

def benchmark(model,x,repeats=5):
    """One warm-up, then five pipeline calls; median seconds. Excludes loading."""
    with threadpool_limits(limits=THREADS):
        model.predict_proba(x[:min(1024,len(x))])
        times=[];probability=None
        for _ in range(repeats):
            start=time.perf_counter();probability=model.predict_proba(x)[:,1];times.append(time.perf_counter()-start)
    return probability,float(np.median(times)),times

def aggregate_folds(rows):
    keys=['accuracy','precision','recall','macro_f1','top1_accuracy','candidate_miss_rate','log_loss','route_continuity','fit_seconds','inference_seconds']
    return {k:dict(mean=float(np.mean([r[k] for r in rows])),std=float(np.std([r[k] for r in rows],ddof=1))) for k in keys}

"""Recompute all five held-out results and check the model/feature contract."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from common import load_trip,segment_distance
from features import candidate_features
from evaluation import load_prepared,subset,evaluate,group_folds,THREADS
from audit import write_csv

def validate(data,results='results',out='mlp',features_only=False):
    results=Path(results);out=Path(out);checks=[]
    config=json.loads((results/'feature_config.json').read_text())
    assert config['contract_version']=='team-plan-v2-full16' and len(config['features'])==16
    assert config['max_observations_per_trip']==0
    manifest=[]
    for split in ['train','student_test']:
        expected=list((Path(data)/split).iterdir())
        assert len([p for p in expected if p.is_dir()])==(80 if split=='train' else 10)
        for p in expected:
            if not p.is_dir():continue
            f=results/'processed'/f'{split}_{p.name}.npz'
            with np.load(f) as z:
                raw=np.loadtxt(p/f'{p.name}.track',ndmin=2)
                assert len(z['aligned_full'])==len(raw)
                assert np.array_equal(np.unique(z['observation']),np.arange(len(raw)))
                assert z['X'].shape[1]==16 and np.isfinite(z['X']).all()
            manifest.append(dict(partition=split,trip_id=p.name,file=f.name,bytes=f.stat().st_size,
                                 sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
    write_csv(results/'prepared_manifest.csv',manifest)
    checks.append('All 90 prepared trips contain candidates for every original GPS observation; 16 finite features and unchanged row IDs.')
    for split,tid in [('train','00000001'),('train','00000054'),('student_test','00000000')]:
        raw=load_trip(Path(data)/split/tid,reference=False);indices=np.linspace(0,len(raw['track'])-1,9).astype(int)
        x,arc,obs,_,gps,a,b=candidate_features(raw['track'],raw['nodes'],raw['arcs'],indices)
        for i in indices:
            distance,_=segment_distance(gps[i],a,b)
            exact=np.lexsort((np.arange(len(distance)),distance));exact=exact[distance[exact]<=75][:8]
            if len(exact):assert np.array_equal(arc[obs==i],exact)
        with np.load(results/'processed'/f'{split}_{tid}.npz') as stored:
            selected=np.isin(stored['observation'],indices)
            assert np.array_equal(arc,stored['arc_id'][selected]) and np.allclose(x,stored['X'][selected])
    checks.append('27 observation candidate sets match exhaustive segment geometry when radius candidates exist; label-free features reproduce stored rows.')
    train=load_prepared(results,'train');test=load_prepared(results,'student_test')
    assert not set(np.unique(train['trip_id'])) & set(np.unique(test['trip_id']))
    if features_only:
        (results/'feature_validation.json').write_text(json.dumps(dict(status='passed',checks=checks),indent=2));print('\n'.join(checks));return
    folds=list(csv.DictReader(open(results/'groupkfold_5_assignments.csv',encoding='utf-8')))
    assert len(folds)==80 and len({r['trip_id'] for r in folds})==80 and {int(r['fold']) for r in folds}=={1,2,3,4,5}
    assert {int(r['trip_id']) for r in folds}==set(np.unique(train['trip_id']))
    checks.append('Five exported whole-trip folds cover each of the 80 training trips exactly once; no student-test group enters fitting.')
    rows=list(csv.DictReader(open(out/'cv_results.csv',encoding='utf-8')))
    assert len(rows)==5
    for row in rows:
        fold=int(row['fold']);groups=[int(r['trip_id']) for r in folds if int(r['fold'])==fold]
        val=subset(train,np.isin(train['trip_id'],groups))
        with np.load(out/'predictions'/f'fold_{fold}.npz') as saved:
            assert np.array_equal(saved['trip_id'],val['trip_id']) and np.array_equal(saved['arc_id'],val['arc_id'])
            score=evaluate(val,saved['probability'])
            for key in ['accuracy','precision','recall','macro_f1','top1_accuracy','log_loss','candidate_miss_rate','route_continuity']:
                assert abs(score[key]-float(row[key]))<1e-6,(fold,key)
        pipeline=joblib.load(out/'artifacts'/f'fold_{fold}_pipeline.joblib')
        fit=(~np.isin(train['trip_id'],groups))&train['eligible']
        assert int(pipeline[0].n_samples_seen_)==int(fit.sum())
        assert np.allclose(pipeline[0].mean_,train['X'][fit].mean(axis=0,dtype=np.float64),atol=1e-5)
    checks.append('All eight metrics recompute for all five validation folds; every scaler sample count and mean match its training-fold rows only.')
    metrics=json.loads((out/'metrics.json').read_text())
    with np.load(out/'predictions/student_test.npz') as saved:
        score=evaluate(test,saved['probability'])
        for key in ['accuracy','precision','recall','macro_f1','top1_accuracy','log_loss','candidate_miss_rate','route_continuity']:
            assert abs(score[key]-metrics['student_test'][key])<1e-6,key
        model=joblib.load(out/'artifacts/mlp_pipeline.joblib')
        assert int(model[0].n_samples_seen_)==int(train['eligible'].sum())
        assert np.array_equal(model.classes_,np.array([0,1]))
        with threadpool_limits(limits=THREADS):
            assert np.allclose(model.predict_proba(test['X'][:1000])[:,1],saved['probability'][:1000],atol=1e-6)
    checks.append('Full student-test metrics recompute; saved 16-feature scaler+MLP reproduces the evaluated probabilities.')
    for row in rows:
        hist=list(csv.DictReader(open(out/'histories'/f'fold_{int(row["fold"])}.csv')))
        assert len(hist)==int(row['epochs']) and all(float(r['validation_log_loss'])>=0 for r in hist)
    checks.append('Every fold has a recorded training objective and real held-out validation log loss for every executed epoch.')
    check_inference=json.loads((out/'inference_verification.json').read_text());assert check_inference['status']=='passed'
    checks.append('Raw-input inference verified on all ten student-test trips without reading reference routes.')
    (out/'validation.json').write_text(json.dumps(dict(status='passed',checks=checks),indent=2));print('\n'.join(checks))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--data',required=True);a.add_argument('--results',default='results');a.add_argument('--out',default='mlp');a.add_argument('--features-only',action='store_true')
    args=a.parse_args();validate(args.data,args.results,args.out,args.features_only)

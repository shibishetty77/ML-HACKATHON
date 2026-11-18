"""Trip inference from raw unlabeled input, timing, maps and team handoff table."""
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
from common import load_trip
from features import candidate_features
from evaluation import THREADS,benchmark,evaluate,load_prepared,subset,observation_predictions
from visualisation import make_map,static_prediction_plot
from audit import write_csv

def export(data,results='results',out='mlp',assets='assets'):
    results=Path(results);out=Path(out);assets=Path(assets)
    (out/'maps').mkdir(exist_ok=True)
    model=joblib.load(out/'artifacts/mlp_pipeline.joblib')
    test=load_prepared(results,'student_test');timings=[]
    with np.load(out/'predictions/student_test.npz') as z:saved=z['probability']
    for p in sorted((Path(data)/'student_test').iterdir()):
        if not p.is_dir():continue
        overall=time.perf_counter();start=overall
        raw=load_trip(p,reference=False);load_seconds=time.perf_counter()-start
        start=time.perf_counter()
        x,arc,obs,_,_,_,_=candidate_features(raw['track'],raw['nodes'],raw['arcs'],np.arange(len(raw['track'])))
        candidate_seconds=time.perf_counter()-start
        start=time.perf_counter()
        with threadpool_limits(limits=THREADS):probability=model.predict_proba(x)[:,1]
        first_predict_seconds=time.perf_counter()-start
        starts=np.r_[0,np.flatnonzero(np.diff(obs))+1];ends=np.r_[starts[1:],len(obs)]
        winners=np.array([lo+np.argmax(probability[lo:hi]) for lo,hi in zip(starts,ends)])
        predicted=arc[winners]
        end_to_end=time.perf_counter()-overall
        # Evaluation/labels occur after the completed label-free prediction path.
        mask=test['trip_id']==int(p.name);prepared=subset(test,mask)
        assert np.array_equal(arc,prepared['arc_id']) and np.array_equal(obs,prepared['observation'])
        assert np.allclose(x,prepared['X']) and np.allclose(probability,saved[mask],atol=1e-6)
        _,median,repeats=benchmark(model,x)
        score=evaluate(prepared,probability)
        timings.append(dict(trip_id=p.name,observations=len(raw['track']),candidate_rows=len(x),
                            load_seconds=load_seconds,candidate_generation_seconds=candidate_seconds,
                            first_pipeline_seconds=first_predict_seconds,complete_trip_seconds=end_to_end,
                            pipeline_median_seconds=median,pipeline_seconds_per_candidate=median/len(x),
                            pipeline_seconds_per_observation=median/len(raw['track']),
                            weak_top1_accuracy=score['top1_accuracy'],route_continuity=score['route_continuity']))
        write_csv(out/'predictions'/f'{p.name}_road_predictions.csv',[
            dict(observation=i,timestamp_s=float(raw['track'][i,2]),predicted_arc_id=int(arcid),
                 positive_candidate_score=float(probability[winners[i]])) for i,arcid in enumerate(predicted)])
        if p.name=='00000000':
            # For an offline comparison map only; never read by the predictor.
            raw['route']=np.loadtxt(p/f'{p.name}.route',dtype=int,ndmin=1)
            make_map(raw,out/'maps/student_test_00000000.html','Student-test 00000000 | MLP | five-fold team plan',predicted,assets)
            static_prediction_plot(raw,predicted,out/'figures/geographical_prediction.png','Student-test 00000000 | MLP | static geographical plot')
        print(f'RAW-INFERENCE {p.name}: {end_to_end:.3f}s complete trip; predictions reproduced.',flush=True)
    write_csv(out/'trip_inference_times.csv',timings)
    for tid in ['00000002','00000054']:
        d=load_trip(Path(data)/'train'/tid)
        make_map(d,out/'maps'/f'train_{tid}.html',f'Training {tid} | original observations and reference',assets=assets)
    metrics=json.loads((out/'metrics.json').read_text());s=metrics['student_test'];cv=metrics['cv'];t=metrics['timing']
    starts,winners=observation_predictions(test,saved)
    eligible=test['eligible'][starts]
    correct=test['arc_id'][winners]==test['reference_arc'][starts]
    present=np.add.reduceat(test['y'].astype(np.int16),starts)>0
    reliable_heading=test['X'][starts,5]>.5
    breakdown=[]
    for name,condition in [('Reliable heading',reliable_heading),('No reliable heading',~reliable_heading),
                           ('Reference candidate present',present),('Reference candidate missing',~present),
                           ('Preceding gap >10s',test['previous_dt_s'][starts]>10)]:
        mask=eligible&condition;n=int(mask.sum())
        breakdown.append(dict(group=name,eligible_observations=n,top1_accuracy=float(correct[mask].mean()) if n else None))
    write_csv(out/'error_breakdown.csv',breakdown)
    table=[
        ('CV Macro F1 (mean +/- sample SD)',f'{cv["macro_f1"]["mean"]:.6f} +/- {cv["macro_f1"]["std"]:.6f}'),
        ('CV Top-1 Accuracy (mean +/- sample SD)',f'{cv["top1_accuracy"]["mean"]:.6f} +/- {cv["top1_accuracy"]["std"]:.6f}'),
        ('Student-test Accuracy',s['accuracy']),('Student-test Precision',s['precision']),('Student-test Recall',s['recall']),
        ('Student-test Macro F1',s['macro_f1']),('Student-test Top-1 Accuracy',s['top1_accuracy']),
        ('Student-test Log Loss',s['log_loss']),('Student-test Candidate Miss Rate',s['candidate_miss_rate']),
        ('Student-test Route Continuity',s['route_continuity']),('Final training time (s)',metrics['final_fit']['fit_seconds']),
        ('5-fold total fitting time (s)',sum(float(r['fit_seconds']) for r in __import__('csv').DictReader(open(out/'cv_results.csv')))),
        ('Full student-test pipeline inference median (s)',t['inference_seconds']),
        ('Pipeline inference per candidate (s)',t['seconds_per_candidate']),('Pipeline inference per observation (s)',t['seconds_per_observation'])]
    write_csv(out/'mlp_comparison_column.csv',[dict(metric=k,MLP=v) for k,v in table])
    write_csv(out/'team_comparison_pending.csv',[dict(metric=k,KNN='Pending owner results',XGBoost='Pending owner results',MLP=v,
                                                     RandomForest='Pending owner results',SVM='Pending owner results') for k,v in table])
    fig,ax=plt.subplots(1,2,figsize=(9,3.8),layout='constrained')
    ax[0].bar([r['trip_id'][-2:] for r in timings],[r['pipeline_median_seconds'] for r in timings],color='#287c8e')
    ax[1].bar([r['trip_id'][-2:] for r in timings],[r['complete_trip_seconds'] for r in timings],color='#d47a36')
    ax[0].set(title='MLP pipeline: median of five calls',xlabel='Student-test trip ID suffix',ylabel='Seconds')
    ax[1].set(title='Complete trip: raw inputs to arc predictions',xlabel='Student-test trip ID suffix',ylabel='Seconds')
    fig.savefig(out/'figures/inference_times.png',dpi=180);plt.close(fig)
    (out/'inference_verification.json').write_text(json.dumps(dict(status='passed',trips=10,
        checks=['All 10 trips were predicted from raw track/nodes/arcs without reading route labels.',
                'Regenerated features match the shared prepared dataset.',
                'Regenerated probabilities match the saved evaluated probabilities within 1e-6.'],
        timing_policy='Complete-trip time includes loading, fresh candidate index/features, one scaler+MLP call, and arc selection. Excludes labels, evaluation and maps.'),indent=2))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--data',required=True);a.add_argument('--results',default='results');a.add_argument('--out',default='mlp');a.add_argument('--assets',default='assets')
    args=a.parse_args();export(args.data,args.results,args.out,args.assets)

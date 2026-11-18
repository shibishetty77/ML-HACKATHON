"""Full-observation MLP experiment implementing the user's evaluation plan."""
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits
from common import trip_directories
from audit import write_csv
from evaluation import SEED,THREADS,EPSILON,load_prepared,subset,group_folds,evaluate,benchmark,aggregate_folds

def fit_network(x,y,tag,out,max_epochs=200,validation=None):
    start=time.perf_counter()
    scaler=StandardScaler();scaled=scaler.fit_transform(x)
    validation_x=scaler.transform(validation[0]) if validation else None
    net=MLPClassifier(hidden_layer_sizes=(32,16),activation='relu',solver='adam',alpha=.001,
                      batch_size=2048,learning_rate_init=.001,max_iter=max_epochs,
                      early_stopping=False,random_state=SEED,shuffle=True)
    history=[];best=np.inf;bad=0;status='epoch_cap_reached'
    with threadpool_limits(limits=THREADS):
        for epoch in range(1,max_epochs+1):
            if epoch==1:net.partial_fit(scaled,y,classes=np.array([0,1]))
            else:net.partial_fit(scaled,y)
            objective=float(net.loss_)
            row=dict(epoch=epoch,training_objective=objective)
            if validation:
                pv=net.predict_proba(validation_x)[:,1]
                row['validation_log_loss']=float(log_loss(validation[1],np.clip(pv,EPSILON,1-EPSILON),labels=[0,1]))
            history.append(row)
            if objective < best-.0001:bad=0
            else:bad+=1
            best=min(best,objective)
            if epoch==1 or epoch%5==0:
                print(f'{tag} epoch {epoch}: training objective {objective:.6f}'+(f', validation log loss {row["validation_log_loss"]:.6f}' if validation else ''),flush=True)
            if bad>=15:
                status='training_objective_plateau';break
    seconds=time.perf_counter()-start
    pipeline=make_pipeline(scaler,net)
    write_csv(out/'histories'/f'{tag}.csv',history)
    joblib.dump(pipeline,out/'artifacts'/f'{tag}_pipeline.joblib',compress=3)
    print(f'{tag} completed: {len(history)} epochs, {status}, {seconds:.2f}s',flush=True)
    return pipeline,dict(epochs=len(history),status=status,fit_seconds=seconds),history

def plot_results(out,folds,history,test,cv):
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    ax.bar([r['fold'] for r in folds],[r['macro_f1'] for r in folds],color='#287c8e')
    ax.axhline(cv['macro_f1']['mean'],color='#d47a36',linestyle='--',label='Five-fold mean')
    ax.set(xlabel='GroupKFold validation fold',ylabel='Candidate binary Macro F1',ylim=(0,1),xticks=[1,2,3,4,5],title='MLP: actual five-fold validation results')
    ax.legend();fig.savefig(out/'figures/cv_macro_f1.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(10,4.3),layout='constrained')
    for i in range(1,6):
        import csv
        rows=list(csv.DictReader(open(out/'histories'/f'fold_{i}.csv',encoding='utf-8')))
        ax[0].plot([int(r['epoch']) for r in rows],[float(r['training_objective']) for r in rows],label=f'Fold {i}')
        ax[1].plot([int(r['epoch']) for r in rows],[float(r['validation_log_loss']) for r in rows],label=f'Fold {i}')
    ax[0].set(title='Actual regularised training objective',xlabel='Epoch',ylabel='Objective')
    ax[1].set(title='Actual held-out-trip validation log loss',xlabel='Epoch',ylabel='Binary log loss')
    ax[1].legend(fontsize=8);fig.savefig(out/'figures/fold_learning_curves.png',dpi=180);plt.close(fig)
    cm=np.array(test['confusion_matrix']);fig,ax=plt.subplots(figsize=(4.5,4),layout='constrained');ax.imshow(cm,cmap='Blues')
    for (i,j),v in np.ndenumerate(cm):ax.text(j,i,f'{v:,}',ha='center',va='center',color='white' if v>cm.max()/2 else 'black')
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Non-match','Match'],yticklabels=['Non-match','Match'],xlabel='Predicted candidate class',ylabel='Weak reference class',title='MLP: all student-test observations')
    fig.savefig(out/'figures/confusion_matrix.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,3.5),layout='constrained')
    ax.plot([r['epoch'] for r in history],[r['training_objective'] for r in history],color='#287c8e')
    ax.set(title='Final refit on 80 trips: actual training history',xlabel='Epoch',ylabel='Regularised training objective')
    fig.savefig(out/'figures/final_training_curve.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(8,3.7),layout='constrained')
    ax[0].bar(['Macro F1','Top-1','Continuity'],[test['macro_f1'],test['top1_accuracy'],test['route_continuity']],color=['#287c8e','#d47a36','#5c9d72']);ax[0].set(ylim=(0,1),title='MLP student-test metrics')
    ax[1].bar(['MLP'],[test['log_loss']],color='#287c8e');ax[1].set(title='Candidate probability log loss',ylabel='Lower is better')
    fig.savefig(out/'figures/student_test_metrics.png',dpi=180);plt.close(fig)

def run(results='results',out='mlp',max_epochs=200):
    results=Path(results);out=Path(out)
    for d in ['artifacts','histories','figures','predictions']:(out/d).mkdir(parents=True,exist_ok=True)
    config=json.loads((results/'feature_config.json').read_text())
    assert config['max_observations_per_trip']==0,'The team plan requires every observation, not the earlier sampled pilot.'
    train=load_prepared(results,'train');test=load_prepared(results,'student_test')
    assert len(np.unique(train['trip_id']))==80 and len(np.unique(test['trip_id']))==10
    assert not set(np.unique(train['trip_id'])) & set(np.unique(test['trip_id']))
    balance=dict(training_positive=int(train['y'][train['eligible']].sum()),training_candidate_rows=int(train['eligible'].sum()),
                 student_test_positive=int(test['y'][test['eligible']].sum()),student_test_candidate_rows=int(test['eligible'].sum()),
                 policy='No class weighting, no under/oversampling. Natural eligible candidate distribution.')
    (out/'class_balance.json').write_text(json.dumps(balance,indent=2))
    folds=group_folds(train,results/'groupkfold_5_assignments.csv')
    rows=[]
    for fold,validation_trips in enumerate(folds,1):
        valmask=np.isin(train['trip_id'],validation_trips);fitmask=(~valmask)&train['eligible']
        val=subset(train,valmask);v=val['eligible']
        model,fit,_=fit_network(train['X'][fitmask],train['y'][fitmask],f'fold_{fold}',out,max_epochs,
                                validation=(val['X'][v],val['y'][v]))
        probability,seconds,repeats=benchmark(model,val['X'])
        score=evaluate(val,probability)
        row=dict(fold=fold,training_trips=len(np.unique(train['trip_id'][fitmask])),validation_trips=len(validation_trips),
                 **fit,inference_seconds=seconds,**{k:v for k,v in score.items() if k!='confusion_matrix'})
        rows.append(row);write_csv(out/'cv_results.csv',rows)
        np.savez_compressed(out/'predictions'/f'fold_{fold}.npz',probability=probability,trip_id=val['trip_id'],observation=val['observation'],arc_id=val['arc_id'])
        (out/'predictions'/f'fold_{fold}_confusion.json').write_text(json.dumps(score['confusion_matrix']))
        print(f'FOLD {fold} SCORE: F1={score["macro_f1"]:.6f}; Top1={score["top1_accuracy"]:.6f}; continuity={score["route_continuity"]:.6f}',flush=True)
    cv=aggregate_folds(rows)
    eligible=train['eligible']
    model,fit,history=fit_network(train['X'][eligible],train['y'][eligible],'final',out,max_epochs)
    p,inference,repeats=benchmark(model,test['X']);test_score=evaluate(test,p)
    with threadpool_limits(limits=THREADS):train_probability=model.predict_proba(train['X'])[:,1]
    training_score=evaluate(train,train_probability)
    joblib.dump(model,out/'artifacts/mlp_pipeline.joblib',compress=3)
    joblib.dump(model[0],out/'artifacts/scaler.joblib',compress=3)
    joblib.dump(model[-1],out/'artifacts/mlp_classifier.joblib',compress=3)
    np.savez_compressed(out/'predictions/student_test.npz',probability=p,trip_id=test['trip_id'],observation=test['observation'],arc_id=test['arc_id'])
    pertrip=[]
    for trip in np.unique(test['trip_id']):
        mask=test['trip_id']==trip;d=subset(test,mask);_,seconds,_=benchmark(model,d['X']);s=evaluate(d,p[mask])
        pertrip.append(dict(trip_id=f'{trip:08d}',classifier_seconds=seconds,seconds_per_candidate=seconds/len(d['X']),
                            seconds_per_observation=seconds/s['observations'],**{k:v for k,v in s.items() if k!='confusion_matrix'}))
    write_csv(out/'student_test_by_trip.csv',pertrip)
    result=dict(model='MLPClassifier',owner='Member 3',contract_version=config['contract_version'],feature_count=len(config['features']),
                architecture=[len(config['features']),32,16,1],seed=SEED,folds=5,
                cv_strategy='sklearn GroupKFold(n_splits=5, shuffle=False), groups=trip_id, balanced by eligible candidate-row counts.',
                cv=cv,final_fit=fit,training=training_score,student_test=test_score,class_balance=balance,
                timing=dict(threads=THREADS,warmup_rows=min(1024,len(test['X'])),repeats=5,statistic='median',
                            inference_seconds=inference,repeat_seconds=repeats,seconds_per_candidate=inference/len(test['X']),
                            seconds_per_observation=inference/test_score['observations'],
                            included='StandardScaler transform and MLP predict_proba; excluded loading/candidate generation/evaluation.'),
                initial_configuration=dict(hidden_layer_sizes=[32,16],activation='relu',output='sigmoid',solver='adam',alpha=.001,
                                           learning_rate_init=.001,batch_size=2048,max_epochs=max_epochs,tol=.0001,patience=15,
                                           epoch_method='partial_fit preserves Adam state; one complete pass per epoch.',
                                           stop_rule='Training objective plateau only; validation history is observed, not used for stopping.',
                                           preprocessing='StandardScaler fitted on training-fold eligible rows only.'),
                label_caveat='Pointwise targets are weak monotone reference alignment; distances >50m excluded from supervised metrics.',
                continuity_policy='All consecutive original predictions in a trip, same arc or directed end-to-start node adjacency, 0<dt<=10s; independent of reference labels.',
                winner='Not selected: KNN, XGBoost, Random Forest and calibrated RBF-SVM owner results are pending.')
    (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    plot_results(out,rows,history,test_score,cv)
    print('MLP final metrics: '+json.dumps(dict(cv_macro_f1=cv['macro_f1'],student_test=test_score,final_fit=fit)),flush=True)
    return result

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--results',default='results');a.add_argument('--out',default='mlp');a.add_argument('--max-epochs',type=int,default=200)
    args=a.parse_args();run(args.results,args.out,args.max_epochs)

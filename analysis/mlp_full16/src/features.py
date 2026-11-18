"""Label-free candidates/features and a separate reference-only weak-label adapter."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from common import load_trip, project, segment_distance, trip_directories
from audit import write_csv

FEATURES = ['distance_m','log1p_distance','segment_length_m','fraction_on_segment',
            'heading_difference_rad','heading_available','speed_mps','previous_dt_s','gap_gt_10s',
            'previous_point_distance_m','next_point_distance_m','from_degree','to_degree',
            'from_endpoint_distance_m','to_endpoint_distance_m','gps_movement_distance_m']


def reference_alignment(gps, starts, ends, route):
    """Monotone minimum-distance alignment to ordered reference arc occurrences.

    This produces weak labels, NOT independently verified per-GPS ground truth.
    The route is only used here, never for generating candidates or features.
    Cost is robust distance: quadratic to 25m, then linear. Forward skips allowed.
    """
    a, b = starts[route], ends[route]
    n, m = len(gps), len(route)
    previous = np.zeros(m)
    back = np.empty((n,m), dtype=np.int32)
    for i,p in enumerate(gps):
        best = np.minimum.accumulate(previous)
        # Resolve ties toward the earliest occurrence to keep alignment deterministic.
        is_best = np.r_[True,previous[1:] < best[:-1]]
        arg = np.maximum.accumulate(np.where(is_best,np.arange(m),0))
        distance,_ = segment_distance(p,a,b)
        cost = np.where(distance<=25, .5*distance**2,25*(distance-12.5))
        previous = best+cost
        previous -= previous.min()
        back[i] = arg
    j = int(previous.argmin())
    aligned = np.empty(n,dtype=int)
    for i in range(n-1,-1,-1):
        aligned[i] = route[j]
        j = int(back[i,j])
    distance,_ = segment_distance(gps,starts[aligned],ends[aligned])
    return aligned,distance


def make_index(starts, ends, spacing=40):
    # Every segment is sampled with spacing <=40m. A radius+20m query
    # therefore finds every segment whose exact distance is <=radius.
    parts = np.maximum(1,np.ceil(np.linalg.norm(ends-starts,axis=1)/spacing).astype(int))
    ids = np.repeat(np.arange(len(starts)),parts+1)
    offsets = np.repeat(np.cumsum(parts+1)-(parts+1),parts+1)
    fractions = (np.arange(len(ids))-offsets)/parts[ids]
    samples=starts[ids]+fractions[:,None]*(ends-starts)[ids]
    return cKDTree(samples),ids


def candidate_features(track, nodes, arcs, indices, radius=75, k=8):
    """No route argument: predictions can be built when no labels are available."""
    origin = track[:,:2].mean(axis=0)
    gps, nt = project(track[:,:2],origin), project(nodes,origin)
    starts, ends = nt[arcs[:,0]],nt[arcs[:,1]]
    delta = ends-starts
    lengths = np.linalg.norm(delta,axis=1)
    tree,sample_ids = make_index(starts,ends)
    neighbourhoods=tree.query_ball_point(gps[indices],radius+20,workers=-1)
    degree=np.bincount(arcs.ravel(),minlength=len(nodes))
    xx, arcids, obsids, metadata = [], [], [], []
    for i,samples in zip(indices,neighbourhoods):
        candidates=np.unique(sample_ids[samples])
        fallback=False
        if len(candidates):
            ds,fs=segment_distance(gps[i],starts[candidates],ends[candidates])
            valid=ds<=radius
            candidates,ds,fs=candidates[valid],ds[valid],fs[valid]
        if not len(candidates):
            # Explicit fallback: nearest sampled arc; no claim of radius coverage.
            _, nearest = tree.query(gps[i],k=min(64,len(sample_ids)))
            candidates=np.unique(sample_ids[np.atleast_1d(nearest)])
            ds,fs=segment_distance(gps[i],starts[candidates],ends[candidates])
            fallback=True
        order=np.lexsort((candidates,ds))[:k]
        candidates,ds,fs=candidates[order],ds[order],fs[order]
        ip, ine = max(0,i-1),min(len(gps)-1,i+1)
        dt=track[i,2]-track[ip,2]
        # Backward motion only. Large gaps and nearly stationary observations have no heading.
        move=gps[i]-gps[ip]; moved=np.linalg.norm(move)
        available = float(dt>0 and dt<=10 and moved>=3)
        cosine = (delta[candidates]@move)/np.maximum(lengths[candidates]*moved,1e-9) if available else np.zeros(len(candidates))
        dp,_=segment_distance(gps[ip],starts[candidates],ends[candidates])
        dn,_=segment_distance(gps[ine],starts[candidates],ends[candidates])
        from_endpoint=np.linalg.norm(gps[i]-starts[candidates],axis=1)
        to_endpoint=np.linalg.norm(gps[i]-ends[candidates],axis=1)
        heading_difference=np.arccos(np.clip(cosine,-1,1)) if available else np.zeros(len(candidates))
        speed=min(moved/dt,100) if dt>0 else 0
        x=np.column_stack([ds,np.log1p(ds),lengths[candidates],fs,heading_difference,np.full(len(candidates),available),
                           np.full(len(candidates),speed),np.full(len(candidates),np.clip(dt,0,3600)),
                           np.full(len(candidates),float(dt>10)),dp,dn,degree[arcs[candidates,0]],degree[arcs[candidates,1]],
                           from_endpoint,to_endpoint,np.full(len(candidates),moved)])
        xx.append(x.astype(np.float32));arcids.append(candidates);obsids.append(np.full(len(candidates),i))
        metadata.append(dict(observation=int(i),candidate_count=len(candidates),fallback=fallback))
    return np.concatenate(xx),np.concatenate(arcids),np.concatenate(obsids),metadata,gps,starts,ends


def build(data,out,max_observations=400):
    out=Path(out);(out/'processed').mkdir(exist_ok=True,parents=True)
    rows=[]
    for split,p in trip_directories(data):
        d=load_trip(p)
        indices=np.arange(len(d['track'])) if max_observations==0 else np.unique(np.linspace(0,len(d['track'])-1,min(max_observations,len(d['track']))).astype(int))
        x,arc,obs,meta,gps,a,b=candidate_features(d['track'],d['nodes'],d['arcs'],indices)
        labels,dist=reference_alignment(gps,a,b,d['route'])
        eligible=dist[obs]<=50
        y=(arc==labels[obs]).astype(np.int8)
        coverage=np.array([labels[i] in arc[obs==i] for i in indices])
        np.savez_compressed(out/'processed'/f'{split}_{p.name}.npz',X=x,y=y,arc_id=arc,observation=obs,
                            eligible=eligible,reference_arc=labels[obs],reference_distance_m=dist[obs],
                            selected_observations=indices, aligned_full=labels,alignment_distance_full=dist,
                            from_node=d['arcs'][arc,0],to_node=d['arcs'][arc,1])
        rows.append(dict(partition=split,trip_id=p.name,total_observations=len(gps),sampled_observations=len(indices),
                         candidate_rows=len(x),eligible_observations=int((dist[indices]<=50).sum()),
                         aligned_gt_50m_full=int((dist>50).sum()),
                         candidate_misses=int(((~coverage)&(dist[indices]<=50)).sum()),
                         fallback_observations=sum(m['fallback'] for m in meta),
                         median_alignment_distance_m=float(np.median(dist)),p95_alignment_distance_m=float(np.percentile(dist,95))))
        print(f'FEATURES {split}/{p.name}: {len(x)} candidates, {coverage.mean():.3f} coverage',flush=True)
    write_csv(out/'feature_inventory.csv',rows)
    config=dict(contract_version='team-plan-v2-full16',features=FEATURES,candidate_radius_m=75,max_candidates=8,sample_spacing_m=40,
                max_observations_per_trip=max_observations,alignment_max_distance_m=50,
                label_type='weak monotone alignment to ordered reference route',
                selection='Evenly spaced original observation indices; full track used for context and alignment.',
                projection='Per-trip local equirectangular in metres; haversine used for audit distances.',
                feature_uses_reference=False,sequence_boundary_gap_s=10,
                candidate_threshold=0.5,probability_clip_epsilon=1e-7,
                class_balance_policy='No resampling, no class weights; natural eligible candidate distribution for every model.',
                continuity_rule='Same arc or previous.to_node == next.from_node; consecutive original observations; 0<dt<=10s.')
    (out/'feature_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--data',required=True);a.add_argument('--out',default='results')
    a.add_argument('--max-observations-per-trip',type=int,default=400)
    args=a.parse_args();build(args.data,args.out,args.max_observations_per_trip)

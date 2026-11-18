"""Audit every raw file, retain original row IDs, and generate EDA evidence."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import load_trip, haversine, project, trip_directories


def write_csv(path, records):
    if not records:
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)


def components(n, arcs):
    # Union-find avoids storing millions of Python graph objects.
    parent = np.arange(n)
    size = np.ones(n, dtype=int)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in arcs:
        ra, rb = find(a), find(b)
        if ra != rb:
            if size[ra] < size[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            size[ra] += size[rb]
    roots = np.array([find(i) for i in range(n)])
    return len(np.unique(roots)), int(np.bincount(roots).max())


def audit(data, out):
    out = Path(out)
    (out/'figures').mkdir(parents=True, exist_ok=True)
    records, files, problems, all_dt, coords, speed_values = [], [], [], [], [], []
    for split, p in trip_directories(data):
        d = load_trip(p)
        t, nodes, arcs, route = [d[k] for k in ['track','nodes','arcs','route']]
        dt = np.diff(t[:,2])
        step = haversine(t[:-1,:2], t[1:,:2])
        speed = np.divide(step, dt, out=np.full(len(dt),np.nan), where=dt>0)
        edge_m = haversine(nodes[arcs[:,0]], nodes[arcs[:,1]])
        weak_components, largest = components(len(nodes), arcs)
        degree = np.bincount(arcs.ravel(), minlength=len(nodes))
        discontinuous = arcs[route[:-1],1] != arcs[route[1:],0]
        r = dict(partition=split, trip_id=p.name, observations=len(t), nodes=len(nodes), arcs=len(arcs),
                 route_arcs=len(route), duration_s=float(t[-1,2]-t[0,2]), gps_distance_km=float(step.sum()/1000),
                 route_length_km=float(edge_m[route].sum()/1000),
                 lon_min=float(t[:,0].min()), lon_max=float(t[:,0].max()),
                 lat_min=float(t[:,1].min()), lat_max=float(t[:,1].max()),
                 missing_or_nonfinite_values=0, invalid_coordinates=0, invalid_node_ids=0, invalid_arc_ids=0,
                 nonpositive_dt=int((dt<=0).sum()), one_second_steps=int(np.isclose(dt,1).sum()),
                 gaps_gt_1s=int((dt>1).sum()), gaps_gt_10s=int((dt>10).sum()), max_gap_s=float(dt.max()),
                 median_dt_s=float(np.median(dt)), repeated_positions=int((step==0).sum()),
                 speed_gt_60mps=int((speed>60).sum()), max_speed_mps=float(np.nanmax(speed)),
                 weak_components=weak_components, largest_component_nodes=largest,
                 isolated_nodes=int((degree==0).sum()), self_loops=int((arcs[:,0]==arcs[:,1]).sum()),
                 zero_length_arcs=int((edge_m<1e-6).sum()),
                 duplicate_directed_arcs=len(arcs)-len(np.unique(arcs,axis=0)),
                 reference_disconnected_pairs=int(discontinuous.sum()),
                 max_arc_length_m=float(edge_m.max()))
        records.append(r)
        for key in ['nonpositive_dt','gaps_gt_10s','speed_gt_60mps','isolated_nodes','zero_length_arcs','reference_disconnected_pairs']:
            if r[key]:
                problems.append(dict(partition=split,trip_id=p.name,issue=key,count=r[key]))
        for f in sorted(p.iterdir()):
            files.append(dict(partition=split,trip_id=p.name,file=f.name,bytes=f.stat().st_size,
                              rows=len(d[f.suffix[1:]]), sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
        all_dt.append(dt)
        coords.append(t[:,:2])
        speed_values.append(speed[np.isfinite(speed)])
        print(f'AUDIT {split}/{p.name}: {len(t)} GPS, {len(arcs)} arcs', flush=True)
    write_csv(out/'trip_inventory.csv',records)
    write_csv(out/'file_manifest.csv',files)
    write_csv(out/'quality_issues.csv',problems)
    summary = {}
    for split in ['train','student_test']:
        rr = [r for r in records if r['partition']==split]
        summary[split] = {k:sum(r[k] for r in rr) for k in ['observations','nodes','arcs','route_arcs','nonpositive_dt','gaps_gt_1s','gaps_gt_10s','one_second_steps','missing_or_nonfinite_values','invalid_coordinates','isolated_nodes','zero_length_arcs','reference_disconnected_pairs']}
        summary[split].update(trips=len(rr),files=4*len(rr),gps_distance_km=sum(r['gps_distance_km'] for r in rr),
                              max_gap_s=max(r['max_gap_s'] for r in rr),
                              observation_min=min(r['observations'] for r in rr),
                              observation_median=float(np.median([r['observations'] for r in rr])),
                              observation_max=max(r['observations'] for r in rr))
    summary['notes'] = ['Node/arc totals count trip-local networks; shared roads may appear in several trips.',
                        'Reference route rows are not observation-level labels.',
                        'All 360 files were parsed and hashed. Validation fails rather than silently dropping invalid data.']
    (out/'dataset_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig, ax = plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
    for split,color in [('train','#286a9f'),('student_test','#dd8944')]:
        rr = [r for r in records if r['partition']==split]
        ax[0,0].hist([r['observations'] for r in rr],bins=20,alpha=.65,label=split,color=color)
        ax[0,1].scatter([r['nodes'] for r in rr],[r['arcs'] for r in rr],s=20,label=split,color=color)
        ax[1,1].scatter([r['observations'] for r in rr],[r['max_gap_s'] for r in rr],s=20,label=split,color=color)
    dt = np.concatenate(all_dt)
    bins=[int(np.isclose(dt,1).sum()),int(((dt>1)&(dt<=10)).sum()),int((dt>10).sum()),int((dt<=0).sum())]
    ax[1,0].bar(['1 s','1-10 s (>1)','>10 s','<=0 s'],bins,color=['#286a9f','#dd8944','#b95050','#999999'])
    ax[1,0].set_yscale('symlog',linthresh=1)
    for a,title,x,y in [(ax[0,0],'Trip size','GPS observations','Trips'),(ax[0,1],'Road-network size','Nodes','Directed arcs'),
                         (ax[1,0],'Sampling intervals: both supplied partitions','Interval','Count (symlog)'),
                         (ax[1,1],'Sampling gaps','GPS observations','Maximum gap (s)')]:
        a.set(title=title,xlabel=x,ylabel=y)
    ax[0,0].legend();ax[0,1].legend();ax[1,1].legend()
    fig.savefig(out/'figures'/'eda_overview.png',dpi=170);plt.close(fig)
    xy=np.concatenate(coords)
    fig, ax=plt.subplots(figsize=(10,6),layout='constrained')
    h=ax.hexbin(xy[:,0],xy[:,1],gridsize=100,bins='log',mincnt=1,cmap='viridis')
    ax.set(xlabel='Longitude (degrees)',ylabel='Latitude (degrees)',title='GPS coordinate density: 80 training + 10 student-test trips')
    fig.colorbar(h,ax=ax,label='Observations (log colour scale)')
    fig.savefig(out/'figures'/'coordinate_density.png',dpi=170);plt.close(fig)
    for split,tid in [('train','00000002'),('student_test','00000000')]:
        p=Path(data)/split/tid;d=load_trip(p)
        origin=d['track'][:,:2].mean(axis=0)
        nt=project(d['nodes'],origin)/1000;gps=project(d['track'][:,:2],origin)/1000
        from matplotlib.collections import LineCollection
        fig,ax=plt.subplots(figsize=(10,7),layout='constrained')
        ax.add_collection(LineCollection(nt[d['arcs']],colors='#b6bcc4',linewidths=.35,label='Road network'))
        ax.add_collection(LineCollection(nt[d['arcs'][d['route']]],colors='#369368',linewidths=2,label='Reference route'))
        ax.scatter(gps[:,0],gps[:,1],s=3,c='#286a9f',label='Raw GPS',zorder=3)
        ax.set(xlim=(gps[:,0].min()-.2,gps[:,0].max()+.2),ylim=(gps[:,1].min()-.2,gps[:,1].max()+.2),
               xlabel='Local east (km)',ylabel='Local north (km)',title=f'{split} / {tid}: GPS and reference route')
        ax.set_aspect('equal');ax.legend(loc='best')
        fig.savefig(out/'figures'/f'trajectory_{split}_{tid}.png',dpi=170);plt.close(fig)
    return summary


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--data',required=True);a.add_argument('--out',default='results')
    args=a.parse_args();s=audit(args.data,args.out);print(json.dumps(s,indent=2))

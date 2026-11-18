"""Actual offline Folium and static views. This module does not train or select a model."""
from pathlib import Path
import numpy as np
import folium
from branca.element import Element
from common import project

def make_map(d,path,title,predictions=None,assets=None):
    gps=d['track'][:,:2];nodes=d['nodes'];arcs=d['arcs']
    m=folium.Map(location=gps.mean(axis=0)[::-1].tolist(),tiles=None,control_scale=True,
                 prefer_canvas=True,zoom_start=13)
    # Keep the required JS/CSS minimal. Local copies permit fully offline maps.
    m.default_js=[] if assets else [('leaflet','https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.js')]
    m.default_css=[] if assets else [('leaflet_css','https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.css')]
    if assets:
        assets=Path(assets)
        m.get_root().header.add_child(Element('<style>'+ (assets/'leaflet.css').read_text(encoding='utf-8')+'</style>'))
        m.get_root().header.add_child(Element('<script>'+ (assets/'leaflet.js').read_text(encoding='utf-8')+'</script>'))
    # Optional online background; not selected and not required for offline geometry.
    folium.TileLayer('OpenStreetMap',name='Online street background (internet required)',show=False).add_to(m)
    lo=gps.min(axis=0)-.006;hi=gps.max(axis=0)+.006
    ab=nodes[arcs]
    intersects=((ab.max(axis=1)>=lo)&(ab.min(axis=1)<=hi)).all(axis=1)
    def layer(name,ids,color,weight,opacity):
        collection=folium.FeatureGroup(name=name,show=True)
        geo=dict(type='Feature',geometry=dict(type='MultiLineString',coordinates=nodes[arcs[ids]].tolist()),properties={})
        folium.GeoJson(geo,style_function=lambda _,c=color,w=weight,o=opacity:dict(color=c,weight=w,opacity=o)).add_to(collection)
        collection.add_to(m)
    layer('Relevant road network',np.flatnonzero(intersects),'#8593a6',1,.5)
    layer('Supplied reference route (offline analysis)',d['route'],'#208f4d',6,.7)
    if predictions is not None:
        # Segments are independent classifier choices. No lines are invented to bridge gaps.
        layer('Classifier-selected segments (no HMM)',np.unique(predictions),'#d85f1f',3,.95)
    observed=folium.FeatureGroup(name='Original GPS observations',show=True)
    select=np.unique(np.linspace(0,len(gps)-1,min(700,len(gps))).astype(int))
    for i in select:
        folium.CircleMarker(gps[i,::-1].tolist(),radius=2,color='#1768c0',fill=True,fill_opacity=.9,
                            tooltip=f'Original GPS row {i}, t={d["track"][i,2]:g}s').add_to(observed)
    observed.add_to(m)
    folium.CircleMarker(gps[0,::-1].tolist(),radius=7,color='#111111',fill=False,tooltip='First GPS observation').add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([gps.min(axis=0)[::-1].tolist(),gps.max(axis=0)[::-1].tolist()],padding=(30,30))
    legend=f'''<div style="position:fixed;bottom:30px;left:25px;z-index:9999;background:white;padding:14px 18px;border:1px solid #adb7c5;border-radius:7px;font:13px Arial;max-width:440px;box-shadow:0 2px 8px #9996">
    <b>{title}</b><br><span style="color:#1768c0">Blue dots</span>: raw GPS (up to 700 shown)<br>
    <span style="color:#208f4d">Green</span>: supplied reference route<br>
    <span style="color:#d85f1f">Orange</span>: classifier-selected segments, where shown<br>
    Grey: road network. Reference is for offline comparison only.<br>
    Geographical road geometry works offline; optional street tiles need internet.</div>'''
    m.get_root().html.add_child(Element(legend))
    m.save(str(path))


def static_prediction_plot(d,predicted,path,title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    origin=d['track'][:,:2].mean(axis=0)
    n=project(d['nodes'],origin)/1000;g=project(d['track'][:,:2],origin)/1000
    fig,ax=plt.subplots(figsize=(10,6.5),layout='constrained')
    ax.add_collection(LineCollection(n[d['arcs']],colors='#b6bcc4',linewidths=.45,label='Road network',zorder=0))
    ax.scatter(g[:,0],g[:,1],s=3,c='#1768c0',alpha=.65,label='Original GPS',zorder=1)
    ax.add_collection(LineCollection(n[d['arcs'][d['route']]],colors='#208f4d',linewidths=4,alpha=.65,label='Supplied reference',zorder=2))
    ax.add_collection(LineCollection(n[d['arcs'][np.unique(predicted)]],colors='#d85f1f',linewidths=1.5,label='Classifier-selected segments',zorder=3))
    ax.set(xlim=(g[:,0].min()-.12,g[:,0].max()+.12),ylim=(g[:,1].min()-.12,g[:,1].max()+.12),
           xlabel='Local east (km)',ylabel='Local north (km)',title=title)
    ax.set_aspect('equal');ax.legend(loc='best',fontsize=8)
    fig.savefig(path,dpi=180);plt.close(fig)

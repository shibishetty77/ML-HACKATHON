"""Generate an interactive evaluator-trip map from deployed XGBoost and HMM.

The reference route is loaded only after predictions and is displayed solely
for offline comparison. The HTML needs internet for Leaflet and optional OSM
tiles; all road, GPS and route geometry comes from the evaluator ZIP.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import folium
import numpy as np
from branca.element import Element

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.data import TripRepository  # noqa: E402
from backend.app.geometry import RoadNetwork, motion_features  # noqa: E402
from backend.app.hmm import decode  # noqa: E402
from backend.app.model import CandidateClassifier  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator-zip", type=Path,
                        default=Path.home() / "Downloads" / "evaluator_test.zip")
    parser.add_argument("--trip", default="00000030")
    parser.add_argument("--first-observation", type=int, default=220)
    parser.add_argument("--observations", type=int, default=100)
    args = parser.parse_args()
    repo = TripRepository(args.evaluator_zip)
    trip = repo.load(args.trip)
    road = RoadNetwork(trip)
    xy, speed, heading = motion_features(trip.track, road)
    groups = [road.candidates(xy[i], speed[i], None if np.isnan(heading[i]) else float(heading[i]))
              for i in range(len(trip.track))]
    CandidateClassifier(ROOT).score(groups)
    decoded = decode(road, groups, trip.track, xy)
    reference = repo.reference_route(args.trip)
    repo.close()

    first = max(0, args.first_observation)
    last = min(len(trip.track), first + args.observations)
    gps = trip.track[first:last, :2]
    lower = gps.min(axis=0) - np.array([0.0015, 0.0015])
    upper = gps.max(axis=0) + np.array([0.0015, 0.0015])
    endpoints = trip.nodes[trip.arcs]
    visible = (endpoints.max(axis=1) >= lower).all(axis=1) & \
              (endpoints.min(axis=1) <= upper).all(axis=1)
    network_ids = np.flatnonzero(visible)
    reference_ids = np.unique(reference[visible[reference]])
    predicted_ids = np.unique(np.array([arc for chunk in decoded.chunks for arc in chunk], dtype=int))
    predicted_ids = predicted_ids[visible[predicted_ids]]

    m = folium.Map(location=[float(gps[:, 1].mean()), float(gps[:, 0].mean())],
                   zoom_start=15, tiles=None, prefer_canvas=True, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="Street background", show=True).add_to(m)

    def roads(name: str, ids: np.ndarray, color: str, width: int, opacity: float):
        layer = folium.FeatureGroup(name=name, show=True)
        geometry = {"type": "Feature", "geometry": {
            "type": "MultiLineString", "coordinates": trip.nodes[trip.arcs[ids]].tolist()},
            "properties": {}}
        folium.GeoJson(geometry, style_function=lambda _, c=color, w=width, o=opacity:
                       {"color": c, "weight": w, "opacity": o}).add_to(layer)
        layer.add_to(m)

    roads("Road network", network_ids, "#64748b", 1, 0.40)
    roads("Reference route", reference_ids, "#16a34a", 7, 0.72)
    roads("XGBoost plus HMM route", predicted_ids, "#d946ef", 4, 0.95)
    raw = folium.FeatureGroup(name="Original GPS points", show=True)
    for i in range(first, last):
        lon, lat, timestamp = trip.track[i]
        folium.CircleMarker([float(lat), float(lon)], radius=1.7,
                            color="#1d4ed8", fill=True, fill_opacity=0.65,
                            tooltip=f"GPS observation {i}; t={timestamp:g}s").add_to(raw)
    raw.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[float(lower[1]), float(lower[0])],
                  [float(upper[1]), float(upper[0])]], padding=(25, 25))
    legend = f'''<div style="position:fixed;bottom:24px;left:24px;z-index:9999;
    background:#fff;padding:12px 16px;border:1px solid #94a3b8;border-radius:6px;
    font:13px Arial;box-shadow:0 2px 8px #8888">
    <b>Team 9 PathSnap - trip {args.trip}</b><br>
    Blue: raw GPS observations {first}-{last-1}<br>
    Grey: relevant directed road arcs<br>
    Green: supplied reference arc geometry<br>
    Magenta: deployed XGBoost + HMM arc geometry<br>
    Reference route displayed only after inference.</div>'''
    m.get_root().html.add_child(Element(legend))
    out = ROOT / "analysis" / "results" / f"folium_trip_{args.trip}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    print(out)
    print("gps", len(gps), "network", len(network_ids),
          "reference", len(reference_ids), "predicted", len(predicted_ids))


if __name__ == "__main__":
    main()

'use client';

import { useEffect, useRef } from 'react';
import L from 'leaflet';
import { GpsObservation } from '@/types/trip';
import { PredictedRoute } from '@/types/prediction';

interface LeafletInnerMapProps {
  gpsObservations?: GpsObservation[];
  predictedRoute?: PredictedRoute;
  showGpsLayer: boolean;
  showRouteLayer: boolean;
  activeObsId?: string | null;
  onSelectObs?: (obs: GpsObservation) => void;
}

function makeGpsPopup(obs: GpsObservation): HTMLElement {
  const popup = document.createElement('div');
  popup.className = 'text-xs p-1 space-y-1.5 min-w-[160px]';

  const heading = document.createElement('div');
  heading.className = 'font-mono font-bold text-blue-400';
  heading.textContent = obs.obs_id;
  popup.appendChild(heading);

  const details = document.createElement('div');
  details.className = 'text-[11px] text-slate-300 space-y-0.5';
  const rows = [
    `Lat/Lon: ${obs.lat.toFixed(5)}, ${obs.lon.toFixed(5)}`,
    ...(obs.speed_mps != null ? [`Speed: ${obs.speed_mps.toFixed(1)} m/s`] : []),
    ...(obs.matched_arc_id != null ? [`Matched Arc: #${obs.matched_arc_id}`] : []),
    ...(obs.confidence != null ? [`Confidence: ${(obs.confidence * 100).toFixed(1)}%`] : []),
  ];
  for (const text of rows) {
    const row = document.createElement('div');
    row.textContent = text;
    details.appendChild(row);
  }
  popup.appendChild(details);
  return popup;
}

function makeRoutePopup(segmentCount: number): HTMLElement {
  const popup = document.createElement('div');
  popup.className = 'text-xs space-y-1';
  const heading = document.createElement('div');
  heading.className = 'font-bold text-cyan-400';
  heading.textContent = 'Predicted Reconstructed Route';
  const details = document.createElement('div');
  details.className = 'text-slate-300';
  details.textContent = `Segments Matched: ${segmentCount}`;
  popup.append(heading, details);
  return popup;
}

export default function LeafletInnerMap({
  gpsObservations = [],
  predictedRoute,
  showGpsLayer,
  showRouteLayer,
  activeObsId,
  onSelectObs,
}: LeafletInnerMapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<L.Map | null>(null);

  // React can replay effects in development; a removed map has no panes to reuse.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const map = L.map(container, {
      zoomControl: true,
      scrollWheelZoom: true,
    }).setView([51.5074, -0.1278], 14);
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a>',
      maxZoom: 19,
    }).addTo(map);
    mapRef.current = map;

    const frame = requestAnimationFrame(() => map.invalidateSize());
    return () => {
      cancelAnimationFrame(frame);
      mapRef.current = null;
      map.remove();
    };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    const layers = L.layerGroup().addTo(map);
    const bounds: L.LatLngTuple[] = [];
    for (const obs of gpsObservations) bounds.push([obs.lat, obs.lon]);

    if (predictedRoute) {
      const lines = predictedRoute.polylines?.length
        ? predictedRoute.polylines
        : [predictedRoute.predicted_polyline];
      for (const line of lines) {
        if (line.length === 0) continue;
        bounds.push(...line);
        if (!showRouteLayer) continue;

        L.polyline(line, {
          color: '#06b6d4', weight: 6, opacity: 0.4,
          lineCap: 'round', lineJoin: 'round',
        }).addTo(layers);
        L.polyline(line, {
          color: '#38bdf8', weight: 3.5, opacity: 0.95,
          lineCap: 'round', lineJoin: 'round',
        }).bindPopup(makeRoutePopup(predictedRoute.num_matched_segments)).addTo(layers);
      }
    }

    if (showGpsLayer) {
      for (const obs of gpsObservations) {
        const selected = activeObsId === obs.obs_id;
        L.circleMarker([obs.lat, obs.lon], {
          radius: selected ? 6 : 3.5,
          color: selected ? '#ffffff' : '#3b82f6',
          fillColor: selected ? '#38bdf8' : '#2563eb',
          fillOpacity: selected ? 1 : 0.75,
          weight: selected ? 2 : 1,
        })
          .bindPopup(makeGpsPopup(obs))
          .on('click', () => onSelectObs?.(obs))
          .addTo(layers);
      }
    }

    if (bounds.length > 0) {
      map.fitBounds(L.latLngBounds(bounds), { padding: [40, 40], maxZoom: 17 });
    }

    return () => {
      if (mapRef.current === map) map.removeLayer(layers);
    };
  }, [gpsObservations, predictedRoute, showGpsLayer, showRouteLayer, activeObsId, onSelectObs]);

  return (
    <div className="absolute inset-0 rounded-xl overflow-hidden border border-slate-800 bg-slate-950">
      <div ref={containerRef} className="w-full h-full" />
    </div>
  );
}

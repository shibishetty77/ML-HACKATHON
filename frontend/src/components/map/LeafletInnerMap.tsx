'use client';

import React, { useEffect, useMemo } from 'react';
import {
  MapContainer,
  TileLayer,
  Polyline,
  CircleMarker,
  Popup,
  useMap,
} from 'react-leaflet';
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

// Helper component to auto-fit map bounds to coordinates
function MapAutoFitter({
  points,
  route,
}: {
  points?: GpsObservation[];
  route?: PredictedRoute;
}) {
  const map = useMap();

  useEffect(() => {
    const latLngs: L.LatLngTuple[] = [];

    if (points && points.length > 0) {
      points.forEach((p) => latLngs.push([p.lat, p.lon]));
    }
    if (route && route.predicted_polyline && route.predicted_polyline.length > 0) {
      route.predicted_polyline.forEach((pt) => latLngs.push([pt[0], pt[1]]));
    }

    if (latLngs.length > 0) {
      const bounds = L.latLngBounds(latLngs);
      map.fitBounds(bounds, { padding: [40, 40], maxZoom: 17 });
    }
  }, [map, points, route]);

  return null;
}

export default function LeafletInnerMap({
  gpsObservations = [],
  predictedRoute,
  showGpsLayer,
  showRouteLayer,
  activeObsId,
  onSelectObs,
}: LeafletInnerMapProps) {
  // Center fallback (e.g. London / trajectory center)
  const defaultCenter: [number, number] = useMemo(() => {
    if (gpsObservations.length > 0) {
      return [gpsObservations[0].lat, gpsObservations[0].lon];
    }
    if (predictedRoute?.predicted_polyline && predictedRoute.predicted_polyline.length > 0) {
      return predictedRoute.predicted_polyline[0];
    }
    return [51.5074, -0.1278];
  }, [gpsObservations, predictedRoute]);

  const polylineCoords: [number, number][] = useMemo(() => {
    if (!predictedRoute?.predicted_polyline) return [];
    return predictedRoute.predicted_polyline;
  }, [predictedRoute]);

  return (
    <div className="w-full h-full relative rounded-xl overflow-hidden border border-slate-800 bg-slate-950">
      <MapContainer
        center={defaultCenter}
        zoom={14}
        className="w-full h-full"
        zoomControl={true}
        scrollWheelZoom={true}
      >
        {/* Dark Matter Basemap Tiles */}
        <TileLayer
          attribution='&copy; <a href="https://carto.com/">CARTO</a>'
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          maxZoom={19}
        />

        <MapAutoFitter points={gpsObservations} route={predictedRoute} />

        {/* 1. Predicted Route Polyline */}
        {showRouteLayer && polylineCoords.length > 0 && (
          <>
            {/* Outer glow stroke */}
            <Polyline
              positions={polylineCoords}
              pathOptions={{
                color: '#06b6d4',
                weight: 6,
                opacity: 0.4,
                lineCap: 'round',
                lineJoin: 'round',
              }}
            />
            {/* Solid core line */}
            <Polyline
              positions={polylineCoords}
              pathOptions={{
                color: '#38bdf8',
                weight: 3.5,
                opacity: 0.95,
                lineCap: 'round',
                lineJoin: 'round',
              }}
            >
              <Popup>
                <div className="text-xs space-y-1">
                  <div className="font-bold text-cyan-400">
                    Predicted Reconstructed Route
                  </div>
                  <div className="text-slate-300">
                    Segments Matched: {predictedRoute?.num_matched_segments || 0}
                  </div>
                </div>
              </Popup>
            </Polyline>
          </>
        )}

        {/* 2. GPS Observations (Circles) */}
        {showGpsLayer &&
          gpsObservations.map((obs) => {
            const isSelected = activeObsId === obs.obs_id;
            return (
              <CircleMarker
                key={obs.obs_id}
                center={[obs.lat, obs.lon]}
                radius={isSelected ? 6 : 3.5}
                pathOptions={{
                  color: isSelected ? '#ffffff' : '#3b82f6',
                  fillColor: isSelected ? '#38bdf8' : '#2563eb',
                  fillOpacity: isSelected ? 1 : 0.75,
                  weight: isSelected ? 2 : 1,
                }}
                eventHandlers={{
                  click: () => onSelectObs && onSelectObs(obs),
                }}
              >
                <Popup>
                  <div className="text-xs p-1 space-y-1.5 min-w-[160px]">
                    <div className="font-mono font-bold text-blue-400">
                      {obs.obs_id}
                    </div>
                    <div className="text-[11px] text-slate-300 space-y-0.5">
                      <div>
                        Lat/Lon: {obs.lat.toFixed(5)}, {obs.lon.toFixed(5)}
                      </div>
                      {obs.speed_mps !== undefined && (
                        <div>Speed: {obs.speed_mps.toFixed(1)} m/s</div>
                      )}
                      {obs.matched_arc_id !== undefined && (
                        <div>
                          Matched Arc: <span className="font-mono text-cyan-400">#{obs.matched_arc_id}</span>
                        </div>
                      )}
                      {obs.confidence !== undefined && (
                        <div>
                          Confidence: {(obs.confidence * 100).toFixed(1)}%
                        </div>
                      )}
                    </div>
                  </div>
                </Popup>
              </CircleMarker>
            );
          })}
      </MapContainer>
    </div>
  );
}

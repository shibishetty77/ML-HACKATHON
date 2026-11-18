'use client';

import React, { useState } from 'react';
import dynamic from 'next/dynamic';
import { GpsObservation } from '@/types/trip';
import { PredictedRoute } from '@/types/prediction';
import {
  Layers,
  Eye,
  EyeOff,
  Navigation,
  Compass,
  MapPin,
  Maximize2,
  RefreshCw,
} from 'lucide-react';

// Dynamically import Leaflet map with SSR disabled to avoid 'window is not defined'
const DynamicLeafletMap = dynamic(() => import('./LeafletInnerMap'), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full min-h-[420px] rounded-xl border border-slate-800 bg-slate-950 flex flex-col items-center justify-center text-slate-500 gap-3">
      <RefreshCw className="w-6 h-6 animate-spin text-cyan-400" />
      <span className="text-xs font-mono">Initializing Geospatial Leaflet Engine...</span>
    </div>
  ),
});

interface RouteMapProps {
  gpsObservations?: GpsObservation[];
  predictedRoute?: PredictedRoute;
  tripId?: string;
  isProcessing?: boolean;
  className?: string;
}

export function RouteMap({
  gpsObservations = [],
  predictedRoute,
  tripId,
  isProcessing = false,
  className = '',
}: RouteMapProps) {
  const [showGps, setShowGps] = useState(true);
  const [showRoute, setShowRoute] = useState(true);
  const [activeObs, setActiveObs] = useState<GpsObservation | null>(null);

  const hasData = gpsObservations.length > 0 || (predictedRoute?.predicted_polyline && predictedRoute.predicted_polyline.length > 0);

  return (
    <div className={`flex flex-col relative h-full ${className}`}>
      {/* Map Control Bar */}
      <div className="flex items-center justify-between pb-3">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
            <Compass className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              Geographical Route Visualisation
              {tripId && (
                <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-800 text-cyan-400 border border-slate-700">
                  Trip #{tripId}
                </span>
              )}
            </h3>
            <p className="text-[11px] text-slate-400">
              Interactive Leaflet layer matching GPS observations to road network
            </p>
          </div>
        </div>

        {/* Layer Toggles & Stats */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowGps(!showGps)}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium border transition-colors ${
              showGps
                ? 'bg-blue-500/10 border-blue-500/30 text-blue-400'
                : 'bg-slate-900 border-slate-800 text-slate-500 hover:text-slate-300'
            }`}
            title="Toggle raw GPS observations"
          >
            {showGps ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
            <span>GPS Points ({gpsObservations.length})</span>
          </button>

          <button
            onClick={() => setShowRoute(!showRoute)}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium border transition-colors ${
              showRoute
                ? 'bg-cyan-500/10 border-cyan-500/30 text-cyan-400'
                : 'bg-slate-900 border-slate-800 text-slate-500 hover:text-slate-300'
            }`}
            title="Toggle predicted road segments"
          >
            {showRoute ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
            <span>Predicted Route ({predictedRoute?.num_matched_segments || 0})</span>
          </button>
        </div>
      </div>

      {/* Main Map Box */}
      <div className="flex-1 w-full min-h-[460px] relative rounded-xl overflow-hidden border border-slate-800 bg-slate-950">
        {isProcessing && (
          <div className="absolute inset-0 z-30 flex flex-col items-center justify-center bg-slate-950/80 backdrop-blur-sm gap-3">
            <RefreshCw className="w-8 h-8 animate-spin text-cyan-400" />
            <div className="text-center">
              <div className="text-sm font-semibold text-white">Reconstructing Route...</div>
              <div className="text-xs text-slate-400 font-mono mt-0.5">
                Viterbi sequence decoding on Laptop 1
              </div>
            </div>
          </div>
        )}

        {!hasData && !isProcessing ? (
          <div className="w-full h-full flex flex-col items-center justify-center text-slate-500 p-8 text-center gap-3">
            <Navigation className="w-10 h-10 text-slate-700" />
            <div>
              <p className="text-sm font-medium text-slate-400">No Route Loaded</p>
              <p className="text-xs text-slate-600 max-w-sm mt-1">
                Select an evaluation trip from the dashboard and click &quot;Run Inference&quot; to fetch the GPS track and predicted route.
              </p>
            </div>
          </div>
        ) : (
          <>
            <DynamicLeafletMap
              gpsObservations={gpsObservations}
              predictedRoute={predictedRoute}
              showGpsLayer={showGps}
              showRouteLayer={showRoute}
              activeObsId={activeObs?.obs_id}
              onSelectObs={(obs) => setActiveObs(obs)}
            />

            {/* Floating Map Legend */}
            <div className="absolute bottom-4 right-4 z-[1000] bg-slate-900/90 border border-slate-800 rounded-lg p-2.5 backdrop-blur shadow-xl text-[11px] space-y-1.5 min-w-[170px]">
              <div className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">
                Map Layers Legend
              </div>
              <div className="flex items-center gap-2 text-slate-300">
                <span className="w-2.5 h-2.5 rounded-full bg-blue-500 ring-2 ring-blue-900 inline-block"></span>
                <span>GPS Observations</span>
              </div>
              <div className="flex items-center gap-2 text-slate-300">
                <span className="w-4 h-1 rounded bg-cyan-400 inline-block shadow-sm shadow-cyan-400"></span>
                <span>XGBoost + HMM Route</span>
              </div>
              <div className="text-[10px] text-slate-500 pt-1 border-t border-slate-800/80 italic">
                Ground truth kept on Laptop 1
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

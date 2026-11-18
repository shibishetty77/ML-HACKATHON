'use client';

import React from 'react';
import { useApp } from '@/context/AppContext';
import { RouteMap } from '@/components/map/RouteMap';
import { CandidateInspector } from '@/components/results/CandidateInspector';
import {
  Compass,
  MapPin,
  Play,
  RefreshCw,
  Navigation,
  CheckCircle2,
} from 'lucide-react';

export default function RouteExplorerPage() {
  const {
    trips,
    selectedTripId,
    setSelectedTripId,
    predictionResult,
    isPredicting,
    executePrediction,
    serverStatus,
  } = useApp();

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-12">
      {/* Top action bar */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 p-4 rounded-xl border border-slate-800 bg-slate-900/60 backdrop-blur-sm">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
            <Compass className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-base font-bold text-white">Geospatial Route Explorer</h1>
            <p className="text-xs text-slate-400">
              Interactive map-matching canvas rendering GPS observations and Viterbi-reconstructed paths
            </p>
          </div>
        </div>

        {/* Quick Trip Switcher */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <label className="text-xs text-slate-400 font-medium">Trip:</label>
            <select
              value={selectedTripId || ''}
              onChange={(e) => setSelectedTripId(e.target.value)}
              className="bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-white font-mono focus:outline-none focus:border-cyan-500"
            >
              {trips.map((t) => (
                <option key={t.trip_id} value={t.trip_id}>
                  Trip #{t.trip_id} ({t.num_gps_points} pts)
                </option>
              ))}
            </select>
          </div>

          <button
            onClick={() => selectedTripId && executePrediction(selectedTripId)}
            disabled={!selectedTripId || isPredicting || serverStatus !== 'connected'}
            className="flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-cyan-400 hover:bg-cyan-300 text-black transition-all disabled:opacity-40"
          >
            {isPredicting ? (
              <>
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                <span>Running Viterbi...</span>
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5 fill-black" />
                <span>Execute Map-Match</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Large Map */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 backdrop-blur-sm">
        <RouteMap
          gpsObservations={predictionResult?.gps_observations}
          predictedRoute={predictionResult?.route}
          tripId={selectedTripId || undefined}
          isProcessing={isPredicting}
          className="min-h-[580px]"
        />
      </div>

      {/* Observation sequence inspector table */}
      <CandidateInspector observations={predictionResult?.gps_observations} />
    </div>
  );
}

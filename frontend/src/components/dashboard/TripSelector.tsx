'use client';

import React, { useState } from 'react';
import { useApp } from '@/context/AppContext';
import { TripSummary } from '@/types/trip';
import {
  MapPin,
  Clock,
  Navigation,
  Play,
  RefreshCw,
  Search,
  Check,
  AlertCircle,
} from 'lucide-react';

export function TripSelector() {
  const {
    trips,
    loadingTrips,
    tripsError,
    refetchTrips,
    selectedTripId,
    setSelectedTripId,
    isPredicting,
    executePrediction,
  } = useApp();

  const [searchQuery, setSearchQuery] = useState('');

  const filteredTrips = trips.filter((t) =>
    t.trip_id.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-sm">
      <div className="flex items-center justify-between pb-4 border-b border-slate-800">
        <div>
          <h2 className="text-sm font-semibold text-white flex items-center gap-2">
            Evaluator Trip Selection
            <span className="text-xs font-mono font-medium px-2 py-0.5 rounded bg-cyan-950 text-cyan-400 border border-cyan-800/40">
              {trips.length} Available
            </span>
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Select any of the 10 evaluation trips to run distributed map-matching inference
          </p>
        </div>

        <div className="flex items-center gap-2">
          {/* Search box */}
          <div className="relative">
            <Search className="w-3.5 h-3.5 absolute left-2.5 top-2.5 text-slate-500" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Filter trips..."
              className="w-36 pl-8 pr-3 py-1.5 rounded-lg border border-slate-800 bg-slate-950 text-xs text-white placeholder:text-slate-600 focus:outline-none focus:border-cyan-500 font-mono"
            />
          </div>

          <button
            onClick={() => refetchTrips()}
            disabled={loadingTrips}
            className="p-1.5 rounded-lg border border-slate-800 bg-slate-950 text-slate-400 hover:text-white hover:bg-slate-800 transition-colors disabled:opacity-50"
            title="Refresh trips from Laptop 1"
          >
            <RefreshCw className={`w-4 h-4 ${loadingTrips ? 'animate-spin text-cyan-400' : ''}`} />
          </button>
        </div>
      </div>

      {/* Error state */}
      {tripsError && (
        <div className="mt-4 p-3 rounded-lg border border-rose-500/30 bg-rose-500/10 text-rose-400 text-xs flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{tripsError}</span>
          </div>
          <button
            onClick={() => refetchTrips()}
            className="px-2.5 py-1 rounded bg-rose-500/20 hover:bg-rose-500/30 font-medium underline"
          >
            Retry
          </button>
        </div>
      )}

      {/* Loading Skeleton */}
      {loadingTrips && trips.length === 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-3 mt-4">
          {[...Array(5)].map((_, i) => (
            <div
              key={i}
              className="h-28 rounded-lg border border-slate-800 bg-slate-950/60 animate-pulse p-3 space-y-2"
            >
              <div className="h-4 w-20 bg-slate-800 rounded"></div>
              <div className="h-3 w-28 bg-slate-800/60 rounded"></div>
              <div className="h-3 w-16 bg-slate-800/40 rounded"></div>
            </div>
          ))}
        </div>
      )}

      {/* Empty State */}
      {!loadingTrips && trips.length === 0 && !tripsError && (
        <div className="text-center py-10 text-slate-500 text-xs">
          <Navigation className="w-8 h-8 mx-auto mb-2 text-slate-700" />
          No evaluation trips found on the backend. Ensure Laptop 1 has the evaluator dataset loaded.
        </div>
      )}

      {/* Trips Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-3 mt-4">
        {filteredTrips.map((trip: TripSummary) => {
          const isSelected = selectedTripId === trip.trip_id;
          const isCurrentlyInferring = isPredicting && isSelected;

          return (
            <div
              key={trip.trip_id}
              onClick={() => setSelectedTripId(trip.trip_id)}
              className={`group relative p-3 rounded-lg border transition-all cursor-pointer ${
                isSelected
                  ? 'border-cyan-500/60 bg-cyan-950/20 shadow-md shadow-cyan-950/30 ring-1 ring-cyan-500/30'
                  : 'border-slate-800 bg-slate-950/60 hover:border-slate-700 hover:bg-slate-900/60'
              }`}
            >
              <div className="flex items-center justify-between mb-1.5">
                <span className="font-mono text-sm font-bold text-white flex items-center gap-1.5">
                  Trip #{trip.trip_id}
                  {isSelected && <Check className="w-3.5 h-3.5 text-cyan-400 stroke-[3]" />}
                </span>
                <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-slate-900 text-slate-400 border border-slate-800">
                  {trip.status}
                </span>
              </div>

              <div className="space-y-1 text-xs text-slate-400">
                <div className="flex items-center gap-1.5">
                  <MapPin className="w-3 h-3 text-slate-500" />
                  <span>{trip.num_gps_points.toLocaleString()} points</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <Clock className="w-3 h-3 text-slate-500" />
                  <span>~{Math.round(trip.approx_duration_s)}s duration</span>
                </div>
              </div>

              {/* Action Button inside card */}
              <div className="mt-3 pt-2 border-t border-slate-800/80 flex items-center justify-between">
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    setSelectedTripId(trip.trip_id);
                    executePrediction(trip.trip_id);
                  }}
                  disabled={isPredicting}
                  className={`w-full py-1.5 px-2 rounded-md text-[11px] font-semibold flex items-center justify-center gap-1.5 transition-all ${
                    isSelected
                      ? 'bg-cyan-500 hover:bg-cyan-400 text-black shadow-sm'
                      : 'bg-slate-800 hover:bg-slate-700 text-slate-200'
                  } disabled:opacity-50`}
                >
                  {isCurrentlyInferring ? (
                    <>
                      <RefreshCw className="w-3 h-3 animate-spin" />
                      <span>Reconstructing...</span>
                    </>
                  ) : (
                    <>
                      <Play className="w-3 h-3 fill-current" />
                      <span>Run Prediction</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

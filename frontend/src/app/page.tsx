'use client';

import React from 'react';
import Link from 'next/link';
import { useApp } from '@/context/AppContext';
import { SystemStatusCard } from '@/components/dashboard/SystemStatusCard';
import { TripSelector } from '@/components/dashboard/TripSelector';
import { SummaryMetrics } from '@/components/dashboard/SummaryMetrics';
import { RouteMap } from '@/components/map/RouteMap';
import {
  Compass,
  ArrowRight,
  Layers,
  Sparkles,
  Zap,
} from 'lucide-react';

export default function DashboardPage() {
  const {
    predictionResult,
    isPredicting,
    selectedTripId,
  } = useApp();

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-12">
      {/* Header Banner */}
      <div className="rounded-2xl border border-slate-800 bg-gradient-to-r from-slate-900 via-slate-900/90 to-cyan-950/30 p-6 backdrop-blur shadow-xl relative overflow-hidden">
        <div className="absolute right-0 top-0 w-96 h-96 bg-cyan-500/5 rounded-full blur-3xl pointer-events-none"></div>

        <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className="px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 flex items-center gap-1.5">
                <Sparkles className="w-3 h-3" />
                UE24CS352A Machine Learning Hackathon
              </span>
            </div>
            <h1 className="text-2xl font-bold text-white tracking-tight">
              PathSnap — Intelligent GPS Map-Matching
            </h1>
            <p className="text-xs text-slate-400 max-w-2xl mt-1 leading-relaxed">
              Distributed machine learning platform pairing an ensemble XGBoost candidate classifier with a
              Hidden Markov Model (HMM) Viterbi sequence decoder to reconstruct topologically valid trajectories across two laptops.
            </p>
          </div>

          <div className="flex items-center gap-3 shrink-0">
            <Link
              href="/explorer"
              className="px-4 py-2 rounded-lg text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-white border border-slate-700 flex items-center gap-2 transition-all shadow-sm"
            >
              <Compass className="w-3.5 h-3.5 text-cyan-400" />
              <span>Full Route Explorer</span>
              <ArrowRight className="w-3 h-3 text-slate-400" />
            </Link>
          </div>
        </div>
      </div>

      {/* System Status Summary */}
      <SystemStatusCard />

      {/* Summary Metrics Bar */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs text-slate-400 px-1 font-medium">
          <span>Active Prediction Performance Metrics</span>
          {predictionResult && (
            <span className="font-mono text-cyan-400">
              Trip #{predictionResult.trip_id} Evaluated ({predictionResult.timing.total_pipeline_ms.toFixed(1)}ms)
            </span>
          )}
        </div>
        <SummaryMetrics />
      </div>

      {/* Trip Selector (10 Evaluator Trips) */}
      <TripSelector />

      {/* Interactive Map Section */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-sm space-y-4">
        <RouteMap
          gpsObservations={predictionResult?.gps_observations}
          predictedRoute={predictionResult?.route}
          tripId={selectedTripId || undefined}
          isProcessing={isPredicting}
          className="min-h-[480px]"
        />
      </div>
    </div>
  );
}

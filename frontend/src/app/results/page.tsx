'use client';

import React from 'react';
import { useApp } from '@/context/AppContext';
import { SummaryMetrics } from '@/components/dashboard/SummaryMetrics';
import { PerformanceCharts } from '@/components/results/PerformanceCharts';
import { CandidateInspector } from '@/components/results/CandidateInspector';
import {
  BarChart3,
  Play,
  RefreshCw,
  Clock,
  Layers,
  Target,
  CheckCircle2,
  AlertTriangle,
} from 'lucide-react';

export default function PredictionResultsPage() {
  const {
    predictionResult,
    isPredicting,
    selectedTripId,
    executePrediction,
    serverStatus,
  } = useApp();

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-12">
      {/* Top Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 p-5 rounded-xl border border-slate-800 bg-slate-900/60 backdrop-blur-sm">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-lg bg-blue-500/10 border border-blue-500/20 text-blue-400">
            <BarChart3 className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-base font-bold text-white flex items-center gap-2">
              Prediction Results & Evaluation Metrics
              {predictionResult && (
                <span className="text-xs font-mono px-2.5 py-0.5 rounded bg-cyan-950 text-cyan-400 border border-cyan-800/40">
                  Trip #{predictionResult.trip_id}
                </span>
              )}
            </h1>
            <p className="text-xs text-slate-400 mt-0.5">
              Server-side route arc overlap and directed continuity; candidate metrics require observation labels
            </p>
          </div>
        </div>

        {selectedTripId && (
          <button
            onClick={() => executePrediction(selectedTripId)}
            disabled={isPredicting || serverStatus !== 'connected'}
            className="flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold bg-cyan-400 hover:bg-cyan-300 text-black transition-all disabled:opacity-40"
          >
            {isPredicting ? (
              <>
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                <span>Processing Evaluation...</span>
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5 fill-black" />
                <span>Re-Run Inference for Trip #{selectedTripId}</span>
              </>
            )}
          </button>
        )}
      </div>

      {!predictionResult && !isPredicting ? (
        <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-12 text-center text-slate-500 space-y-3">
          <Target className="w-12 h-12 mx-auto text-slate-700" />
          <div className="text-sm font-semibold text-slate-300">No Prediction Results Yet</div>
          <p className="text-xs text-slate-500 max-w-md mx-auto">
            Select a trip from the Overview dashboard or use the top button to trigger the server-side XGBoost + Viterbi inference pipeline.
          </p>
        </div>
      ) : (
        <>
          {/* Metrics summary */}
          <SummaryMetrics />

          {predictionResult && predictionResult.metrics.top1_accuracy == null && predictionResult.metrics.macro_f1 == null && (
            <p className="rounded-lg border border-blue-500/20 bg-blue-500/5 px-4 py-3 text-xs text-slate-300">
              Candidate Top-1 Accuracy and Macro F1 need a reference arc for each GPS point. This evaluator set provides a reference route only, so the cards show measured route arc precision and F1 instead.
            </p>
          )}

          {/* Charts breakdown */}
          <PerformanceCharts
            metrics={predictionResult?.metrics}
            timing={predictionResult?.timing}
          />

          {/* Sequence Observation Inspector */}
          <CandidateInspector observations={predictionResult?.gps_observations} />
        </>
      )}
    </div>
  );
}

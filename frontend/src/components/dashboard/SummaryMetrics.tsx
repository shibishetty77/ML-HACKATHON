'use client';

import React from 'react';
import { useApp } from '@/context/AppContext';
import {
  Target,
  GitCommit,
  CheckCircle,
  Clock,
  Layers,
  Percent,
} from 'lucide-react';

export function SummaryMetrics() {
  const { predictionResult, isPredicting, selectedTripId } = useApp();

  if (isPredicting) {
    return (
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        {[...Array(6)].map((_, i) => (
          <div
            key={i}
            className="p-3.5 rounded-xl border border-slate-800 bg-slate-900/60 animate-pulse space-y-2"
          >
            <div className="h-3 w-16 bg-slate-800 rounded"></div>
            <div className="h-6 w-20 bg-slate-800/80 rounded"></div>
            <div className="h-2.5 w-12 bg-slate-800/40 rounded"></div>
          </div>
        ))}
      </div>
    );
  }

  const metrics = predictionResult?.metrics;
  const timing = predictionResult?.timing;
  const stats = predictionResult?.candidate_stats;

  const METRIC_CARDS = [
    {
      title: metrics?.top1_accuracy != null ? 'Top-1 Accuracy' : 'Route Arc Precision',
      value: metrics?.top1_accuracy != null
        ? `${(metrics.top1_accuracy * 100).toFixed(2)}%`
        : metrics?.route_arc_precision != null
          ? `${(metrics.route_arc_precision * 100).toFixed(2)}%`
          : '—',
      sub: metrics?.top1_accuracy != null
        ? metrics.top1_accuracy_conditional != null
          ? `Cond: ${(metrics.top1_accuracy_conditional * 100).toFixed(1)}%`
          : 'Per-observation score'
        : 'Predicted arcs in reference route',
      icon: Target,
      accent: 'text-cyan-400',
      border: 'border-cyan-500/20',
      bg: 'bg-cyan-500/10',
    },
    {
      title: metrics?.macro_f1 != null ? 'Macro F1-Score' : 'Route Arc F1',
      value: metrics?.macro_f1 != null
        ? `${(metrics.macro_f1 * 100).toFixed(2)}%`
        : metrics?.route_arc_f1 != null
          ? `${(metrics.route_arc_f1 * 100).toFixed(2)}%`
          : '—',
      sub: metrics?.macro_f1 != null
        ? metrics.precision != null && metrics.recall != null
          ? `P: ${(metrics.precision * 100).toFixed(1)}% | R: ${(metrics.recall * 100).toFixed(1)}%`
          : 'Per-observation score'
        : metrics?.route_arc_recall != null
          ? `Route arc recall: ${(metrics.route_arc_recall * 100).toFixed(1)}%`
          : 'Measured against reference route',
      icon: Percent,
      accent: 'text-blue-400',
      border: 'border-blue-500/20',
      bg: 'bg-blue-500/10',
    },
    {
      title: 'Route Continuity',
      value: metrics?.route_continuity != null ? `${(metrics.route_continuity * 100).toFixed(2)}%` : '—',
      sub: 'Topological validity',
      icon: GitCommit,
      accent: 'text-emerald-400',
      border: 'border-emerald-500/20',
      bg: 'bg-emerald-500/10',
    },
    {
      title: 'No-Candidate Rate',
      value: metrics?.no_candidate_rate != null ? `${(metrics.no_candidate_rate * 100).toFixed(2)}%` : '—',
      sub: stats ? `${stats.candidate_misses} points without road candidates` : 'Spatial coverage',
      icon: CheckCircle,
      accent: 'text-amber-400',
      border: 'border-amber-500/20',
      bg: 'bg-amber-500/10',
    },
    {
      title: 'Inference Latency',
      value: timing ? `${timing.total_pipeline_ms.toFixed(1)}ms` : '—',
      sub: timing?.viterbi_ms ? `Viterbi: ${timing.viterbi_ms.toFixed(1)}ms` : 'End-to-end server',
      icon: Clock,
      accent: 'text-purple-400',
      border: 'border-purple-500/20',
      bg: 'bg-purple-500/10',
    },
    {
      title: 'Matched Segments',
      value: predictionResult?.route ? `${predictionResult.route.num_matched_segments}` : '—',
      sub: stats ? `From ${stats.total_observations} obs` : 'Reconstructed path',
      icon: Layers,
      accent: 'text-teal-400',
      border: 'border-teal-500/20',
      bg: 'bg-teal-500/10',
    },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
      {METRIC_CARDS.map((card, i) => {
        const Icon = card.icon;
        return (
          <div
            key={i}
            className={`p-3.5 rounded-xl border border-slate-800 bg-slate-900/60 backdrop-blur-sm relative overflow-hidden group hover:border-slate-700 transition-all`}
          >
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-medium text-slate-400 truncate">
                {card.title}
              </span>
              <div className={`p-1 rounded-md ${card.bg} ${card.border} border ${card.accent}`}>
                <Icon className="w-3.5 h-3.5" />
              </div>
            </div>

            <div className="mt-2 font-mono text-xl font-bold text-white tracking-tight">
              {card.value}
            </div>

            <div className="mt-1 text-[10px] text-slate-500 truncate font-mono">
              {card.sub}
            </div>
          </div>
        );
      })}
    </div>
  );
}

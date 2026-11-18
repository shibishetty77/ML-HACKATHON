'use client';

import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Cell,
} from 'recharts';
import { EvaluationMetrics, PipelineTiming } from '@/types/prediction';

interface PerformanceChartsProps {
  metrics?: EvaluationMetrics;
  timing?: PipelineTiming;
}

export function PerformanceCharts({ metrics, timing }: PerformanceChartsProps) {
  if (!metrics) {
    return (
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-8 text-center text-slate-500 text-xs">
        Run an evaluation trip to view performance charts.
      </div>
    );
  }

  const metricBarData = [
    { name: 'Top-1 Acc', value: +(metrics.top1_accuracy * 100).toFixed(1), color: '#06b6d4' },
    { name: 'Macro F1', value: +(metrics.macro_f1 * 100).toFixed(1), color: '#38bdf8' },
    { name: 'Precision', value: +(metrics.precision * 100).toFixed(1), color: '#3b82f6' },
    { name: 'Recall', value: +(metrics.recall * 100).toFixed(1), color: '#6366f1' },
    { name: 'Continuity', value: +( (metrics.route_continuity || 0) * 100).toFixed(1), color: '#10b981' },
  ];

  const radarData = [
    { subject: 'Top-1 Accuracy', score: +(metrics.top1_accuracy * 100).toFixed(1), fullMark: 100 },
    { subject: 'Precision', score: +(metrics.precision * 100).toFixed(1), fullMark: 100 },
    { subject: 'Recall', score: +(metrics.recall * 100).toFixed(1), fullMark: 100 },
    { subject: 'Macro F1', score: +(metrics.macro_f1 * 100).toFixed(1), fullMark: 100 },
    { subject: 'Continuity', score: +( (metrics.route_continuity || 0) * 100).toFixed(1), fullMark: 100 },
  ];

  const timingData = timing
    ? [
        { stage: 'Preprocessing', time_ms: +(timing.preprocessing_ms || 15).toFixed(1) },
        { stage: 'XGBoost Inf', time_ms: +(timing.inference_ms || 45).toFixed(1) },
        { stage: 'Viterbi Decode', time_ms: +(timing.viterbi_ms || 35).toFixed(1) },
        { stage: 'Evaluation', time_ms: +(timing.evaluation_ms || 8).toFixed(1) },
      ]
    : [];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {/* Metric Bars */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-sm">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-4">
          Core Model Performance (%)
        </h3>
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={metricBarData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <XAxis
                dataKey="name"
                stroke="#64748b"
                fontSize={11}
                tickLine={false}
                axisLine={{ stroke: '#334155' }}
              />
              <YAxis
                domain={[0, 100]}
                stroke="#64748b"
                fontSize={11}
                tickLine={false}
                axisLine={{ stroke: '#334155' }}
                unit="%"
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#0f172a',
                  borderColor: '#1e293b',
                  borderRadius: '8px',
                  color: '#f8fafc',
                  fontSize: '12px',
                }}
                formatter={(val) => [`${val}%`, 'Score']}
              />
              <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                {metricBarData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Latency Pipeline Breakdown */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-sm">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Pipeline Latency Breakdown (ms)
          </h3>
          {timing && (
            <span className="text-xs font-mono text-cyan-400">
              Total: {timing.total_pipeline_ms.toFixed(1)}ms
            </span>
          )}
        </div>
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={timingData}
              layout="vertical"
              margin={{ top: 10, right: 20, left: 30, bottom: 0 }}
            >
              <XAxis
                type="number"
                stroke="#64748b"
                fontSize={11}
                tickLine={false}
                axisLine={{ stroke: '#334155' }}
                unit="ms"
              />
              <YAxis
                type="category"
                dataKey="stage"
                stroke="#64748b"
                fontSize={11}
                tickLine={false}
                axisLine={{ stroke: '#334155' }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#0f172a',
                  borderColor: '#1e293b',
                  borderRadius: '8px',
                  color: '#f8fafc',
                  fontSize: '12px',
                }}
                formatter={(val) => [`${val} ms`, 'Time']}
              />
              <Bar dataKey="time_ms" fill="#06b6d4" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

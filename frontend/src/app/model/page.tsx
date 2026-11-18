'use client';

import React, { useEffect, useState } from 'react';
import { fetchModelInfo } from '@/services/prediction.service';
import { ModelInfoResponse } from '@/types/prediction';
import {
  Cpu,
  Layers,
  GitBranch,
  Sliders,
} from 'lucide-react';

export default function ModelInfoPage() {
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchModelInfo()
      .then((data) => setModelInfo(data))
      .catch(() => setError('Model information is unavailable until the backend is connected.'))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-12">
      {/* Header */}
      <div className="flex items-center gap-3 p-5 rounded-xl border border-slate-800 bg-slate-900/60 backdrop-blur-sm">
        <div className="p-2.5 rounded-lg bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
          <Cpu className="w-5 h-5" />
        </div>
        <div>
          <h1 className="text-base font-bold text-white flex items-center gap-2">
            Model Architecture
          </h1>
          <p className="text-xs text-slate-400">
            Selected Champion Model: <span className="text-cyan-400 font-semibold">XGBoost Classifier + HMM Viterbi Decoder</span>
          </p>
        </div>
      </div>

      {loading && <div className="text-xs text-slate-400">Loading model details...</div>}
      {error && <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 p-4 text-sm text-rose-300">{error}</div>}

      {/* Model Spec Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Basic Info */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 space-y-3">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
            <Cpu className="w-3.5 h-3.5 text-cyan-400" />
            <span>Algorithm Specification</span>
          </h3>
          <div className="space-y-2 text-xs divide-y divide-slate-800/80">
            <div className="flex justify-between py-1 text-slate-300">
              <span>Model Family:</span>
              <span className="font-semibold text-white">{modelInfo?.model_family ?? '—'}</span>
            </div>
            <div className="flex justify-between py-1 text-slate-300">
              <span>Classifier:</span>
              <span className="font-mono text-cyan-400 font-bold">{modelInfo?.algorithm ?? '—'}</span>
            </div>
            <div className="flex justify-between py-1 text-slate-300">
              <span>Engine Framework:</span>
              <span className="font-mono text-slate-300">{modelInfo?.framework ?? '—'}</span>
            </div>
            <div className="flex justify-between py-1 text-slate-300">
              <span>Feature Columns:</span>
              <span className="font-mono text-emerald-400 font-bold">{modelInfo?.feature_count ?? '—'} features</span>
            </div>
          </div>
        </div>

        {/* Feature Columns */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 space-y-3">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
            <Layers className="w-3.5 h-3.5 text-blue-400" />
            <span>Agreed Feature Vector</span>
          </h3>
          <div className="space-y-1.5">
            {modelInfo?.feature_names.map((feat, i) => (
              <div
                key={i}
                className="flex items-center gap-2 px-2.5 py-1.5 rounded bg-slate-950 border border-slate-800 text-xs font-mono text-slate-300"
              >
                <span className="text-[10px] text-slate-500 font-semibold">#{i + 1}</span>
                <span className="text-cyan-400 font-medium">{feat}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Decoder Architecture */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 space-y-3">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
            <GitBranch className="w-3.5 h-3.5 text-emerald-400" />
            <span>Sequential Viterbi Decoder</span>
          </h3>
          <div className="space-y-2 text-xs">
            <div>
              <div className="text-[10px] text-slate-500 uppercase tracking-wider">Decoder Type</div>
              <div className="font-medium text-white">{modelInfo?.decoder.type}</div>
            </div>
            <div>
              <div className="text-[10px] text-slate-500 uppercase tracking-wider">Emission Probability</div>
              <div className="text-slate-300">{modelInfo?.decoder.emission_source}</div>
            </div>
            <div>
              <div className="text-[10px] text-slate-500 uppercase tracking-wider">Transition Matrix Penalty</div>
              <div className="text-slate-300">{modelInfo?.decoder.transition_metric}</div>
            </div>
          </div>
        </div>
      </div>

      {/* Hyperparameters Card */}
      {modelInfo?.hyperparameters && (
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 space-y-3">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
            <Sliders className="w-3.5 h-3.5 text-purple-400" />
            <span>XGBoost Hyperparameter Configuration</span>
          </h3>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
            {Object.entries(modelInfo.hyperparameters).map(([key, val]) => (
              <div key={key} className="p-2.5 rounded-lg border border-slate-800/80 bg-slate-950/60">
                <div className="text-[10px] text-slate-500 truncate">{key}</div>
                <div className="font-bold text-white mt-0.5 truncate">{String(val)}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 text-xs text-slate-400">
        Five-model comparison results require the team&apos;s training artifacts and measured experiment logs.
        The available model archive verifies the tuned XGBoost model only.
      </div>
    </div>
  );
}

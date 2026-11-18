'use client';

import React, { useState } from 'react';
import { useApp } from '@/context/AppContext';
import {
  Wifi,
  RefreshCw,
  Play,
  Settings2,
  CheckCircle2,
  AlertTriangle,
  Radio,
} from 'lucide-react';
import { ConnectionModal } from './ConnectionModal';

export function TopNavbar() {
  const {
    serverStatus,
    serverLatencyMs,
    serverBaseUrl,
    selectedTripId,
    isPredicting,
    executePrediction,
    retryConnection,
  } = useApp();

  const [isModalOpen, setIsModalOpen] = useState(false);

  const handleQuickRun = () => {
    if (selectedTripId && !isPredicting) {
      executePrediction(selectedTripId);
    }
  };

  return (
    <>
      <header className="h-14 border-b border-slate-800 bg-slate-950/80 backdrop-blur px-6 flex items-center justify-between sticky top-0 z-40">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-xs font-medium text-slate-400">
            <span>Role:</span>
            <span className="px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-200 font-mono">
              Laptop 2 — Evaluator Client
            </span>
          </div>

          {selectedTripId && (
            <div className="flex items-center gap-2 text-xs font-medium">
              <span className="text-slate-500">|</span>
              <span className="text-slate-400">Selected Trip:</span>
              <span className="font-mono text-cyan-400 font-bold bg-cyan-950/60 px-2 py-0.5 rounded border border-cyan-800/40">
                {selectedTripId}
              </span>
            </div>
          )}
        </div>

        <div className="flex items-center gap-3">
          {/* Run Prediction Button */}
          {selectedTripId && (
            <button
              onClick={handleQuickRun}
              disabled={isPredicting || serverStatus !== 'connected'}
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold bg-cyan-500 hover:bg-cyan-400 text-black transition-all shadow-md shadow-cyan-950/40 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {isPredicting ? (
                <>
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  <span>Processing...</span>
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5 fill-black" />
                  <span>Run Inference</span>
                </>
              )}
            </button>
          )}

          {/* Connection Status Pill */}
          <button
            onClick={() => setIsModalOpen(true)}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border text-xs font-mono transition-all hover:bg-slate-900 ${
              serverStatus === 'connected'
                ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-400 hover:border-emerald-500/50'
                : serverStatus === 'checking'
                ? 'border-amber-500/30 bg-amber-500/10 text-amber-400 hover:border-amber-500/50'
                : 'border-rose-500/30 bg-rose-500/10 text-rose-400 hover:border-rose-500/50'
            }`}
            title="Click to view network connection settings"
          >
            <span className="relative flex h-2 w-2">
              {serverStatus === 'connected' && (
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              )}
              <span
                className={`relative inline-flex rounded-full h-2 w-2 ${
                  serverStatus === 'connected'
                    ? 'bg-emerald-400'
                    : serverStatus === 'checking'
                    ? 'bg-amber-400'
                    : 'bg-rose-400'
                }`}
              ></span>
            </span>

            <span className="font-sans font-medium text-slate-300 truncate max-w-[140px]">
              {serverBaseUrl.replace(/^http:\/\//, '')}
            </span>

            {serverLatencyMs !== null && (
              <span className="text-[10px] text-slate-400">({serverLatencyMs}ms)</span>
            )}

            <Settings2 className="w-3.5 h-3.5 text-slate-400 ml-1 hover:text-white" />
          </button>
        </div>
      </header>

      {isModalOpen && <ConnectionModal isOpen={true} onClose={() => setIsModalOpen(false)} />}
    </>
  );
}

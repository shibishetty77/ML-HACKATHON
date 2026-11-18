'use client';

import React from 'react';
import { useApp } from '@/context/AppContext';
import {
  Server,
  Cpu,
  Wifi,
  Activity,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  GitBranch,
} from 'lucide-react';

export function SystemStatusCard() {
  const {
    serverStatus,
    serverLatencyMs,
    serverHealth,
    serverBaseUrl,
    retryConnection,
  } = useApp();

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-sm">
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-blue-500/10 border border-blue-500/20 text-blue-400">
            <Server className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-white">Laptop 1 ML Server Health</h3>
            <p className="text-[11px] text-slate-400">
              Host: <span className="font-mono text-cyan-400">{serverBaseUrl}</span>
            </p>
          </div>
        </div>

        <button
          onClick={() => retryConnection()}
          className="p-1.5 rounded-lg border border-slate-800 bg-slate-950 text-slate-400 hover:text-white transition-colors"
          title="Re-check server health"
        >
          <RefreshCw className="w-3.5 h-3.5" />
        </button>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4">
        {/* Network Status */}
        <div className="p-3 rounded-lg border border-slate-800/80 bg-slate-950/60">
          <div className="text-[10px] text-slate-400 uppercase tracking-wider font-medium flex items-center gap-1.5">
            <Wifi className="w-3 h-3 text-cyan-400" />
            <span>Connection</span>
          </div>
          <div className="mt-1 flex items-center gap-1.5 font-medium text-sm capitalize">
            {serverStatus === 'connected' ? (
              <>
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                <span className="text-emerald-400">Online</span>
              </>
            ) : serverStatus === 'checking' ? (
              <>
                <RefreshCw className="w-4 h-4 text-amber-400 animate-spin" />
                <span className="text-amber-400">Checking</span>
              </>
            ) : (
              <>
                <AlertTriangle className="w-4 h-4 text-rose-400" />
                <span className="text-rose-400">Offline</span>
              </>
            )}
          </div>
          <div className="text-[10px] text-slate-500 font-mono mt-0.5">
            RTT: {serverLatencyMs !== null ? `${serverLatencyMs}ms` : 'N/A'}
          </div>
        </div>

        {/* Model State */}
        <div className="p-3 rounded-lg border border-slate-800/80 bg-slate-950/60">
          <div className="text-[10px] text-slate-400 uppercase tracking-wider font-medium flex items-center gap-1.5">
            <Cpu className="w-3 h-3 text-blue-400" />
            <span>ML Classifier</span>
          </div>
          <div className="mt-1 font-mono text-sm font-bold text-white">
            {serverHealth?.model_algorithm || 'XGBoost'}
          </div>
          <div className="text-[10px] text-emerald-400 font-mono mt-0.5">
            {serverHealth?.model_loaded ? '● Weights Loaded' : 'Waiting for server'}
          </div>
        </div>

        {/* Decoder State */}
        <div className="p-3 rounded-lg border border-slate-800/80 bg-slate-950/60">
          <div className="text-[10px] text-slate-400 uppercase tracking-wider font-medium flex items-center gap-1.5">
            <GitBranch className="w-3 h-3 text-emerald-400" />
            <span>Route Decoder</span>
          </div>
          <div className="mt-1 font-mono text-sm font-bold text-white">
            HMM + Viterbi
          </div>
          <div className="text-[10px] text-emerald-400 font-mono mt-0.5">
            Graph Adjacency Active
          </div>
        </div>

        {/* Evaluator Scope */}
        <div className="p-3 rounded-lg border border-slate-800/80 bg-slate-950/60">
          <div className="text-[10px] text-slate-400 uppercase tracking-wider font-medium flex items-center gap-1.5">
            <Activity className="w-3 h-3 text-amber-400" />
            <span>Evaluator Trips</span>
          </div>
          <div className="mt-1 font-mono text-sm font-bold text-cyan-400">
            {serverHealth?.evaluator_trips_available ?? 10} Trips
          </div>
          <div className="text-[10px] text-slate-500 font-mono mt-0.5">
            Server Ground-Truth
          </div>
        </div>
      </div>
    </div>
  );
}

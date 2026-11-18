'use client';

import React, { useState } from 'react';
import { useApp } from '@/context/AppContext';
import { X, Wifi, RefreshCw, CheckCircle2, AlertTriangle, Laptop, ArrowRight } from 'lucide-react';

interface ConnectionModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function ConnectionModal({ isOpen, onClose }: ConnectionModalProps) {
  const { serverStatus, serverLatencyMs, serverHealth, serverBaseUrl, updateServerUrl, retryConnection } =
    useApp();

  const [inputUrl, setInputUrl] = useState(serverBaseUrl);
  const [testing, setTesting] = useState(false);

  if (!isOpen) return null;

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    setTesting(true);
    updateServerUrl(inputUrl);
    await retryConnection();
    setTesting(false);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4">
      <div className="relative w-full max-w-lg rounded-xl border border-slate-800 bg-slate-900 shadow-2xl p-6">
        <button
          onClick={onClose}
          className="absolute right-4 top-4 text-slate-400 hover:text-white transition-colors"
          aria-label="Close modal"
        >
          <X className="w-5 h-5" />
        </button>

        <div className="flex items-center gap-3 mb-5">
          <div className="p-2 rounded-lg bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
            <Wifi className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-white">Laptop 1 — ML Server Connection</h2>
            <p className="text-xs text-slate-400">Two-Laptop Local Area Network Configuration</p>
          </div>
        </div>

        {/* Current status banner */}
        <div
          className={`flex items-center justify-between p-3 rounded-lg border mb-5 ${
            serverStatus === 'connected'
              ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
              : serverStatus === 'checking'
              ? 'bg-amber-500/10 border-amber-500/30 text-amber-400'
              : 'bg-rose-500/10 border-rose-500/30 text-rose-400'
          }`}
        >
          <div className="flex items-center gap-2 text-sm font-medium">
            {serverStatus === 'connected' ? (
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
            ) : (
              <AlertTriangle className="w-4 h-4" />
            )}
            <span className="capitalize">Status: {serverStatus}</span>
          </div>
          {serverLatencyMs !== null && (
            <span className="text-xs px-2 py-0.5 rounded bg-slate-800/80 text-slate-300 font-mono">
              {serverLatencyMs} ms latency
            </span>
          )}
        </div>

        <form onSubmit={handleSave} className="space-y-4">
          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1">
              Laptop 1 FastAPI Base URL (Listening on 0.0.0.0:8000)
            </label>
            <div className="relative">
              <input
                type="text"
                value={inputUrl}
                onChange={(e) => setInputUrl(e.target.value)}
                placeholder="http://192.168.43.120:8000"
                className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white font-mono placeholder:text-slate-600 focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500"
              />
            </div>
            <p className="text-[11px] text-slate-400 mt-1">
              Find Laptop 1&apos;s IP via: <code className="text-cyan-400">ipconfig getifaddr en0</code> (macOS) or{' '}
              <code className="text-cyan-400">hostname -I</code> (Linux).
            </p>
          </div>

          {serverHealth && (
            <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-3 space-y-1.5 text-xs">
              <div className="text-slate-400 font-semibold uppercase tracking-wider text-[10px]">
                Active Backend Details
              </div>
              <div className="flex justify-between text-slate-300">
                <span>Model Algorithm:</span>
                <span className="font-mono text-cyan-400 font-medium">{serverHealth.model_algorithm}</span>
              </div>
              <div className="flex justify-between text-slate-300">
                <span>Evaluator Trips Available:</span>
                <span className="font-mono text-white">{serverHealth.evaluator_trips_available} trips</span>
              </div>
              <div className="flex justify-between text-slate-300">
                <span>HMM/Viterbi Route Decoder:</span>
                <span className="text-emerald-400 font-medium">Ready</span>
              </div>
            </div>
          )}

          <div className="flex items-center justify-end gap-3 pt-2">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-xs font-medium text-slate-300 hover:text-white rounded-lg border border-slate-800 bg-slate-800/40 hover:bg-slate-800 transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={testing}
              className="flex items-center gap-2 px-4 py-2 text-xs font-medium text-black bg-cyan-400 hover:bg-cyan-300 rounded-lg transition-colors font-semibold disabled:opacity-50"
            >
              {testing ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <RefreshCw className="w-3.5 h-3.5" />}
              Save & Test Connection
            </button>
          </div>
        </form>

        <div className="mt-5 pt-4 border-t border-slate-800 text-[11px] text-slate-400 flex items-start gap-2">
          <Laptop className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
          <span>
            Both laptops must connect to the same mobile hotspot. The server on Laptop 1 must be started with{' '}
            <code className="text-slate-300 font-mono">--host 0.0.0.0 --port 8000</code>.
          </span>
        </div>
      </div>
    </div>
  );
}

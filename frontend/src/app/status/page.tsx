'use client';

import React, { useState } from 'react';
import { useApp } from '@/context/AppContext';
import {
  Activity,
  Server,
  Wifi,
  RefreshCw,
  CheckCircle2,
  AlertTriangle,
  Clock,
  Terminal,
  Laptop,
  Check,
  Send,
} from 'lucide-react';
import { checkServerHealth } from '@/services/health.service';
import { fetchModelInfo } from '@/services/prediction.service';

export default function SystemStatusPage() {
  const {
    serverStatus,
    serverLatencyMs,
    serverHealth,
    serverBaseUrl,
    updateServerUrl,
    retryConnection,
  } = useApp();

  const [inputUrl, setInputUrl] = useState(serverBaseUrl);
  const [pingLog, setPingLog] = useState<string[]>([]);
  const [pinging, setPinging] = useState(false);

  const runDiagnostics = async () => {
    setPinging(true);
    const logs: string[] = [];
    const timestamp = new Date().toLocaleTimeString();

    logs.push(`[${timestamp}] Initiating diagnostics to ${serverBaseUrl}...`);

    try {
      const start = performance.now();
      const res = await checkServerHealth(5000);
      const elapsed = Math.round(performance.now() - start);

      logs.push(`[${timestamp}] ✓ Health check responded in ${elapsed}ms: ${res.health.status.toUpperCase()}`);
      logs.push(`[${timestamp}]   - Model Algorithm: ${res.health.model_algorithm}`);
      logs.push(`[${timestamp}]   - Features Loaded: ${res.health.features_loaded}`);
      logs.push(`[${timestamp}]   - HMM Decoder: ${res.health.hmm_ready ? 'READY' : 'NOT READY'}`);
      logs.push(`[${timestamp}]   - Available Trips: ${res.health.evaluator_trips_available}`);

      // Probe model info
      try {
        const mInfo = await fetchModelInfo();
        logs.push(`[${timestamp}] ✓ /api/v1/model/info responded: ${mInfo.algorithm} ${mInfo.version}`);
      } catch {
        logs.push(`[${timestamp}] ⚠ /api/v1/model/info unavailable or returned error.`);
      }

      await retryConnection();
    } catch (err: unknown) {
      logs.push(`[${timestamp}] ✗ FAILED to connect to ${serverBaseUrl}:`);
      const msg =
        err && typeof err === 'object' && 'message' in err
          ? String((err as { message: unknown }).message)
          : 'Network timeout';
      logs.push(`[${timestamp}]   Error: ${msg}`);
      logs.push(`[${timestamp}]   Tip: Ensure Laptop 1 has firewall ports open and is bound to 0.0.0.0.`);
    }

    setPingLog((prev) => [...logs, ...prev].slice(0, 30));
    setPinging(false);
  };

  const handleUpdate = (e: React.FormEvent) => {
    e.preventDefault();
    updateServerUrl(inputUrl);
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-12">
      {/* Header */}
      <div className="flex items-center justify-between p-5 rounded-xl border border-slate-800 bg-slate-900/60 backdrop-blur-sm">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
            <Activity className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-base font-bold text-white flex items-center gap-2">
              System Status & Network Diagnostics
            </h1>
            <p className="text-xs text-slate-400">
              Live monitor for Laptop 2 (Client) ↔ Laptop 1 (ML Server) connectivity
            </p>
          </div>
        </div>

        <button
          onClick={runDiagnostics}
          disabled={pinging}
          className="flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-cyan-400 hover:bg-cyan-300 text-black transition-all disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${pinging ? 'animate-spin' : ''}`} />
          <span>Run Probe Test</span>
        </button>
      </div>

      {/* Connection Config Bar */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 space-y-4">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
          <Wifi className="w-3.5 h-3.5 text-cyan-400" />
          <span>Laptop 1 Host Configuration</span>
        </h3>

        <form onSubmit={handleUpdate} className="flex flex-col sm:flex-row gap-3">
          <div className="flex-1">
            <input
              type="text"
              value={inputUrl}
              onChange={(e) => setInputUrl(e.target.value)}
              placeholder="http://192.168.43.120:8000"
              className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3.5 py-2 text-xs font-mono text-white placeholder:text-slate-600 focus:outline-none focus:border-cyan-500"
            />
          </div>
          <button
            type="submit"
            className="px-4 py-2 rounded-lg text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-white border border-slate-700 flex items-center justify-center gap-1.5"
          >
            <Check className="w-3.5 h-3.5 text-cyan-400" />
            <span>Apply URL</span>
          </button>
        </form>

        <div className="text-[11px] text-slate-400">
          Current Active Target:{' '}
          <code className="text-cyan-400 font-mono font-semibold">{serverBaseUrl}</code>
        </div>
      </div>

      {/* Diagnostics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 space-y-2">
          <div className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">
            Connection State
          </div>
          <div className="flex items-center gap-2 text-base font-bold">
            {serverStatus === 'connected' ? (
              <>
                <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                <span className="text-emerald-400 capitalize">Online</span>
              </>
            ) : serverStatus === 'checking' ? (
              <>
                <RefreshCw className="w-5 h-5 text-amber-400 animate-spin" />
                <span className="text-amber-400 capitalize">Probing...</span>
              </>
            ) : (
              <>
                <AlertTriangle className="w-5 h-5 text-rose-400" />
                <span className="text-rose-400 capitalize">Disconnected</span>
              </>
            )}
          </div>
          <p className="text-xs text-slate-400">
            {serverStatus === 'connected'
              ? 'FastAPI server is responding normally to REST requests.'
              : 'Server not reachable. Check mobile hotspot IP and port.'}
          </p>
        </div>

        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 space-y-2">
          <div className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">
            Round-Trip Latency
          </div>
          <div className="font-mono text-2xl font-bold text-white">
            {serverLatencyMs !== null ? `${serverLatencyMs} ms` : '—'}
          </div>
          <p className="text-xs text-slate-400">
            Local hotspot Wi-Fi response time for health check probe.
          </p>
        </div>

        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 space-y-2">
          <div className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">
            Server Model Readiness
          </div>
          <div className="font-mono text-base font-bold text-cyan-400">
            {serverHealth?.model_algorithm || 'XGBoost'}
          </div>
          <div className="text-xs text-emerald-400 font-medium">
            {serverHealth?.model_loaded ? 'Ready for Evaluator Inference' : 'Not Loaded'}
          </div>
        </div>
      </div>

      {/* Live Probe Console */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 space-y-3">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
          <Terminal className="w-3.5 h-3.5 text-cyan-400" />
          <span>Diagnostic Probe Output Console</span>
        </h3>

        <div className="h-44 w-full rounded-lg border border-slate-800 bg-slate-950 p-3 font-mono text-[11px] text-slate-300 overflow-y-auto space-y-1">
          {pingLog.length === 0 ? (
            <div className="text-slate-600 italic">
              Click &quot;Run Probe Test&quot; above to trace connectivity to Laptop 1...
            </div>
          ) : (
            pingLog.map((log, i) => (
              <div key={i} className="leading-relaxed">
                {log}
              </div>
            ))
          )}
        </div>
      </div>

      {/* Two-Laptop Hotspot Checklist */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 space-y-3">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
          <Laptop className="w-3.5 h-3.5 text-cyan-400" />
          <span>Two-Laptop Demonstration Checklist</span>
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs text-slate-300">
          <div className="p-3 rounded-lg border border-slate-800/80 bg-slate-950/60 space-y-1">
            <div className="font-bold text-white flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-cyan-400"></span>
              Laptop 1 (Teammate ML Server)
            </div>
            <ul className="list-disc list-inside space-y-0.5 text-slate-400 text-[11px]">
              <li>Connected to mobile hotspot</li>
              <li>IP noted: <code className="text-cyan-400 font-mono">ipconfig getifaddr en0</code></li>
              <li>Started server: <code className="text-slate-300 font-mono">--host 0.0.0.0 --port 8000</code></li>
              <li>Firewall allows incoming port 8000</li>
            </ul>
          </div>

          <div className="p-3 rounded-lg border border-slate-800/80 bg-slate-950/60 space-y-1">
            <div className="font-bold text-white flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
              Laptop 2 (Evaluator UI Client)
            </div>
            <ul className="list-disc list-inside space-y-0.5 text-slate-400 text-[11px]">
              <li>Connected to same mobile hotspot</li>
              <li>Target URL configured in top navbar</li>
              <li>Probe returns HTTP 200 OK</li>
              <li>Evaluator can choose from 10 trips</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

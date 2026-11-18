'use client';

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  LayoutDashboard,
  MapPin,
  BarChart3,
  Cpu,
  Activity,
  Layers,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
} from 'lucide-react';
import { useApp } from '@/context/AppContext';

const NAV_ITEMS = [
  { label: 'Overview', href: '/', icon: LayoutDashboard },
  { label: 'Route Explorer', href: '/explorer', icon: MapPin },
  { label: 'Prediction Results', href: '/results', icon: BarChart3 },
  { label: 'Model Information', href: '/model', icon: Cpu },
  { label: 'System Status', href: '/status', icon: Activity },
];

export function Sidebar() {
  const pathname = usePathname();
  const { serverStatus, serverLatencyMs, selectedTripId, isPredicting } = useApp();

  return (
    <aside className="w-64 border-r border-slate-800 bg-slate-950 flex flex-col shrink-0 min-h-screen">
      {/* Brand Header */}
      <div className="p-5 border-b border-slate-800">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-lg bg-gradient-to-tr from-cyan-600 to-blue-600 text-white shadow-lg shadow-cyan-900/30">
            <Layers className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-base font-bold tracking-wider text-white flex items-center gap-1.5">
              PATHSNAP
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-cyan-950 text-cyan-400 border border-cyan-800/50">
                v1.0
              </span>
            </h1>
            <p className="text-[10px] text-slate-400 font-medium tracking-tight">
              Distributed Map-Matching
            </p>
          </div>
        </div>
      </div>

      {/* Selected Trip Indicator if active */}
      {selectedTripId && (
        <div className="mx-3 mt-3 p-2.5 rounded-lg border border-slate-800/80 bg-slate-900/60">
          <div className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">
            Active Evaluator Trip
          </div>
          <div className="flex items-center justify-between mt-1">
            <span className="text-sm font-mono font-bold text-cyan-400">
              Trip #{selectedTripId}
            </span>
            {isPredicting && (
              <span className="flex items-center gap-1 text-[11px] text-amber-400 font-mono">
                <RefreshCw className="w-3 h-3 animate-spin" />
                Inferring...
              </span>
            )}
          </div>
        </div>
      )}

      {/* Navigation */}
      <nav className="flex-1 p-3 space-y-1 mt-1">
        {NAV_ITEMS.map((item) => {
          const isActive = pathname === item.href;
          const Icon = item.icon;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all ${
                isActive
                  ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 font-semibold'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
              }`}
            >
              <Icon
                className={`w-4 h-4 ${isActive ? 'text-cyan-400' : 'text-slate-500'}`}
              />
              <span>{item.label}</span>
            </Link>
          );
        })}
      </nav>

      {/* Footer Connection Pill */}
      <div className="p-3 border-t border-slate-800">
        <div className="p-3 rounded-lg bg-slate-900/80 border border-slate-800 text-xs">
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-slate-400 text-[11px] font-medium">Laptop 1 ML Server</span>
            <span
              className={`flex items-center gap-1 text-[11px] font-mono capitalize ${
                serverStatus === 'connected'
                  ? 'text-emerald-400'
                  : serverStatus === 'checking'
                  ? 'text-amber-400'
                  : 'text-rose-400'
              }`}
            >
              {serverStatus === 'connected' ? (
                <CheckCircle2 className="w-3 h-3" />
              ) : (
                <AlertCircle className="w-3 h-3" />
              )}
              {serverStatus}
            </span>
          </div>
          {serverLatencyMs !== null && (
            <div className="text-[10px] text-slate-500 font-mono">
              RTT: {serverLatencyMs}ms
            </div>
          )}
          <div className="text-[10px] text-slate-500 font-mono truncate mt-0.5">
            Port: 8000 (0.0.0.0)
          </div>
        </div>
      </div>
    </aside>
  );
}

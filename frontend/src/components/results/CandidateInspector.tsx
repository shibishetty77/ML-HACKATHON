'use client';

import React, { useState } from 'react';
import { GpsObservation } from '@/types/trip';
import { Search, ChevronLeft, ChevronRight, Navigation } from 'lucide-react';

interface CandidateInspectorProps {
  observations?: GpsObservation[];
}

export function CandidateInspector({ observations = [] }: CandidateInspectorProps) {
  const [search, setSearch] = useState('');
  const [page, setPage] = useState(1);
  const pageSize = 8;

  const filtered = observations.filter(
    (o) =>
      o.obs_id.toLowerCase().includes(search.toLowerCase()) ||
      (o.matched_arc_id !== undefined && String(o.matched_arc_id).includes(search))
  );

  const totalPages = Math.ceil(filtered.length / pageSize) || 1;
  const paginated = filtered.slice((page - 1) * pageSize, page * pageSize);

  if (observations.length === 0) {
    return null;
  }

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-sm">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-800">
        <div>
          <h3 className="text-sm font-semibold text-white flex items-center gap-2">
            Sequence Observation Inspector
            <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300">
              {observations.length} observations
            </span>
          </h3>
          <p className="text-xs text-slate-400">
            Observation-level GPS coordinates and Viterbi-decoded road segment assignments
          </p>
        </div>

        <div className="flex items-center gap-2">
          <div className="relative">
            <Search className="w-3.5 h-3.5 absolute left-2.5 top-2.5 text-slate-500" />
            <input
              type="text"
              value={search}
              onChange={(e) => {
                setSearch(e.target.value);
                setPage(1);
              }}
              placeholder="Search Obs ID / Arc..."
              className="w-44 pl-8 pr-3 py-1.5 rounded-lg border border-slate-800 bg-slate-950 text-xs text-white placeholder:text-slate-600 focus:outline-none focus:border-cyan-500 font-mono"
            />
          </div>
        </div>
      </div>

      <div className="overflow-x-auto mt-3">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="border-b border-slate-800 text-slate-400 font-medium">
              <th className="py-2.5 px-3">Observation ID</th>
              <th className="py-2.5 px-3">Latitude</th>
              <th className="py-2.5 px-3">Longitude</th>
              <th className="py-2.5 px-3">Speed (m/s)</th>
              <th className="py-2.5 px-3">Matched Arc</th>
              <th className="py-2.5 px-3">Confidence</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 font-mono text-[11px]">
            {paginated.map((obs) => (
              <tr key={obs.obs_id} className="hover:bg-slate-800/30 transition-colors">
                <td className="py-2 px-3 text-cyan-400 font-semibold">{obs.obs_id}</td>
                <td className="py-2 px-3 text-slate-300">{obs.lat.toFixed(6)}</td>
                <td className="py-2 px-3 text-slate-300">{obs.lon.toFixed(6)}</td>
                <td className="py-2 px-3 text-slate-400">
                  {obs.speed_mps !== undefined ? obs.speed_mps.toFixed(1) : '—'}
                </td>
                <td className="py-2 px-3">
                  <span className="px-2 py-0.5 rounded bg-slate-800 text-emerald-400 font-bold border border-emerald-500/20">
                    Arc #{obs.matched_arc_id ?? '—'}
                  </span>
                </td>
                <td className="py-2 px-3">
                  {obs.confidence !== undefined ? (
                    <span
                      className={`font-semibold ${
                        obs.confidence > 0.8
                          ? 'text-emerald-400'
                          : obs.confidence > 0.5
                          ? 'text-cyan-400'
                          : 'text-amber-400'
                      }`}
                    >
                      {(obs.confidence * 100).toFixed(1)}%
                    </span>
                  ) : (
                    '—'
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination controls */}
      <div className="flex items-center justify-between pt-3 mt-2 border-t border-slate-800/80 text-xs text-slate-400">
        <div>
          Showing {(page - 1) * pageSize + 1} to {Math.min(page * pageSize, filtered.length)} of{' '}
          {filtered.length}
        </div>
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page === 1}
            className="p-1 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-40 transition-colors"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
          <span className="px-2 py-0.5 font-mono text-white">
            {page} / {totalPages}
          </span>
          <button
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            disabled={page === totalPages}
            className="p-1 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-40 transition-colors"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}

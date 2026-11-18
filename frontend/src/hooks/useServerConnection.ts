'use client';

import { useState, useEffect, useCallback } from 'react';
import { checkServerHealth } from '@/services/health.service';
import { getApiBaseUrl, getDefaultApiBaseUrl, setApiBaseUrl } from '@/services/api';
import { HealthResponse } from '@/types/api';

export type ConnectionStatus = 'checking' | 'connected' | 'disconnected' | 'error';

export function useServerConnection(autoPollIntervalMs: number = 10000) {
  const [status, setStatus] = useState<ConnectionStatus>('checking');
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [healthData, setHealthData] = useState<HealthResponse | null>(null);
  const [lastChecked, setLastChecked] = useState<Date | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [baseUrl, setBaseUrlState] = useState<string>(getDefaultApiBaseUrl());

  const checkConnection = useCallback(async () => {
    setStatus((prev) => (prev === 'connected' ? 'connected' : 'checking'));
    try {
      const res = await checkServerHealth();
      setHealthData(res.health);
      setLatencyMs(res.latencyMs);
      setStatus('connected');
      setErrorMessage(null);
      setLastChecked(new Date());
    } catch (err: unknown) {
      setStatus('disconnected');
      setLatencyMs(null);
      setHealthData(null);
      setLastChecked(new Date());
      const msg =
        err && typeof err === 'object' && 'message' in err
          ? String((err as { message: unknown }).message)
          : 'Server unreachable';
      setErrorMessage(msg);
    }
  }, []);

  const updateServerUrl = useCallback(
    (newUrl: string) => {
      setApiBaseUrl(newUrl);
      setBaseUrlState(newUrl);
      setStatus('checking');
      checkConnection();
    },
    [checkConnection]
  );

  useEffect(() => {
    const handleUrlChange = () => {
      setBaseUrlState(getApiBaseUrl());
      void checkConnection();
    };

    window.addEventListener('pathsnap_api_url_changed', handleUrlChange);
    const initialCheck = window.setTimeout(handleUrlChange, 0);

    const interval = setInterval(() => {
      checkConnection();
    }, autoPollIntervalMs);

    return () => {
      window.clearTimeout(initialCheck);
      window.removeEventListener('pathsnap_api_url_changed', handleUrlChange);
      clearInterval(interval);
    };
  }, [checkConnection, autoPollIntervalMs]);

  return {
    status,
    latencyMs,
    healthData,
    lastChecked,
    errorMessage,
    baseUrl,
    updateServerUrl,
    retryConnection: checkConnection,
  };
}

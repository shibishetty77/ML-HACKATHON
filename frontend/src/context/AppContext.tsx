'use client';

import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { TripSummary } from '@/types/trip';
import { PredictionResponse } from '@/types/prediction';
import { fetchEvaluatorTrips } from '@/services/trips.service';
import { runTripPrediction } from '@/services/prediction.service';
import { useServerConnection, ConnectionStatus } from '@/hooks/useServerConnection';
import { HealthResponse } from '@/types/api';

interface AppContextType {
  // Server state
  serverStatus: ConnectionStatus;
  serverLatencyMs: number | null;
  serverHealth: HealthResponse | null;
  serverBaseUrl: string;
  updateServerUrl: (url: string) => void;
  retryConnection: () => Promise<void>;

  // Trips state
  trips: TripSummary[];
  loadingTrips: boolean;
  tripsError: string | null;
  refetchTrips: () => Promise<void>;
  selectedTripId: string | null;
  setSelectedTripId: (id: string | null) => void;

  // Prediction state
  isPredicting: boolean;
  predictError: string | null;
  predictionResult: PredictionResponse | null;
  executePrediction: (tripId: string) => Promise<PredictionResponse | null>;
  clearPrediction: () => void;
}

const AppContext = createContext<AppContextType | null>(null);

export function AppProvider({ children }: { children: React.ReactNode }) {
  const {
    status: serverStatus,
    latencyMs: serverLatencyMs,
    healthData: serverHealth,
    baseUrl: serverBaseUrl,
    updateServerUrl,
    retryConnection,
  } = useServerConnection();

  const [trips, setTrips] = useState<TripSummary[]>([]);
  const [loadingTrips, setLoadingTrips] = useState<boolean>(false);
  const [tripsError, setTripsError] = useState<string | null>(null);
  const [selectedTripId, setSelectedTripId] = useState<string | null>(null);

  const [isPredicting, setIsPredicting] = useState<boolean>(false);
  const [predictError, setPredictError] = useState<string | null>(null);
  const [predictionResult, setPredictionResult] = useState<PredictionResponse | null>(null);

  const loadTrips = useCallback(async () => {
    if (serverStatus === 'disconnected') {
      return;
    }
    setLoadingTrips(true);
    setTripsError(null);
    try {
      const res = await fetchEvaluatorTrips();
      setTrips(res.trips || []);
      if (res.trips?.length > 0 && !selectedTripId) {
        setSelectedTripId(res.trips[0].trip_id);
      }
    } catch (err: unknown) {
      const msg =
        err && typeof err === 'object' && 'message' in err
          ? String((err as { message: unknown }).message)
          : 'Failed to load evaluation trips';
      setTripsError(msg);
    } finally {
      setLoadingTrips(false);
    }
  }, [serverStatus, selectedTripId]);

  useEffect(() => {
    if (serverStatus === 'connected') {
      loadTrips();
    }
  }, [serverStatus, loadTrips]);

  const executePrediction = async (tripId: string): Promise<PredictionResponse | null> => {
    setIsPredicting(true);
    setPredictError(null);
    try {
      const result = await runTripPrediction(tripId);
      setPredictionResult(result);
      setSelectedTripId(tripId);
      return result;
    } catch (err: unknown) {
      const msg =
        err && typeof err === 'object' && 'message' in err
          ? String((err as { message: unknown }).message)
          : `Prediction failed for trip ${tripId}`;
      setPredictError(msg);
      return null;
    } finally {
      setIsPredicting(false);
    }
  };

  const clearPrediction = () => {
    setPredictionResult(null);
    setPredictError(null);
  };

  return (
    <AppContext.Provider
      value={{
        serverStatus,
        serverLatencyMs,
        serverHealth,
        serverBaseUrl,
        updateServerUrl,
        retryConnection,
        trips,
        loadingTrips,
        tripsError,
        refetchTrips: loadTrips,
        selectedTripId,
        setSelectedTripId,
        isPredicting,
        predictError,
        predictionResult,
        executePrediction,
        clearPrediction,
      }}
    >
      {children}
    </AppContext.Provider>
  );
}

export function useApp() {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
}

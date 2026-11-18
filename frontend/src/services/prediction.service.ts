import { apiClient } from './api';
import {
  PredictionRequest,
  PredictionResponse,
  ModelInfoResponse,
} from '@/types/prediction';
import { SystemMetricsResponse } from '@/types/api';

export async function runTripPrediction(
  tripId: string,
  timeoutMs: number = 60000 // Predictions with HMM / Viterbi may take several seconds on large trips
): Promise<PredictionResponse> {
  const payload: PredictionRequest = { trip_id: tripId };
  return apiClient<PredictionResponse>(
    '/api/v1/predict',
    {
      method: 'POST',
      body: JSON.stringify(payload),
    },
    timeoutMs
  );
}

export async function fetchModelInfo(): Promise<ModelInfoResponse> {
  return apiClient<ModelInfoResponse>('/api/v1/model/info', { method: 'GET' });
}

export async function fetchSystemMetrics(): Promise<SystemMetricsResponse> {
  return apiClient<SystemMetricsResponse>('/api/v1/system/metrics', {
    method: 'GET',
  });
}

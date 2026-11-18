import { apiClient } from './api';
import { TripListResponse } from '@/types/trip';

export async function fetchEvaluatorTrips(): Promise<TripListResponse> {
  return apiClient<TripListResponse>('/api/v1/trips', { method: 'GET' });
}

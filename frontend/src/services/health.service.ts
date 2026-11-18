import { apiClient } from './api';
import { HealthResponse } from '@/types/api';

export async function checkServerHealth(timeoutMs: number = 4000): Promise<{
  health: HealthResponse;
  latencyMs: number;
}> {
  const start = performance.now();
  const health = await apiClient<HealthResponse>('/health', { method: 'GET' }, timeoutMs);
  const latencyMs = Math.round(performance.now() - start);
  return { health, latencyMs };
}

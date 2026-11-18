export interface ApiResponse<T> {
  data: T;
  status: number;
  message?: string;
}

export interface ApiError {
  message: string;
  statusCode?: number;
  details?: unknown;
  isNetworkError?: boolean;
}

export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  version: string;
  server_time: string;
  model_loaded: boolean;
  model_algorithm: string;
  features_loaded: number;
  hmm_ready: boolean;
  evaluator_trips_available: number;
}

export interface SystemMetricsResponse {
  uptime_seconds: number;
  total_requests: number;
  avg_inference_latency_ms: number;
  memory_usage_mb?: number;
  cpu_percent?: number;
}

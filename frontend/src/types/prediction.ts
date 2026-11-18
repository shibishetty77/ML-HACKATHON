import { GpsObservation } from './trip';

export interface EvaluationMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  macro_f1: number;
  top1_accuracy: number;
  top1_accuracy_conditional?: number;
  candidate_miss_rate?: number;
  route_continuity?: number;
  log_loss?: number;
}

export interface PipelineTiming {
  preprocessing_ms?: number;
  inference_ms?: number;
  viterbi_ms?: number;
  evaluation_ms?: number;
  total_pipeline_ms: number;
}

export interface CandidateStats {
  total_observations: number;
  total_candidates: number;
  avg_candidates_per_obs: number;
  candidate_misses: number;
}

export interface PredictedRoute {
  num_matched_segments: number;
  matched_arc_ids: number[];
  predicted_polyline: [number, number][]; // [lat, lon] tuples
}

export interface PredictionRequest {
  trip_id: string;
}

export interface PredictionResponse {
  trip_id: string;
  status: 'success' | 'partial' | 'failed';
  timing: PipelineTiming;
  metrics: EvaluationMetrics;
  candidate_stats: CandidateStats;
  route: PredictedRoute;
  gps_observations: GpsObservation[];
  error_message?: string;
}

export interface ModelInfoResponse {
  model_family: string;
  algorithm: string;
  version: string;
  framework: string;
  feature_count: number;
  feature_names: string[];
  hyperparameters: Record<string, string | number | boolean>;
  decoder: {
    type: string;
    emission_source: string;
    transition_metric: string;
  };
}

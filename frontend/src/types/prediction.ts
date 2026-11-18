import { GpsObservation } from './trip';

export interface EvaluationMetrics {
  accuracy: number | null;
  precision: number | null;
  recall: number | null;
  macro_f1: number | null;
  top1_accuracy: number | null;
  top1_accuracy_conditional?: number | null;
  candidate_miss_rate?: number | null;
  route_continuity?: number | null;
  log_loss?: number | null;
  route_arc_precision?: number;
  route_arc_recall?: number;
  route_arc_f1?: number;
  no_candidate_rate?: number;
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
  polylines?: [number, number][][];
  arc_chunks?: number[][];
  unresolved_steps?: number;
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

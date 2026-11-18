export interface GeoBounds {
  min_lat: number;
  max_lat: number;
  min_lon: number;
  max_lon: number;
}

export interface TripSummary {
  trip_id: string;
  status: 'available' | 'processing' | 'completed' | 'failed';
  num_gps_points: number;
  approx_duration_s: number;
  bounds?: GeoBounds;
  start_time?: string;
  end_time?: string;
}

export interface TripListResponse {
  trips: TripSummary[];
  total: number;
}

export interface GpsObservation {
  obs_id: string;
  timestamp: number;
  lat: number;
  lon: number;
  speed_mps?: number;
  matched_arc_id?: number | null;
  confidence?: number | null;
}

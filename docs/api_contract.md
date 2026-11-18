# PathSnap — API Integration Contract & Schema Specification

**Version:** 1.0.0  
**Authors:** PathSnap Engineering Team  
**Scope:** Strict communication interface between **Laptop 1 (FastAPI ML Backend)** and **Laptop 2 (Next.js Evaluator Frontend)**.

---

## 1. Architecture & Design Principles

1. **Decoupled Responsibilities:**
   - **Laptop 1 (Teammate / ML Server):** Houses raw road-network graphs, GPS candidate generation, trained XGBoost model artifacts, HMM transition matrices, Viterbi decoder, ground-truth reference routes, and server-side evaluation calculations.
   - **Laptop 2 (Evaluator UI):** Pure frontend client. Responsible for trip presentation, interactive map rendering, user-triggered inference requests, and displaying returned metrics.
2. **Absolute Ground-Truth Isolation (Zero-Leakage Security):**
   - Ground truth reference labels and reference route arcs **MUST NEVER** be transmitted over the network or exposed to Laptop 2.
   - All evaluation metrics (Macro F1, Top-1 Accuracy, Route Continuity, Miss Rate) are computed internally on Laptop 1 and returned as scalar values.
3. **GeoJSON & Coordinate Standardization:**
   - All geographical coordinates in responses must be strictly ordered as `[latitude, longitude]` for Leaflet point arrays or GeoJSON `[longitude, latitude]` (specified with standard keys `lat` and `lon`).
   - Projected UTM metric coordinates used internally by the model must be converted back to WGS84 `(lat, lon)` before serializing to the API response.

---

## 2. API Endpoints

### 2.1 Health & Readiness Check

```http
GET /health
```

#### Response (`200 OK`)
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "server_time": "2026-11-18T10:15:30.123Z",
  "model_loaded": true,
  "model_algorithm": "XGBoost",
  "features_loaded": 4,
  "hmm_ready": true,
  "evaluator_trips_available": 10
}
```

---

### 2.2 List Available Evaluation Trips

```http
GET /api/v1/trips
```

Returns the 10 reserved evaluator trips available on the backend server.

#### Response (`200 OK`)
```json
{
  "trips": [
    {
      "trip_id": "00000000",
      "status": "available",
      "num_gps_points": 1845,
      "approx_duration_s": 920.0,
      "bounds": {
        "min_lat": 51.4921,
        "max_lat": 51.5124,
        "min_lon": -0.1420,
        "max_lon": -0.0982
      }
    },
    {
      "trip_id": "00000007",
      "status": "available",
      "num_gps_points": 2130,
      "approx_duration_s": 1065.0,
      "bounds": {
        "min_lat": 51.4880,
        "max_lat": 51.5050,
        "min_lon": -0.1350,
        "max_lon": -0.1010
      }
    }
  ],
  "total": 10
}
```

---

### 2.3 Run Prediction on Selected Trip

```http
POST /api/v1/predict
Content-Type: application/json
```

#### Request Payload
```json
{
  "trip_id": "00000000"
}
```

#### Response (`200 OK`)
```json
{
  "trip_id": "00000000",
  "status": "success",
  "timing": {
    "preprocessing_ms": 42.1,
    "inference_ms": 118.5,
    "viterbi_ms": 85.3,
    "evaluation_ms": 14.2,
    "total_pipeline_ms": 260.1
  },
  "metrics": {
    "accuracy": 0.9421,
    "precision": 0.7812,
    "recall": 0.8654,
    "macro_f1": 0.8910,
    "top1_accuracy": 0.8245,
    "top1_accuracy_conditional": 0.8271,
    "candidate_miss_rate": 0.0031,
    "route_continuity": 0.9125,
    "log_loss": 0.1482
  },
  "candidate_stats": {
    "total_observations": 1845,
    "total_candidates": 14760,
    "avg_candidates_per_obs": 8.0,
    "candidate_misses": 6
  },
  "route": {
    "num_matched_segments": 142,
    "matched_arc_ids": [102, 105, 106, 120, 125],
    "predicted_polyline": [
      [51.49215, -0.14198],
      [51.49230, -0.14150],
      [51.49280, -0.14080]
    ]
  },
  "gps_observations": [
    {
      "obs_id": "00000000_OBS_0000",
      "timestamp": 1698745200,
      "lat": 51.49214,
      "lon": -0.14195,
      "speed_mps": 11.2,
      "matched_arc_id": 102,
      "confidence": 0.941
    }
  ]
}
```

#### Error Responses
- `400 Bad Request`: `{ "error": "Invalid trip ID format" }`
- `404 Not Found`: `{ "error": "Trip '00000099' not found on server" }`
- `500 Internal Server Error`: `{ "error": "Inference pipeline failed: <reason>" }`

---

### 2.4 Model Information

```http
GET /api/v1/model/info
```

#### Response (`200 OK`)
```json
{
  "model_family": "Gradient Boosted Decision Trees",
  "algorithm": "XGBoost",
  "version": "2.1.4",
  "framework": "xgboost-python",
  "feature_count": 4,
  "feature_names": [
    "distance_to_segment_m",
    "projection_fraction",
    "observation_speed_mps",
    "heading_difference_rad"
  ],
  "hyperparameters": {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic"
  },
  "decoder": {
    "type": "Hidden Markov Model with Viterbi Decoding",
    "emission_source": "XGBoost positive-class probability",
    "transition_metric": "Road-network graph adjacency + distance penalty"
  }
}
```

---

## 3. Laptop 1 Server Requirements

FastAPI backend must start with:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
And must configure CORS middleware to allow requests from any origin during the local network demonstration:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

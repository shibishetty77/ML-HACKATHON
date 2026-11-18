# PathSnap API contract

The backend runs on Laptop 1 at `http://<LAPTOP_1_IP>:8000`. The frontend's existing connection dialog sets this base URL at runtime.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Model, decoder, and evaluator trip readiness |
| GET | `/docs` | FastAPI Swagger UI |
| GET | `/api/status` | Alias for readiness |
| GET | `/api/v1/trips` | Ten discovered evaluator trip summaries |
| POST | `/api/v1/predict` | Real candidate scoring and route decoding for `{ "trip_id": "00000023" }` |
| GET | `/api/v1/model/info` | Loaded artifact, feature order, hyperparameters, decoder description |
| GET | `/api/v1/system/metrics` | Process uptime, request count, measured mean pipeline latency |

`POST /api/v1/predict` returns the existing UI keys: `trip_id`, `status`, `timing`, `metrics`, `candidate_stats`, `route`, and `gps_observations`. `status` is `success` for a fully connected decoded sequence, `partial` for unresolved steps, and `failed` if no route can be decoded. `route.predicted_polyline` is the first continuous part for older clients; `route.polylines` contains every separate continuous part. `route.arc_chunks` is the corresponding directed arc sequence, and `route.unresolved_steps` counts breaks. The frontend plots `polylines` separately so it never draws a false bridge. `candidate_rankings` adds the five highest-scored candidates per observation.

The evaluator `.route` file is a chronological arc sequence, **not** a reference arc for each GPS observation. Therefore `metrics.top1_accuracy`, `top1_accuracy_conditional`, `accuracy`, `precision`, `recall`, `macro_f1`, `candidate_miss_rate`, and `log_loss` are `null` for evaluator trips. The UI shows route arc precision and F1 cards in place of the unavailable candidate Top-1 and Macro F1 cards, with an explanation on the results page. The backend computes `route_arc_precision`, `route_arc_recall`, and `route_arc_f1` by comparing the unique decoded arc IDs with the reference route arc IDs **after** inference. These are route-level overlap measures, not candidate classification scores. `route_continuity` is the fraction of consecutive GPS steps connected by the decoded directed route; `no_candidate_rate` is the fraction of GPS points with zero road candidates. Neither is a substitute for the unavailable label-based candidate miss rate.

Reference route IDs, raw ground truth, and labels are never returned by the API or used to build features. Evaluation runs after model scoring and Viterbi decoding. The backend reads the evaluator ZIP directly. A malformed trip ID returns 422, an absent evaluator trip returns 404, and a processing failure returns 500 with details in server logs.

The ordered model features are `distance_to_segment_m`, `projection_fraction`, `observation_speed_mps`, and `heading_difference_rad`. Candidate search uses 50 m. The model is loaded at startup. Configured exact CORS origins and private LAN origins at port 3000 are accepted without credentials.

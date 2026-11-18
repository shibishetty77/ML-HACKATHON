# PathSnap verification results — 18 November 2026

Executed locally on Windows with Python 3.11, XGBoost 3.2.0, the supplied XGBoost 3.4.1 model JSON, Next.js 16.4.6, and the unmodified `evaluator_test.zip`. The model was loaded without retraining. These checks were completed before the GitHub push.

| Check | Result |
| --- | --- |
| Backend import and startup | PASS — Uvicorn started on `0.0.0.0:8000` |
| Tuned model load | PASS — 550 rounds, 4 features, SHA-256 `314f77223c0e8afb268ad281a9618d59ef6324dd7e1051acbb486d4ffb11a1dc` |
| Tuning archive integrity | PASS — 16 files copied byte-for-byte; Optuna DB has 25 complete and 6 failed trials |
| Health, Swagger, status, trip listing, model info | PASS — `/health`, `/docs`, `/api/status`, `/api/v1/trips`, `/api/v1/model/info` |
| Backend unit and evaluator integration suite | PASS — 17 tests, including all ten evaluator trips and directed arc-chain checks |
| Frontend live API suite | PASS — 6 checks against running FastAPI |
| Route metric display | PASS — browser showed 69.41% route arc precision, 81.94% route arc F1, and 100% route arc recall for evaluator trip `00000023`; candidate Top-1 and Macro F1 remain `null` because per-observation labels are absent |
| Saved server URL hydration | PASS — changed the browser's server URL, reloaded the frontend, confirmed the saved URL appeared after hydration with no browser errors, and confirmed the connection dialog retained it |
| Frontend production build and TypeScript | PASS — Next.js build completed |
| Frontend dev server pages | PASS — `/`, `/model`, `/results`, and `/status` returned HTTP 200 |
| One-laptop browser demo | PASS — `localhost:3000` connected to `localhost:8000`; trip `00000023` returned 458 GPS observations and 87 matched segments; the map displayed the route and loaded OpenStreetMap tiles |
| Frontend lint | PASS with 23 existing unused-variable warnings and zero errors |
| CORS preflight for private-network frontend origin | PASS — returned the exact origin, no credentials |
| Laptop 1 Wi-Fi IPv4 access | PASS — `http://10.220.6.135:8000/health` responded locally |
| Laptop 2 hotspot access | NOT TESTED — requires the second physical laptop |
| Original feature score reproduction | PASS with caveat — 10,688/10,692 candidate IDs matched on student-test trip `00000000`; score correlation 0.999987, median absolute difference ~2e-10, 99th percentile ~0.0051 |

## Evaluator trip execution

The final frozen pipeline returned HTTP 200 for every trip. `partial` means it reported disconnected or missing-candidate gaps instead of drawing an invented connecting route. Latencies are observed end-to-end backend pipeline times in this run and will vary by machine.

| Trip | GPS points | Candidates | Status | Continuous chunks | Unresolved steps | Route continuity | Route arc F1 | Pipeline ms |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 00000015 | 6,854 | 55,290 | partial | 9 | 29 | 0.9958 | 0.6667 | 2,178 |
| 00000017 | 1,346 | 33,110 | success | 1 | 0 | 1.0000 | 0.9413 | 2,404 |
| 00000023 | 458 | 4,929 | success | 1 | 0 | 1.0000 | 0.8194 | 369 |
| 00000030 | 654 | 15,805 | success | 1 | 0 | 1.0000 | 0.9103 | 1,167 |
| 00000049 | 2,642 | 14,283 | success | 1 | 0 | 1.0000 | 0.9976 | 934 |
| 00000059 | 1,031 | 19,850 | success | 1 | 0 | 1.0000 | 0.9337 | 1,182 |
| 00000067 | 916 | 6,348 | success | 1 | 0 | 1.0000 | 0.9586 | 438 |
| 00000073 | 1,144 | 11,312 | partial | 6 | 5 | 0.9956 | 0.9634 | 676 |
| 00000087 | 1,399 | 13,636 | success | 1 | 0 | 1.0000 | 0.9836 | 1,154 |
| 00000094 | 3,230 | 39,014 | success | 1 | 0 | 1.0000 | 0.9250 | 1,920 |

`route_arc_f1` compares unique predicted and reference arc IDs after decoding. It is **not** observation Top-1 or candidate macro F1. The evaluator archive has no per-observation reference labels; those fields are returned as `null`. Trip `00000015` had 21 points with no road candidates. The baseline model was absent, so the optional fallback path could not be exercised. `http://10.220.6.135:8000` was reachable from Laptop 1 through its own Wi-Fi address; this is not evidence of reachability from Laptop 2.

The Python test run emitted one deprecation warning from the installed Starlette/httpx TestClient combination. It did not affect test results.

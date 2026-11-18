# PathSnap

PathSnap matches noisy GPS tracks to directed road arcs. Laptop 1 runs FastAPI with the supplied tuned XGBoost model and a log-space HMM/Viterbi decoder. Laptop 2 runs the existing Next.js evaluator UI and selects one of ten evaluator trips.

## Current implementation

The backend reads `evaluator_test.zip` directly, creates 50 m road candidates, computes the four model features in their training order, scores all candidates in one XGBoost batch, and decodes a directed route. It returns separate polylines when a gap cannot be connected. Ground-truth `.route` data is read only after decoding for route arc overlap evaluation. The model is loaded once at startup. The API logs each prediction request and result.

The supplied raw `.route` files do not give a reference arc for every GPS point. The evaluator API therefore reports observation Top-1 accuracy, candidate macro F1, candidate miss rate, and log loss as unavailable (`null`) instead of inventing them. It reports route arc precision/recall/F1 and directed continuity separately. See [API contract](docs/api_contract.md).

## Start on Laptop 1 (Windows)

1. Install Python 3.11 and run `py -3.11 -m pip install -r backend\requirements.txt`.
2. Keep the unmodified evaluator ZIP in Downloads or the repository root, or set `$env:PATHSNAP_EVALUATOR_DATA = 'C:\path\to\evaluator_test.zip'`.
3. From the repo root run `powershell -ExecutionPolicy Bypass -File backend\start_backend.ps1`.
4. Check `http://localhost:8000/health` and `http://localhost:8000/docs`.

The startup script prints the current Laptop 1 Wi-Fi URL. It binds Uvicorn to `0.0.0.0:8000` and keeps logs visible. See [two-laptop network setup](backend/NETWORK_SETUP.md) for Laptop 2, firewall, and hotspot steps.

## Run both services on one Windows laptop

1. From the repository root, start the backend in one PowerShell window: `powershell -ExecutionPolicy Bypass -File backend\start_backend.ps1`.
2. Open a second PowerShell window in `frontend`. Run `npm.cmd ci` once, then `npm.cmd run dev`. The `.cmd` form works when PowerShell blocks `npm.ps1`.
3. Open `http://localhost:3000`. In **Server Connection**, enter `http://localhost:8000`.
4. Select trip `00000023` and click **Run Prediction**. The backend window logs the XGBoost/HMM request; the frontend shows metrics and the route map.

The basemap uses OpenStreetMap tiles and needs internet access. The GPS observations and predicted route come from the local backend.

## Test

Set `PATHSNAP_EVALUATOR_DATA` to `evaluator_test.zip`, then from the repo root run `py -3.11 -m pip install -r backend\requirements-dev.txt` and `py -3.11 -m pytest backend\tests -q`. With the server running on port 8000, run `npm ci`, `npm test`, `npm run build`, and `npm run lint` in `frontend`.

## Repository structure

- `backend/app/data.py`: evaluator ZIP/directory loading and validation.
- `backend/app/geometry.py`: directed road graph, spatial candidate index, four feature values, shortest directed paths.
- `backend/app/model.py`: tuned model loading, optional baseline fallback, feature validation, batch probabilities.
- `backend/app/hmm.py`: log-space Viterbi with gaps and disconnected transitions.
- `backend/app/main.py`: frontend-compatible API and post-prediction evaluation.
- `frontend`: existing Next.js UI with runtime server URL setting and separate route chunks on the map.
- `ML_Modles`: four original model notebooks in the checkout.
- `outputs/xgboost/hyperparameter_tuning`: unchanged supplied tuned JSON, selected parameters, Optuna study, CSV results, report, predictions and plots.

See the [deliverable audit](docs/DELIVERABLE_AUDIT.md) for verified artifacts, the missing baseline model, newly added shared preprocessing source, and remaining submission work. The evaluator ZIP is kept outside Git.

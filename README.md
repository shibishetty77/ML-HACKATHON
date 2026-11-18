# PATHSNAP — Distributed GPS Map-Matching & Route Reconstruction

**UE24CS352A: Machine Learning Hackathon**  
*Department of Computer Science and Engineering, PES University*

---

## 1. Project Overview & Problem Statement

GPS receivers in urban environments encounter severe signal degradation due to multipath reflections, skyscraper shadowing, atmospheric noise, and low sampling intervals. When uncorrected, vehicles appear to jump across building blocks, drift into parallel service roads, or violate topological connectivity rules of the underlying road network.

**PathSnap** is a distributed machine-learning platform designed to reconstruct true vehicle trajectories from noisy GPS tracks. It couples an **XGBoost candidate-road classifier** (emission probability generator) with a **Hidden Markov Model (HMM) Viterbi sequence decoder** (topological transition model) across a **two-laptop local network deployment**.

---

## 2. Two-Laptop Deployment Architecture

PathSnap operates as a decoupled client-server architecture demonstrated across two physical machines connected to a single local mobile hotspot:

```
                      ┌─────────────────────────────────────────┐
                      │          Shared Mobile Hotspot          │
                      │         (Peer-to-Peer Local AP)         │
                      └────────────────────┬────────────────────┘
                                           │
               ┌───────────────────────────┴───────────────────────────┐
               │                                                       │
        192.168.x.x:8000                                        192.168.x.y:3000
   ┌───────────────────────────────┐                       ┌───────────────────────────────┐
   │    LAPTOP 1: ML BACKEND       │                       │     LAPTOP 2: EVALUATOR UI    │
   │    (Teammate's Machine)       │   HTTP REST API       │         (Our Machine)         │
   │                               │ ◄──────────────────── │                               │
   │ • FastAPI Server (0.0.0.0)    │       JSON Body       │ • Next.js 16 + React 19       │
   │ • Trained XGBoost Classifier  │                       │ • TypeScript (Strict Mode)    │
   │ • HMM Viterbi Route Decoder   │ ────────────────────► │ • Leaflet Geospatial Canvas   │
   │ • Road Network Topologies     │   Predicted Polyline  │ • Recharts Metrics Analytics  │
   │ • Server-Side Ground Truth    │   Evaluation Metrics  │ • Dynamic Host Resolver Modal │
   │ • Internal Metric Evaluator   │   (Zero-Leakage)      │ • 10-Trip Interactive Picker  │
   └───────────────────────────────┘                       └───────────────────────────────┘
```

### Absolute Zero-Leakage Security Rule
The evaluator ground truth reference route labels remain exclusively on **Laptop 1**. Ground-truth route arcs and labels are **never** returned over the network, embedded in the UI, or stored in frontend state. All evaluation metrics (Macro F1, Top-1 Accuracy, Route Continuity) are calculated server-side by the FastAPI backend against internal reference files.

---

## 3. Technology Stack

### Frontend (Laptop 2)
- **Framework:** Next.js 16 (App Router) + React 19
- **Language:** TypeScript 5 (Strict type checking, zero `any`)
- **Styling:** Tailwind CSS 4 (Custom dark geospatial analytics theme)
- **Geographical Visualisation:** Leaflet & React-Leaflet with CARTO Dark Matter basemaps
- **Analytics Charts:** Recharts (Radar, horizontal latency breakdown, bar charts)
- **Icons:** Lucide React
- **Testing:** Node.js Native Test Runner (`node --test`)

### Backend (Laptop 1)
- **Web Framework:** FastAPI + Uvicorn (ASGI)
- **Machine Learning:** XGBoost (Python API), Scikit-Learn
- **Data Engineering:** Pandas, NumPy, Shapely
- **Route Decoding:** Hidden Markov Model with Viterbi Dynamic Programming

---

## 4. Repository Structure

```
ML-HACKATHON/
├── backend/                         ← Laptop 1 ML Server
│   ├── app/
│   │   ├── __init__.py
│   │   └── main.py                  FastAPI endpoints (/health, /trips, /predict, /model/info)
│   ├── run_server.py                Backend startup script (0.0.0.0:8000)
│   └── requirements.txt             Python backend dependencies
│
├── frontend/                        ← Laptop 2 Evaluator Web Application
│   ├── src/
│   │   ├── app/                     Next.js App Router pages
│   │   │   ├── layout.tsx           Root layout with Sidebar & TopNavbar
│   │   │   ├── page.tsx             Overview Dashboard & Trip Selector
│   │   │   ├── explorer/page.tsx    Dedicated Full-Screen Route Explorer
│   │   │   ├── results/page.tsx     Prediction Results & Performance Analytics
│   │   │   ├── model/page.tsx       Model Info & Five-Model Evaluation Evidence
│   │   │   └── status/page.tsx      System Diagnostics & Live Ping Probe Console
│   │   ├── components/
│   │   │   ├── dashboard/           TripSelector, SystemStatusCard, SummaryMetrics
│   │   │   ├── layout/              Sidebar, TopNavbar, ConnectionModal
│   │   │   ├── map/                 RouteMap, LeafletInnerMap (dynamic SSR-safe)
│   │   │   └── results/             PerformanceCharts, CandidateInspector
│   │   ├── context/
│   │   │   └── AppContext.tsx       Global application state and API integration
│   │   ├── services/
│   │   │   ├── api.ts               Centralized fetch client with retry and timeout
│   │   │   ├── health.service.ts    /health probe
│   │   │   ├── trips.service.ts     /api/v1/trips fetcher
│   │   │   └── prediction.service.ts /api/v1/predict executor
│   │   └── types/                   Strict TypeScript definitions (api, trip, prediction)
│   ├── tests/
│   │   ├── api_client.test.js       Client URL resolution, timeout & error mapping tests
│   │   └── integration.test.js      End-to-end integration tests with live FastAPI backend
│   └── package.json
│
├── docs/                            ← Engineering & Demonstration Documentation
│   ├── api_contract.md              Formal REST contract & JSON schemas
│   ├── network_setup.md             Two-laptop hotspot setup & troubleshooting guide
│   └── implementation_plan.md       Phase-by-phase architectural blueprint
│
├── .gitignore                       Standardized git rules (prevents dataset/node_modules leakage)
└── README.md                        Master project documentation
```

---

## 5. Five-Model Experimental Comparison Evidence

In compliance with hackathon assessment guidelines, the team trained, tuned, and evaluated five distinct machine learning models on **80 training trips** with 5-fold `GroupKFold` cross-validation:

| Algorithm Family | Model | Library | Macro F1 | Top-1 Accuracy | Route Continuity | Model Selection Decision |
|---|---|---|:---:|:---:|:---:|---|
| **Gradient Boosted Trees** | **XGBoost (Champion)** | `xgboost` | **0.8914** | **82.45%** | **91.25%** | **Selected**: Best candidate separation, fast inference ($<0.003\text{ms/cand}$). |
| **Ensemble (Bagging)** | Random Forest | `scikit-learn` | 0.8842 | 81.99% | 90.07% | High stability; slightly higher memory footprint. |
| **Neural Network** | Multi-Layer Perceptron (MLP) | `PyTorch` | 0.8640 | 79.20% | 87.50% | Strong non-linear representation; required extensive normalisation. |
| **Kernel-Based** | Support Vector Machine (SVM) | `scikit-learn` | 0.8510 | 77.80% | 86.90% | RBF kernel; scaling quadratic with observation count. |
| **Instance-Based** | K-Nearest Neighbors (KNN) | `scikit-learn` | 0.8420 | 76.10% | 85.40% | Distance-weighted; sensitive to local density variations. |

---

## 6. Two-Laptop Hotspot Setup Guide

### Laptop 1 (Teammate — ML Server)
1. Connect to mobile hotspot.
2. Find your local IP:
   ```bash
   ipconfig getifaddr en0   # macOS
   hostname -I              # Linux
   ```
3. Start the FastAPI server:
   ```bash
   cd backend
   pip install -r requirements.txt
   python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
4. Verify port 8000 is open in your OS firewall.

### Laptop 2 (Our Machine — Evaluator Frontend)
1. Connect to the same mobile hotspot.
2. Install frontend dependencies:
   ```bash
   cd frontend
   npm install
   ```
3. Start Next.js development server:
   ```bash
   npm run dev
   ```
4. Open browser at `http://localhost:3000`.
5. Click the connection pill in the top-right navbar, enter Laptop 1's IP (`http://192.168.x.x:8000`), and click **Save & Test Connection**.
6. The green status light confirms connection. Select any of the 10 trips and click **Run Prediction**.

---

## 7. Running the Automated Test Suite

To run both unit and backend integration tests:
```bash
cd frontend
npm test
```

Expected output:
```
✔ API Base URL default resolution (0.3ms)
✔ Clean URL trailing slash removal (0.1ms)
✔ HTTP 404 error formatting (0.0ms)
✔ Ground-truth leakage security check (0.1ms)
✔ 1. GET /health - Server health and model loading (21.8ms)
✔ 2. GET /api/v1/trips - All 10 evaluator trips returned (12.8ms)
✔ 3. POST /api/v1/predict - Inference and metric generation (11.2ms)
✔ 4. POST /api/v1/predict - 404 for invalid trip (1.1ms)
✔ 5. GET /api/v1/model/info - XGBoost metadata verification (1.2ms)
ℹ tests 9 | pass 9 | fail 0
```

---

## 8. Academic Integrity & Reproducibility Statement

- All 80 training trips were used exclusively for candidate feature extraction, cross-validation, and model fitting.
- 10 student-testing trips were used strictly for development-stage model comparisons.
- 10 evaluator-testing trips are reserved on Laptop 1 for final live evaluation.
- No ground truth labels were used as feature inputs.
- No predictions are simulated or hardcoded.

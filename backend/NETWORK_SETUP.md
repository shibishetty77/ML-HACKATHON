# PathSnap two-laptop setup

The backend runs on Laptop 1. The existing Next.js UI runs on Laptop 2. Both must join the same phone hotspot, and the hotspot must allow devices to reach each other.

## Laptop 1: Windows backend

1. Copy this repository and the **unmodified** `evaluator_test.zip` to Laptop 1. Keep the ZIP outside Git. The backend reads it directly without extracting or duplicating the data.
2. From the repository root, install Python 3.11 dependencies: `py -3.11 -m pip install -r backend\requirements.txt`.
3. Set `$env:PATHSNAP_EVALUATOR_DATA = 'C:\path\to\evaluator_test.zip'` if the ZIP is not in the repo root or Downloads.
4. Run `powershell -ExecutionPolicy Bypass -File backend\start_backend.ps1` from the repository root. The script checks dependencies, prints the current Wi-Fi IPv4 address, and starts Uvicorn on `0.0.0.0:8000`. Logs stay visible.
5. On Laptop 1, open `http://localhost:8000/health` and `http://localhost:8000/docs`. Health should report `model_loaded: true`, `hmm_ready: true`, and 10 evaluator trips.

If needed, run `Get-NetIPAddress -AddressFamily IPv4 | Where-Object InterfaceAlias -Match 'Wi-Fi|WLAN|Wireless'` to find the current Laptop 1 hotspot address. It changes when the hotspot changes.

## Laptop 2: Next.js frontend

1. Join the **same** hotspot. From Laptop 2, open `http://<LAPTOP_1_IP>:8000/health` or run `curl http://<LAPTOP_1_IP>:8000/health`.
2. In `frontend`, run `npm ci`, then `npm run dev`. Open `http://localhost:3000`.
3. Use the UI's **Server Connection** dialog to enter `http://<LAPTOP_1_IP>:8000`. This setting is saved in the browser and does not require rebuilding Next.js. `NEXT_PUBLIC_API_BASE_URL` in `frontend/.env.local` is an optional initial default.
4. Select any of the ten evaluator trips and run inference. Watch the Laptop 1 terminal for start, finish, and failure logs.

The backend accepts browser origins on private IPv4 networks at port 3000, plus `localhost:3000`. For another origin, set `PATHSNAP_CORS_ORIGINS` to a comma-separated list of exact origins before starting Laptop 1. Credentials are disabled.

## Firewall and troubleshooting

If Laptop 2 can reach the hotspot but cannot reach port 8000, allow inbound TCP 8000 for the current private network. In an elevated PowerShell terminal on Laptop 1:

`New-NetFirewallRule -DisplayName 'PathSnap API' -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Private`

Check that Windows classifies the hotspot as a Private network, that Uvicorn shows `0.0.0.0:8000`, and that the current Laptop 1 IP is entered in the UI. Some phone hotspots isolate clients; if so, use a hotspot/router that permits peer connections. A CORS browser error indicates an origin mismatch; set `PATHSNAP_CORS_ORIGINS` to Laptop 2's exact `http://IP:3000` origin and restart the backend.

The model file is `outputs/xgboost/hyperparameter_tuning/tuned_xgboost_model.json`. Startup fails if no compatible model or evaluator archive is available. The baseline fallback is optional and was absent from the supplied materials.

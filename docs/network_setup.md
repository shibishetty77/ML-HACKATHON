# PathSnap — Two-Laptop Deployment & Mobile Hotspot Networking Guide

**Architecture:** Distributed Map-Matching Demonstration across Two Physical Machines  
**Demonstration Scope:** Evaluator operates Laptop 2 (Frontend) which queries Laptop 1 (FastAPI + XGBoost ML Server) over a shared local Wi-Fi / Mobile Hotspot.

---

## 1. Network Topology Overview

```
                      ┌────────────────────────────┐
                      │    Shared Wi-Fi Network    │
                      │  (e.g., Phone Mobile AP)   │
                      └─────────────┬──────────────┘
                                    │
               ┌────────────────────┴────────────────────┐
               │                                         │
        192.168.43.120                            192.168.43.205
   ┌───────────────────────┐                 ┌───────────────────────┐
   │  LAPTOP 1: ML SERVER  │                 │  LAPTOP 2: FRONTEND   │
   │  (Teammate's Machine) │ ◄────────────── │     (Our Machine)     │
   │                       │   HTTP REST     │                       │
   │ • FastAPI (0.0.0.0)   │   port 8000     │ • Next.js (port 3000) │
   │ • XGBoost Model       │                 │ • Leaflet Map UI      │
   │ • HMM Viterbi Decoder │                 │ • Trip Selector       │
   │ • Server Evaluator    │                 │ • Live Metrics Cards  │
   └───────────────────────┘                 └───────────────────────┘
```

---

## 2. Step-by-Step Deployment Procedure

### Phase A: Mobile Hotspot Setup
1. Turn on personal mobile hotspot on a smartphone.
2. Connect **Laptop 1** and **Laptop 2** to the same hotspot network.
3. *Important:* Ensure the phone hotspot does NOT have "AP Isolation" or "Guest Mode" enabled (which prevents devices from talking to each other).

---

### Phase B: Laptop 1 (ML Backend Setup)
1. Open terminal on Laptop 1.
2. Find the local LAN IP address assigned by the hotspot:
   - **macOS:**
     ```bash
     ipconfig getifaddr en0
     # Output example: 192.168.43.120
     ```
   - **Linux:**
     ```bash
     hostname -I | awk '{print $1}'
     # Output example: 192.168.43.120
     ```
   - **Windows:**
     ```powershell
     Get-NetIPAddress -InterfaceAlias 'Wi-Fi' -AddressFamily IPv4 | Select-Object -ExpandProperty IPAddress
     ```
3. Start the FastAPI backend server binding strictly to `0.0.0.0`:
   ```bash
   cd backend
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
4. Verify server is listening on port 8000 locally:
   ```bash
   curl http://localhost:8000/health
   ```

---

### Phase C: Laptop 2 (Frontend Setup)
1. Open terminal on Laptop 2.
2. Test network reachability to Laptop 1:
   ```bash
   # Replace with Laptop 1's actual IP discovered in Phase B
   curl http://192.168.43.120:8000/health
   ```
   *Expected response:* JSON containing `{"status": "healthy", ...}`.
3. In `frontend/`, configure `.env.local`:
   ```bash
   echo "NEXT_PUBLIC_API_BASE_URL=http://192.168.43.120:8000" > .env.local
   ```
   *(Note: The frontend UI also includes a live runtime Server Connection bar where the evaluator or developer can change the IP directly in the browser without restarting Next.js!)*
4. Install dependencies (first time only):
   ```bash
   cd frontend
   npm install
   ```
5. Launch the Next.js development server:
   ```bash
   npm run dev
   ```
6. Open browser on Laptop 2 at:
   ```
   http://localhost:3000
   ```
7. Verify the green connection status indicator in the top navbar.

---

## 3. Firewall & Security Settings

### macOS (Laptop 1)
If Laptop 2 cannot connect (`Connection refused` or `ETIMEDOUT`):
1. Go to **System Settings > Network > Firewall**.
2. If Firewall is ON, click **Options...** and ensure `python` or `uvicorn` is set to **Allow incoming connections**.
3. Or temporarily toggle Firewall to OFF during the evaluation session.

### Linux (Laptop 1)
```bash
sudo ufw allow 8000/tcp
```

### Windows (Laptop 1)
Run PowerShell as Administrator:
```powershell
New-NetFirewallRule -DisplayName "FastAPI ML Server" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
```

---

## 4. Troubleshooting Checklist

| Symptom | Probable Cause | Immediate Remediation |
|---|---|---|
| `ERR_CONNECTION_REFUSED` | Server is bound to `127.0.0.1` instead of `0.0.0.0` | In `uvicorn` command, ensure flag is `--host 0.0.0.0`. |
| `ERR_CONNECTION_TIMED_OUT` | Host firewall blocking port 8000 | Disable firewall or allow port 8000 on Laptop 1. |
| `CORS Error` in Browser Console | Missing CORS middleware in FastAPI | Ensure `CORSMiddleware` with `allow_origins=["*"]` is active in `app.main:app`. |
| `Hotspot Client Isolation` | Mobile carrier AP blocks peer-to-peer traffic | Connect both laptops to a local Wi-Fi router, portable travel router, or create an ad-hoc computer-to-computer Wi-Fi network. |
| Server IP changed | Hotspot DHCP re-assigned IP | Re-check `ipconfig getifaddr en0` on Laptop 1 and update the address in the PathSnap UI top bar. |

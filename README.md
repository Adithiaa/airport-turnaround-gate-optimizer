# aeroT: Airport Operations Control Center (AOCC) Digital Twin & Telemetry Suite

[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![Google OR-Tools](https://img.shields.io/badge/Google%20OR--Tools-CP--SAT-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://developers.google.com/optimization)
[![LightGBM](https://img.shields.io/badge/ML%20Inference-LightGBM-2E7D32?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://lightgbm.readthedocs.io/)
[![Spatial 3D](https://img.shields.io/badge/Spatial%203D-Pydeck%20%7C%20Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://deck.gl/pydeck/)


---

## 📌 Executive Summary

Modern mega-hub airports face extreme operational friction: sudden weather deterioration, arrival holding stacks, ground turnaround variance, and gate allocation bottlenecks that cost operators millions in avoidable delays. 

**aeroT** is an enterprise-grade Digital Twin and Decision-Support System engineered for **Airport Operations Control Centers (AOCC)**, modeled directly on the operational topology of **Kempegowda International Airport (BLR / VOBL)** in Bengaluru, India.

Operating within the framework of **ICAO Airport Collaborative Decision Making (A-CDM)**, aeroT synthesizes live ADS-B transponder telemetry, machine learning-driven ground turnaround predictions, and constraint satisfaction programming (CP-SAT) to eliminate stand conflicts, safeguard critical-path turnaround milestones, and provide real-time situational awareness during severe apron and weather disruptions.

---

## 🏗️ System Architecture & Data Pipeline

 [ OpenSky ADS-B Network ]
                                              │ (REST API / JSON)
                                              ▼
                                 ┌─────────────────────────┐
                                 │ Live Telemetry Engine   │
                                 │ - Great-Circle Distance │
                                 │ - Kinetic Descent Rate  │
                                 │ - TMA Inflow & Holding  │
                                 └────────────┬────────────┘
                                              │
                       ┌──────────────────────┴──────────────────────┐
                       ▼                                             ▼
        ┌─────────────────────────────┐               ┌─────────────────────────────┐
        │ Machine Learning Regressor  │               │ A-CDM Reactive Engine       │
        │ - LightGBM Turnaround Model │               │ - CAT III LVP Conditions    │
        │ - Fleet & Route Features    │               │ - Linear Apron Congestion   │
        │ - Target Turnaround (TAT)   │               │ - Separation Buffers        │
        └──────────────┬──────────────┘               └──────────────┬──────────────┘
                       │                                             │
                       └──────────────────────┬──────────────────────┘
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │ Google OR-Tools CP-SAT Solver   │
                             │ - Stand Admissibility (C/E/F)   │
                             │ - Disjunctive Non-Overlap       │
                             │ - Bus Gate Penalty Minimization │
                             └────────────────┬────────────────┘
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │ Streamlit Command Deck (5 UI)   │
                             │ - 3D Radar Vector Pydeck Engine │
                             │ - Great-Circle Corridor Network │
                             │ - Ground SLA Milestone Tracker  │
                             │ - Apron Occupancy Gantt Chart   │
                             └────────────────┬────────────────┘
                                              ▼
## 🧭 Core Dashboard Modules

### 1. ⚡ Executive Command Deck
* **Real-Time A-CDM Throughput:** Live telemetry KPI cards calculating Active Airspace Movements, Mean Inferred Turnaround (TAT), Remote Stand Demotion counts, and Network On-Time Performance (OTP).
* **Tactical Approach Holding Waterfall:** Translates vertical rates and airspeeds into operational queues: On-Time Sequenced ($0\text{--}5\text{m}$), Speed Buffered ($5\text{--}20\text{m}$), and Critical Stack Holding ($>20\text{m}$).
* **Dynamic Stand Utilization:** Live split between Contact Pier Bridges (Direct Terminal Boarding) and Remote Hardstands (Bus Operations).
  ![Executive Command Deck](aerot/Screenshot%202026-11-07%20at%205.31.11%20PM.png)
  
### 2. 📡 Live Flight Radar & Tactical Movement (BLR Airspace)
* **Real-Time ADS-B Transponder Streams:** Live polling of aircraft transponders within the VOBL terminal maneuvering area ($11.5^\circ\text{--}14.8^\circ\text{ N}, 75.8^\circ\text{--}79.5^\circ\text{ E}$).
* **3D Pydeck Airspace Visualization:** 
  * 🔴 **Red Airframes:** Inbound approach vectors descending into VOBL runways.
  * 🟡 **Yellow Airframes:** Outbound departures climbing through the terminal sector.
  * 🟢 **Green Airframes:** Active ground ramp, taxiway, and turnaround movements.
* **Autonomous Telemetry Enrichment:** Decodes callsigns to identify airline operators, aircraft types (e.g., Airbus A350-900, Boeing 777-300ER, A320neo), and connected domestic/international city pairs.
![Live Radar Movement](aerot/Screenshot%202026-11-07%20at%205.33.01%20PM.png)

### 3. 🌐 Inbound Route Network & Air Corridors
* **Great-Circle Feeder Corridors:** Real-time trajectory lines connecting major domestic hubs and Middle Eastern trunks (DEL, BOM, HYD, MAA, CCU, COK, GOI, PNQ, AMD, DXB, DOH, SIN) to Kempegowda Hub.
* **Kinetic Aircraft Projection:** Projects dynamic aircraft positions along airways using spherical trigonometry and Target Time of Arrival (STA).
* **Airway Inflow Manifest:** Tabular feed detailing tactical en-route delays, ETA offsets, and pre-allocated apron stands.
![Inbound Route Network](aerot/Screenshot%202026-11-07%20at%205.34.42%20PM.png)

### 4. ⏱️ Turnaround Milestone & Ground SLA Engine
* **A-CDM Critical Path Decomposition:** Proportional duration modeling across 9 sequential and parallel sub-stages:
  1. *Chocks On & Jet-Bridge Connect*
  2. *Passenger Deboarding*
  3. *Baggage & Cargo Offload*
  4. *Cabin Cleaning & Waste Disposal*
  5. *Hydrant Fueling Operation*
  6. *Catering Replenishment*
  7. *Baggage Loading*
  8. *Passenger Boarding*
  9. *Tug Hookup, Doors & Pushback*
* **Dynamic Ground Risk Matrix:** Real-time calculation of remaining critical-path buffers, fuel uplift targets (hydrant vs. bowser tanker), catering SLAs, and pushback slot clearance risk.
  ![Turnaround Milestone SLA Engine](aerot/Screenshot%202026-11-07%20at%205.39.40%20PM.png)

### 5. 📊 Gate Optimization Telemetry (Google OR-Tools CP-SAT)
* **Mathematical Stand Allocation:** Uses constraint-programming to solve gate assignments across Terminal 1, Terminal 2, and Western/Eastern remote hardstands.
* **Apron Occupancy Gantt Chart:** High-resolution interactive visual timeline mapping gate utilization by airframe class and stand type, preventing simultaneous occupation.
![Apron Gantt Optimization](aerot/Screenshot%202026-11-07%20at%205.41.19%20PM.png)
---

## 🧮 Mathematical Formulation (OR-Tools CP-SAT)

The gate allocation problem is modeled as a **Disjunctive Interval Scheduling Problem with Complex Compatibility Constraints**:

### Sets and Indices
* $\mathcal{F} = \{1, \dots, n\}$: Set of flights in the active operational window.
* $\mathcal{G} = \{1, \dots, m\}$: Set of aircraft stands at BLR (Contact Bridges + Remote Stands).

### Decision Variables
* $x_{i, g} \in \{0, 1\}$: Binary variable indicating whether flight $i \in \mathcal{F}$ is assigned to stand $g \in \mathcal{G}$.

### Objective Function
Minimize remote stand demotion penalties and unassigned aircraft costs:
$$\min \sum_{i \in \mathcal{F}} \sum_{g \in \mathcal{G}} c_g \cdot x_{i, g} + \sum_{i \in \mathcal{F}} M \cdot \left(1 - \sum_{g \in \mathcal{G}} x_{i, g}\right)$$
*where $c_g = 0$ for contact bridges, $c_g = 80$ for remote bus stands, and $M = 500$ is a severe surge penalty for unassigned aircraft.*

### Operational Constraints
1. **Airframe Class Compatibility:** Wide-body aircraft (Code E / Code F) can only occupy certified stands:
   $$x_{i, g} = 0 \quad \forall i \in \mathcal{F}_{\text{widebody}}, \; g \notin \mathcal{G}_{\text{widebody}}$$
2. **Single Stand Assignment:** Each aircraft must be assigned to at most one stand:
   $$\sum_{g \in \mathcal{G}} x_{i, g} \le 1 \quad \forall i \in \mathcal{F}$$
3. **Stand Non-Overlap with Safety Buffer:** If two flights $i$ and $j$ overlap in their occupancy interval $[S_i, E_i + \Delta_{\text{buffer}}]$ and $[S_j, E_j + \Delta_{\text{buffer}}]$, they cannot share the same stand:
   $$x_{i, g} + x_{j, g} \le 1 \quad \forall g \in \mathcal{G}, \; \forall (i, j) \text{ such that } \max(S_i, S_j) < \min(E_i + \Delta, E_j + \Delta)$$

---

## ⚡ Live Disruption & Reactive Engine

The simulation engine models three core real-world variables:

| Disruption Parameter | Range / Mode | Operational Impact |
|---|---|---|
| **A-CDM Meteorological Profile** | Clear Skies / Monsoon / Severe Fog CAT III | Scales baseline TAT by $+0\%$ to $+40\%$, adds $+5\text{--}10\text{m}$ taxi-in penalty, and triggers $+12\text{m}$ tactical stack holding surge under Low-Visibility Procedures (LVP). |
| **Apron Congestion Index** | $5 \text{ to } 35$ (Baseline: 14) | Dynamically increases apron taxi congestion and extends turnaround milestones without dead zones. |
| **Wake & Separation Buffer** | $10 \text{ to } 30\text{ mins}$ | Contracts effective contact gate capacity, forcing tight arrival waves into remote bus stands. |

---

## 💻 Tech Stack & Architecture Matrix

| Layer | Component | Description |
|---|---|---|
| **Frontend Framework** | **Streamlit** (v1.30+) | High-performance reactive web application framework with custom CSS HUD styling. |
| **Optimization Solver** | **Google OR-Tools (CP-SAT)** | Constraint programming solver for interval scheduling and non-overlap constraints. |
| **Machine Learning** | **LightGBM Regressor** | Gradient-boosted decision tree predicting turnaround duration from fleet and congestion features. |
| **Spatial 3D & Radar** | **Pydeck (deck.gl)** | WebGL-powered spatial 3D aircraft telemetry visualization over geographic coordinates. |
| **Visual Analytics** | **Plotly Express & Graph Objects** | Interactive Gantt timelines, horizontal waterfall charts, and great-circle airway maps. |
| **Telemetry Ingestion** | **OpenSky Network REST API** | ADS-B live transponder polling across the South India terminal airspace. |
| **Data Processing** | **Pandas & NumPy** | In-memory tabular transformations, vector distance calculations, and session caching. |

---

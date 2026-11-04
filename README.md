# 🛫 AOCC Dynamic Turnaround Time & Gate Conflict Optimization Engine

An end-to-end tactical decision-support prototype designed for Airport Operations Control Centres (AOCC). The system combines machine learning-driven ground turnaround duration forecasting with discrete combinatorial optimization (Google OR-Tools CP-SAT) to dynamically reassign aircraft stands, resolve gate conflicts, and minimize apron congestion during irregular operations (IROPS).

---

## 📌 Executive Summary

* **Domain Problem:** Ground turnaround delays propagate rapidly through hub-and-spoke networks. Fixed static gate allocations fail when inbound aircraft suffer delays, resulting in gate conflicts, ramp gridlocks, and costly remote stand busing.
* **Solution Architecture:** 
  1. **Turnaround Time (TAT) Prediction:** LightGBM regressor modeling turnaround variations based on aircraft wake turbulence categories, inbound delays, taxi-in times, and ramp congestion.
  2. **Airport Gate Assignment Problem (AGAP):** Formulated as a Constraint Satisfaction / Integer Programming problem solved via Google OR-Tools CP-SAT.
  3. **Tactical Dispatch Interface:** Streamlit & Plotly Gantt dashboard providing dynamic delay injection and instant schedule regeneration.

---

## 🏗️ System Architecture

```text
                                  [ Flight Schedule / Telemetry ]
                                                │
                                                ▼
                            [ Feature Engineering & Preprocessing ]
                            (Wake Class, Congestion Index, Taxi Times)
                                                │
                                                ▼
                                [ LightGBM TAT Prediction Engine ]
                                  Target: Turnaround Duration (min)
                                                │
                                                ▼
                             [ Google OR-Tools CP-SAT Solver ]
                         Constraints:
                           • Aircraft class compatibility
                           • Gate non-overlap with safety buffer (15 min)
                           • Exactly one gate per flight
                         Objective:
                           • Minimize Remote Apron Penalty
                                                │
                                                ▼
                              [ Interactive Dispatcher Dashboard ]
                                (Streamlit + Plotly Gantt Chart)
                                
## 🧮 Mathematical Formulation (AGAP)

Let $\mathcal{F}$ be the set of active flights and $\mathcal{G}$ be the set of available gates/stands.

### 1. Decision Variables
$$x_{i,g} \in \{0, 1\} \quad \forall i \in \mathcal{F}, \forall g \in \mathcal{G}$$

Where $x_{i,g} = 1$ if flight $i$ is assigned to gate $g$, and $0$ otherwise.

### 2. Constraints

* **Single Stand Assignment:** Each aircraft must be assigned to exactly one stand:
  $$\sum_{g \in \mathcal{G}} x_{i,g} = 1 \quad \forall i \in \mathcal{F}$$

* **Physical Compatibility:** Wide-body aircraft cannot dock at Narrow-body-only contact gates:
  $$x_{i,g} = 0 \quad \text{if } \text{Class}(i) \text{ incompatible with Gate } g$$

* **Temporal Separation & Safety Buffer:** For any two flights $i$ and $j$ where time intervals overlap within the safety buffer $\Delta t_{\text{buffer}}$:
  $$[T^{\text{arr}}_i, T^{\text{arr}}_i + \text{TAT}_i + \Delta t_{\text{buffer}}] \cap [T^{\text{arr}}_j, T^{\text{arr}}_j + \text{TAT}_j + \Delta t_{\text{buffer}}] \neq \emptyset \implies x_{i,g} + x_{j,g} \le 1 \quad \forall g \in \mathcal{G}$$

### 3. Objective Function

Minimize the operational penalty incurred by passenger bus transfers and extended taxiing when aircraft are forced onto remote stands:
$$\min \sum_{i \in \mathcal{F}} \sum_{g \in \mathcal{G}} C_g \cdot x_{i,g}$$

Where $C_g = 0$ for terminal contact bridges and $C_g = 50$ for remote apron stands.

---

## 🚀 Key Features

* **A-CDM Aligned Logic:** Incorporates target turnaround times (TTOT/TSAT) principles into the dispatch loop.
* **Dynamic IROPS Delay Injection:** Interactive sliders simulate sudden flight delays and apron saturation, demonstrating real-time re-optimization.
* **Operational KPI Tracking:** Instant visibility into Contact Stand Utilization rates, active gate conflicts, and remote stand overflows.

---

## 🛠️ Tech Stack

* **Language:** Python 3.11+
* **ML / Data Processing:** LightGBM, Pandas, NumPy, Scikit-learn
* **Combinatorial Optimization:** Google OR-Tools (CP-SAT Solver)
* **Visualization & Dispatch UI:** Streamlit, Plotly Express
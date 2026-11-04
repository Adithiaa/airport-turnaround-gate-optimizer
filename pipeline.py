import numpy as np
import pandas as pd
import lightgbm as lgb
from ortools.sat.python import cp_model
from sklearn.model_selection import train_test_split

def train_tat_model():
    """Generates synthetic airport turnaround data and trains a LightGBM regressor."""
    np.random.seed(42)
    n = 3500
    
    classes = ["Narrow-body", "Wide-body", "Regional-Jet"]
    data = {
        "aircraft_class": np.random.choice(classes, size=n, p=[0.7, 0.15, 0.15]),
        "arr_delay_mins": np.random.exponential(scale=15, size=n) - 5,
        "taxi_in_mins": np.random.normal(loc=9, scale=3, size=n).clip(3, 30),
        "hour_of_day": np.random.randint(6, 23, size=n),
        "apron_congestion_index": np.random.randint(5, 30, size=n)
    }
    df = pd.DataFrame(data)
    
    # Ground truth physics & operational drag
    base = df["aircraft_class"].map({"Regional-Jet": 30, "Narrow-body": 45, "Wide-body": 95})
    congestion = df["apron_congestion_index"] * 0.55
    delay_comp = np.where(df["arr_delay_mins"] > 15, -0.2 * df["arr_delay_mins"], 0)
    noise = np.random.normal(0, 3, size=n)
    
    df["actual_turnaround"] = (base + congestion + delay_comp + noise).clip(25, 180)
    df["aircraft_class"] = df["aircraft_class"].astype("category")
    
    X = df[["aircraft_class", "arr_delay_mins", "taxi_in_mins", "hour_of_day", "apron_congestion_index"]]
    y = df["actual_turnaround"]
    
    model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, verbose=-1)
    model.fit(X, y)
    return model

def optimize_gates(flights, gates, buffer_mins=15):
    """Assigns flights to gates avoiding physical overlap and minimizing remote stands."""
    model = cp_model.CpModel()
    n_f, n_g = len(flights), len(gates)
    
    # Decision Variable: x[f, g] = 1 if flight f at gate g
    x = {(f, g): model.NewBoolVar(f"x_{f}_{g}") for f in range(n_f) for g in range(n_g)}
    
    # 1. Unique gate per flight
    for f in range(n_f):
        model.AddExactlyOne(x[f, g] for g in range(n_g))
        
    # 2. Aircraft compatibility
    for f_idx, fl in enumerate(flights):
        for g_idx, gt in enumerate(gates):
            if fl["aircraft_class"] == "Wide-body" and gt["allowed_class"] != "Wide-body":
                model.Add(x[f_idx, g_idx] == 0)
                
    # 3. No temporal overlap on the same gate + Safety buffer
    for g in range(n_g):
        for i in range(n_f):
            for j in range(i + 1, n_f):
                f1, f2 = flights[i], flights[j]
                start_1, end_1 = f1["arrival_time"], f1["arrival_time"] + f1["predicted_tat"] + buffer_mins
                start_2, end_2 = f2["arrival_time"], f2["arrival_time"] + f2["predicted_tat"] + buffer_mins
                
                if max(start_1, start_2) < min(end_1, end_2):
                    model.Add(x[i, g] + x[j, g] <= 1)
                    
    # Objective: Minimize remote apron usage
    penalties = [x[f, g] * (0 if gates[g]["is_contact"] else 50) for f in range(n_f) for g in range(n_g)]
    model.Minimize(sum(penalties))
    
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    
    results = []
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for f in range(n_f):
            for g in range(n_g):
                if solver.Value(x[f, g]) == 1:
                    results.append({
                        "Flight": flights[f]["flight_id"],
                        "Class": flights[f]["aircraft_class"],
                        "Arrival": flights[f]["arrival_time"],
                        "TAT": flights[f]["predicted_tat"],
                        "Departure": flights[f]["arrival_time"] + flights[f]["predicted_tat"],
                        "Gate": gates[g]["gate_id"],
                        "Type": "Contact Bridge" if gates[g]["is_contact"] else "Remote Stand"
                    })
    return pd.DataFrame(results)
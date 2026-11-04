import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from ortools.sat.python import cp_model
import plotly.express as px

st.set_page_config(page_title="AOCC Gate Allocation Engine", layout="wide")
st.title("🛫 Airport Operations Control Centre (AOCC)")
st.subheader("Turnaround Prediction & Dynamic Gate Conflict Engine")

# 1. Train Model (Cached so it runs once)
@st.cache_resource
def get_trained_model():
    np.random.seed(42)
    n = 2000
    classes = ["Narrow-body", "Wide-body", "Regional-Jet"]
    data = {
        "aircraft_class": np.random.choice(classes, size=n, p=[0.7, 0.15, 0.15]),
        "arr_delay_mins": np.random.exponential(scale=15, size=n) - 5,
        "taxi_in_mins": np.random.normal(loc=9, scale=3, size=n).clip(3, 30),
        "hour_of_day": np.random.randint(6, 23, size=n),
        "apron_congestion_index": np.random.randint(5, 30, size=n)
    }
    df = pd.DataFrame(data)
    base = df["aircraft_class"].map({"Regional-Jet": 30, "Narrow-body": 45, "Wide-body": 95})
    df["actual_turnaround"] = (base + df["apron_congestion_index"] * 0.5 + np.random.normal(0, 3, size=n)).clip(25, 180)
    df["aircraft_class"] = df["aircraft_class"].astype("category")
    
    X = df[["aircraft_class", "arr_delay_mins", "taxi_in_mins", "hour_of_day", "apron_congestion_index"]]
    y = df["actual_turnaround"]
    
    m = lgb.LGBMRegressor(n_estimators=80, learning_rate=0.05, max_depth=4, random_state=42, verbose=-1)
    m.fit(X, y)
    return m

model = get_trained_model()

# 2. Sidebar Parameters
st.sidebar.header("Ramp Parameters")
congestion = st.sidebar.slider("Apron Congestion Index", 5, 35, 12)
buffer_mins = st.sidebar.slider("Safety Buffer (mins)", 10, 30, 15)

st.sidebar.markdown("---")
st.sidebar.header("Inject Flight Delays (mins)")

flights_input = [
    {"flight_id": "AI-101", "aircraft_class": "Wide-body", "base_arr": 10, "taxi_in": 8, "hour": 8},
    {"flight_id": "6E-205", "aircraft_class": "Narrow-body", "base_arr": 20, "taxi_in": 6, "hour": 8},
    {"flight_id": "SG-302", "aircraft_class": "Narrow-body", "base_arr": 25, "taxi_in": 10, "hour": 8},
    {"flight_id": "AI-409", "aircraft_class": "Narrow-body", "base_arr": 45, "taxi_in": 5, "hour": 8},
    {"flight_id": "IX-551", "aircraft_class": "Regional-Jet", "base_arr": 50, "taxi_in": 5, "hour": 8},
]

live_flights = []
for f in flights_input:
    d = st.sidebar.number_input(f"{f['flight_id']} Delay", min_value=-15, max_value=120, value=0, step=5)
    live_flights.append({
        "flight_id": f["flight_id"],
        "aircraft_class": f["aircraft_class"],
        "arr_delay_mins": d,
        "taxi_in_mins": f["taxi_in"],
        "hour_of_day": f["hour"],
        "apron_congestion_index": congestion,
        "arrival_time": f["base_arr"] + d,
        "delay": d
    })

# 3. Predict Turnaround Times
df_in = pd.DataFrame(live_flights)
df_in["aircraft_class"] = df_in["aircraft_class"].astype("category")
preds = model.predict(df_in[["aircraft_class", "arr_delay_mins", "taxi_in_mins", "hour_of_day", "apron_congestion_index"]])

for i, fl in enumerate(live_flights):
    fl["predicted_tat"] = int(np.round(preds[i]))

# 4. OR-Tools Gate Optimization
gates = [
    {"gate_id": "Gate-01 (Heavy)", "allowed_class": "Wide-body", "is_contact": True},
    {"gate_id": "Gate-02 (Contact)", "allowed_class": "Narrow-body", "is_contact": True},
    {"gate_id": "Gate-03 (Contact)", "allowed_class": "Narrow-body", "is_contact": True},
    {"gate_id": "Stand-R1 (Remote)", "allowed_class": "Narrow-body", "is_contact": False},
]

cp = cp_model.CpModel()
n_f, n_g = len(live_flights), len(gates)
x = {(i, g): cp.NewBoolVar(f"x_{i}_{g}") for i in range(n_f) for g in range(n_g)}

for i in range(n_f):
    cp.AddExactlyOne(x[i, g] for g in range(n_g))

for i, fl in enumerate(live_flights):
    for g, gt in enumerate(gates):
        if fl["aircraft_class"] == "Wide-body" and gt["allowed_class"] != "Wide-body":
            cp.Add(x[i, g] == 0)

for g in range(n_g):
    for i in range(n_f):
        for j in range(i + 1, n_f):
            f1, f2 = live_flights[i], live_flights[j]
            s1, e1 = f1["arrival_time"], f1["arrival_time"] + f1["predicted_tat"] + buffer_mins
            s2, e2 = f2["arrival_time"], f2["arrival_time"] + f2["predicted_tat"] + buffer_mins
            if max(s1, s2) < min(e1, e2):
                cp.Add(x[i, g] + x[j, g] <= 1)

penalties = [x[i, g] * (0 if gates[g]["is_contact"] else 50) for i in range(n_f) for g in range(n_g)]
cp.Minimize(sum(penalties))

solver = cp_model.CpSolver()
status = solver.Solve(cp)

results = []
if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    for i in range(n_f):
        for g in range(n_g):
            if solver.Value(x[i, g]) == 1:
                results.append({
                    "Flight": live_flights[i]["flight_id"],
                    "Class": live_flights[i]["aircraft_class"],
                    "Arrival": live_flights[i]["arrival_time"],
                    "TAT": live_flights[i]["predicted_tat"],
                    "Departure": live_flights[i]["arrival_time"] + live_flights[i]["predicted_tat"],
                    "Gate": gates[g]["gate_id"],
                    "Type": "Contact Bridge" if gates[g]["is_contact"] else "Remote Stand",
                    "Delay": live_flights[i]["delay"]
                })

df_res = pd.DataFrame(results)

# 5. Render Timeline and Metrics
if not df_res.empty:
    base_t = pd.Timestamp("2026-11-03 08:00:00")
    df_res["Start"] = df_res["Arrival"].apply(lambda t: base_t + pd.Timedelta(minutes=t))
    df_res["End"] = df_res["Departure"].apply(lambda t: base_t + pd.Timedelta(minutes=t))
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### Gate Allocation Schedule")
        fig = px.timeline(
            df_res,
            x_start="Start",
            x_end="End",
            y="Gate",
            color="Class",
            text="Flight",
            hover_data=["TAT", "Type", "Delay"]
        )
        fig.update_yaxes(categoryorder="total ascending")
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("### Apron KPIs")
        st.metric("Total Flights", len(df_res))
        st.metric("Remote Stand Overflows", (df_res["Type"] == "Remote Stand").sum())
        st.dataframe(df_res[["Flight", "Gate", "TAT", "Delay"]], use_container_width=True)
else:
    st.error("No valid gate assignment possible with these delays.")
import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from ortools.sat.python import cp_model
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
import json

# --- 1. System Configuration & Dark Aviation HUD Styling ---
st.set_page_config(
    page_title="AeroTwin AOCC | Digital Twin & Gate Engine",
    page_icon="🛫",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0b0f19; }
    .stMetric {
        background: rgba(16, 24, 39, 0.85);
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 8px;
        padding: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .hud-header {
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #f3f4f6;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .hud-sub {
        color: #9ca3af;
        font-size: 0.9rem;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. Machine Learning: Turnaround Prediction Model ---
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

# --- 3. Global Operational Control Center (Sidebar) ---
st.sidebar.markdown("### 🎛️ Disruption & Environment")

weather_profile = st.sidebar.selectbox(
    "Weather Scenario (A-CDM Trigger)",
    ["Clear Skies (Nominal)", "Monsoon Downpour (+25% TAT)", "Severe Fog / CAT III (+40% TAT)"]
)

if "Clear" in weather_profile:
    weather_multiplier = 1.0
    weather_taxi_add = 0
    weather_color = "#10b981"
elif "Monsoon" in weather_profile:
    weather_multiplier = 1.25
    weather_taxi_add = 5
    weather_color = "#f59e0b"
else:
    weather_multiplier = 1.40
    weather_taxi_add = 10
    weather_color = "#ef4444"

congestion = st.sidebar.slider("Apron Congestion Index", 5, 35, 14)
buffer_mins = st.sidebar.slider("Safety Separation Buffer (mins)", 10, 30, 15)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⏱️ Inbound Flight Delays")

flight_meta = [
    {"flight_id": "AI-101", "class": "Wide-body", "origin": "London (LHR)", "lat": 51.47, "lon": -0.45, "base_arr": 10, "taxi": 8, "hour": 8, "seats": 296, "fare": 450},
    {"flight_id": "6E-205", "class": "Narrow-body", "origin": "Delhi (DEL)", "lat": 28.55, "lon": 77.10, "base_arr": 20, "taxi": 6, "hour": 8, "seats": 180, "fare": 95},
    {"flight_id": "SG-302", "class": "Narrow-body", "origin": "Mumbai (BOM)", "lat": 19.08, "lon": 72.87, "base_arr": 25, "taxi": 10, "hour": 8, "seats": 186, "fare": 80},
    {"flight_id": "AI-409", "class": "Narrow-body", "origin": "Bengaluru (BLR)", "lat": 13.19, "lon": 77.70, "base_arr": 45, "taxi": 5, "hour": 8, "seats": 160, "fare": 70},
    {"flight_id": "IX-551", "class": "Regional-Jet", "origin": "Dubai (DXB)", "lat": 25.25, "lon": 55.36, "base_arr": 50, "taxi": 5, "hour": 8, "seats": 78, "fare": 160},
]

live_flights = []
for f in flight_meta:
    d = st.sidebar.number_input(f"{f['flight_id']} ({f['class']}) Delay", min_value=-15, max_value=120, value=0, step=5)
    live_flights.append({
        "flight_id": f["flight_id"],
        "aircraft_class": f["class"],
        "origin": f["origin"],
        "origin_lat": f["lat"],
        "origin_lon": f["lon"],
        "arr_delay_mins": d,
        "taxi_in_mins": f["taxi"] + weather_taxi_add,
        "hour_of_day": f["hour"],
        "apron_congestion_index": congestion,
        "arrival_time": f["base_arr"] + d,
        "delay": d,
        "total_seats": f["seats"],
        "base_fare": f["fare"]
    })

# --- 4. Turnaround ML Inference & Weather Impact ---
df_in = pd.DataFrame(live_flights)
df_in["aircraft_class"] = df_in["aircraft_class"].astype("category")
preds = model.predict(df_in[["aircraft_class", "arr_delay_mins", "taxi_in_mins", "hour_of_day", "apron_congestion_index"]])

for i, fl in enumerate(live_flights):
    raw_tat = preds[i] * weather_multiplier
    fl["predicted_tat"] = int(np.round(raw_tat))

# --- 5. OR-Tools Optimization Engine ---
gates = [
    {"gate_id": "Gate-01 (Heavy)", "allowed_class": "Wide-body", "is_contact": True, "x": -25, "z": -10},
    {"gate_id": "Gate-02 (Contact)", "allowed_class": "Narrow-body", "is_contact": True, "x": -5, "z": -10},
    {"gate_id": "Gate-03 (Contact)", "allowed_class": "Narrow-body", "is_contact": True, "x": 15, "z": -10},
    {"gate_id": "Stand-R1 (Remote)", "allowed_class": "Narrow-body", "is_contact": False, "x": -15, "z": 25},
    {"gate_id": "Stand-R2 (Remote)", "allowed_class": "Narrow-body", "is_contact": False, "x": 10, "z": 25},
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
solver.parameters.max_time_in_seconds = 2.0
solver.parameters.num_search_workers = 1
status = solver.Solve(cp)

results = []
if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    for i in range(n_f):
        for g in range(n_g):
            if solver.Value(x[i, g]) == 1:
                results.append({
                    "Flight": live_flights[i]["flight_id"],
                    "Class": live_flights[i]["aircraft_class"],
                    "Origin": live_flights[i]["origin"],
                    "Origin_Lat": live_flights[i]["origin_lat"],
                    "Origin_Lon": live_flights[i]["origin_lon"],
                    "Arrival": live_flights[i]["arrival_time"],
                    "TAT": live_flights[i]["predicted_tat"],
                    "Departure": live_flights[i]["arrival_time"] + live_flights[i]["predicted_tat"],
                    "Gate": gates[g]["gate_id"],
                    "Gate_X": gates[g]["x"],
                    "Gate_Z": gates[g]["z"],
                    "Type": "Contact Bridge" if gates[g]["is_contact"] else "Remote Stand",
                    "Delay": live_flights[i]["delay"],
                    "BaseFare": live_flights[i]["base_fare"],
                    "TotalSeats": live_flights[i]["total_seats"]
                })

df_res = pd.DataFrame(results)

# --- 6. Navigation Suite Tabs ---
st.markdown('<h1 class="hud-header">🛫 AeroTwin AOCC | Digital Twin Platform</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="hud-sub">Live Operations Control • Meteorological Profile: <span style="color:{weather_color};font-weight:600;">{weather_profile}</span> • Solver: CP-SAT</p>', unsafe_allow_html=True)

tab_deck, tab_3d, tab_map, tab_comm, tab_gantt = st.tabs([
    "📊 Executive HUD",
    "🏢 3D Apron Twin",
    "🌐 Inbound Route Network",
    "🏷️ Revenue Clearance",
    "📈 Gantt Schedule"
])

# ==========================================
# TAB 1: EXECUTIVE HUD
# ==========================================
with tab_deck:
    remote_overflows = int((df_res["Type"] == "Remote Stand").sum()) if not df_res.empty else 0
    delayed_flights = int((df_res["Delay"] > 0).sum()) if not df_res.empty else 0
    avg_tat = int(df_res["TAT"].mean()) if not df_res.empty else 0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Movements", len(df_res), "Peak Bank")
    col2.metric("Mean Predicted TAT", f"{avg_tat} mins", f"{'+' if weather_multiplier > 1.0 else ''}{int((weather_multiplier - 1.0)*100)}% Weather impact")
    col3.metric("Remote Overflows", remote_overflows, "Bus transfers needed" if remote_overflows > 0 else "All Contact Gates", delta_color="inverse")
    col4.metric("Network Delayed", delayed_flights, f"{len(df_res) - delayed_flights} On-Time", delta_color="inverse")

    st.markdown("---")
    c_left, c_right = st.columns([2, 1])
    with c_left:
        st.markdown("### Operational Stand Allocations")
        st.dataframe(
            df_res[["Flight", "Class", "Origin", "Gate", "Type", "Arrival", "TAT", "Departure", "Delay"]],
            use_container_width=True
        )
    with c_right:
        st.markdown("### Gate Class Capacity Breakdown")
        if not df_res.empty:
            pie = px.pie(df_res, names="Type", color="Type", color_discrete_map={"Contact Bridge": "#3b82f6", "Remote Stand": "#ef4444"}, hole=0.5)
            pie.update_layout(margin=dict(l=0, r=0, t=10, b=10), height=260, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#fff"))
            st.plotly_chart(pie, use_container_width=True)

# ==========================================
# TAB 2: 3D APRON DIGITAL TWIN (WebGL / Three.js)
# ==========================================
with tab_3d:
    st.markdown("### Real-Time Airside Digital Twin (Interactive 3D)")
    st.caption("Drag to rotate • Scroll to zoom • Right-click to pan. Aircraft geometry, gate positions, and alert halos react to CP-SAT assignments.")
    
    twin_payload = []
    if not df_res.empty:
        for _, row in df_res.iterrows():
            twin_payload.append({
                "flight": row["Flight"],
                "gate": row["Gate"],
                "class": row["Class"],
                "x": row["Gate_X"],
                "z": row["Gate_Z"],
                "tat": row["TAT"],
                "delay": row["Delay"],
                "is_remote": row["Type"] == "Remote Stand"
            })
    
    twin_json = json.dumps(twin_payload)
    
    threejs_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        body {{ margin: 0; overflow: hidden; background-color: #0b0f19; font-family: -apple-system, sans-serif; }}
        #canvas-container {{ width: 100vw; height: 500px; }}
      </style>
      <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
      <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    </head>
    <body>
      <div id="canvas-container"></div>
      <script>
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0e131f);
        scene.fog = new THREE.FogExp2(0x0e131f, 0.008);

        const camera = new THREE.PerspectiveCamera(45, window.innerWidth / 500, 1, 1000);
        camera.position.set(0, 45, 65);

        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, 500);
        renderer.shadowMap.enabled = true;
        document.getElementById('canvas-container').appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.maxPolarAngle = Math.PI / 2.1;

        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        const dirLight = new THREE.DirectionalLight(0x60a5fa, 0.9);
        dirLight.position.set(30, 50, 20);
        dirLight.castShadow = true;
        scene.add(dirLight);

        const tarmacGeo = new THREE.PlaneGeometry(160, 120);
        const tarmacMat = new THREE.MeshStandardMaterial({{ color: 0x1f2937, roughness: 0.8 }});
        const tarmac = new THREE.Mesh(tarmacGeo, tarmacMat);
        tarmac.rotation.x = -Math.PI / 2;
        tarmac.receiveShadow = true;
        scene.add(tarmac);

        const lineMat = new THREE.LineBasicMaterial({{ color: 0xfbbf24, linewidth: 2 }});
        const lineGeo = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(-70, 0.05, 10), new THREE.Vector3(70, 0.05, 10),
            new THREE.Vector3(-70, 0.05, 38), new THREE.Vector3(70, 0.05, 38)
        ]);
        scene.add(new THREE.LineSegments(lineGeo, lineMat));

        const termGeo = new THREE.BoxGeometry(75, 10, 14);
        const termMat = new THREE.MeshStandardMaterial({{ color: 0x374151, roughness: 0.3, metalness: 0.5 }});
        const terminal = new THREE.Mesh(termGeo, termMat);
        terminal.position.set(-5, 5, -22);
        terminal.castShadow = true;
        scene.add(terminal);

        [-25, -5, 15].forEach(x => {{
            const jbGeo = new THREE.BoxGeometry(2, 2.5, 9);
            const jbMat = new THREE.MeshStandardMaterial({{ color: 0x4b5563 }});
            const jb = new THREE.Mesh(jbGeo, jbMat);
            jb.position.set(x, 2, -12);
            scene.add(jb);
        }});

        const flightData = {twin_json};

        flightData.forEach(f => {{
            let ringColor = 0x10b981;
            if (f.delay > 0 && f.delay <= 15) ringColor = 0xf59e0b;
            if (f.delay > 15 || f.is_remote) ringColor = 0xef4444;

            const ringGeo = new THREE.RingGeometry(5, 6.2, 32);
            const ringMat = new THREE.MeshBasicMaterial({{ color: ringColor, side: THREE.DoubleSide }});
            const ring = new THREE.Mesh(ringGeo, ringMat);
            ring.rotation.x = -Math.PI / 2;
            ring.position.set(f.x, 0.08, f.z);
            scene.add(ring);

            const scale = (f.class === "Wide-body") ? 1.5 : 1.0;
            const fuseGeo = new THREE.CylinderGeometry(0.7 * scale, 0.8 * scale, 8 * scale, 12);
            const fuseMat = new THREE.MeshStandardMaterial({{ color: 0xe5e7eb, metalness: 0.2, roughness: 0.4 }});
            const fuselage = new THREE.Mesh(fuseGeo, fuseMat);
            fuselage.rotation.x = Math.PI / 2;
            fuselage.position.set(f.x, 1.2 * scale, f.z);

            const wingGeo = new THREE.BoxGeometry(9 * scale, 0.15, 2.2 * scale);
            const wingMat = new THREE.MeshStandardMaterial({{ color: 0x9ca3af }});
            const wings = new THREE.Mesh(wingGeo, wingMat);
            wings.position.set(0, 0, 0.5 * scale);
            fuselage.add(wings);

            const tailGeo = new THREE.BoxGeometry(0.2, 2 * scale, 1.5 * scale);
            const tailMat = new THREE.MeshStandardMaterial({{ color: 0x3b82f6 }});
            const tail = new THREE.Mesh(tailGeo, tailMat);
            tail.position.set(0, 1.2 * scale, -3.2 * scale);
            fuselage.add(tail);

            scene.add(fuselage);
        }});

        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();
      </script>
    </body>
    </html>
    """
    components.html(threejs_code, height=520)

# ==========================================
# TAB 3: INBOUND ROUTE NETWORK (COMPATIBLE MAP)
# ==========================================
with tab_map:
    st.markdown("### Macro Flight Trajectories & Inbound Radar")
    st.caption("Visualizing inbound city connections converging on the central hub.")
    
    hub_lat, hub_lon = 10.15, 76.40  # Destination Hub
    
    # Use standard scatter_geo which works across all Plotly versions without mapbox token dependencies
    route_rows = []
    for _, r in df_res.iterrows():
        status_label = "On-Time" if r["Delay"] == 0 else f"+{r['Delay']}m Delayed"
        route_rows.append({
            "Flight": r["Flight"],
            "Origin": r["Origin"],
            "Origin_Lat": r["Origin_Lat"],
            "Origin_Lon": r["Origin_Lon"],
            "Dest_Lat": hub_lat,
            "Dest_Lon": hub_lon,
            "Gate": r["Gate"],
            "Status": status_label,
            "Color": "#10b981" if r["Delay"] == 0 else "#ef4444"
        })
        
    df_routes = pd.DataFrame(route_rows)
    
    fig_map = go.Figure()
    
    # Plot incoming paths as geodesic arcs
    for _, row in df_routes.iterrows():
        fig_map.add_trace(go.Scattergeo(
            lon=[row["Origin_Lon"], row["Dest_Lon"]],
            lat=[row["Origin_Lat"], row["Dest_Lat"]],
            mode="lines",
            line=dict(width=2.5, color=row["Color"]),
            opacity=0.8,
            hoverinfo="none",
            showlegend=False
        ))
        
    # Plot origin city nodes
    fig_map.add_trace(go.Scattergeo(
        lon=df_routes["Origin_Lon"],
        lat=df_routes["Origin_Lat"],
        mode="markers+text",
        text=df_routes["Flight"],
        textposition="top center",
        marker=dict(size=8, color="#60a5fa", symbol="circle"),
        hoverinfo="text",
        hovertext=[f"{r['Flight']} ({r['Origin']})<br>Assigned Stand: {r['Gate']}<br>Status: {r['Status']}" for _, r in df_routes.iterrows()],
        name="Inbound Flights"
    ))
    
    # Plot Central Airport Hub
    fig_map.add_trace(go.Scattergeo(
        lon=[hub_lon],
        lat=[hub_lat],
        mode="markers+text",
        text=["HUB AOCC"],
        textposition="bottom center",
        marker=dict(size=14, color="#3b82f6", symbol="star"),
        name="Destination Hub"
    ))
    
    fig_map.update_geos(
        projection_type="natural earth",
        showland=True,
        landcolor="#1e293b",
        oceancolor="#0b0f19",
        showocean=True,
        showlakes=True,
        lakecolor="#0b0f19",
        showcountries=True,
        countrycolor="#334155",
        bgcolor="rgba(0,0,0,0)",
        fitbounds="locations"
    )
    
    fig_map.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f3f4f6"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_map, use_container_width=True)

# ==========================================
# TAB 4: COMMERCIAL SEAT CLEARANCE
# ==========================================
with tab_comm:
    st.markdown("### Standby & Dynamic Last-Minute Ancillary Recovery")
    st.caption("Algorithmic clearance pricing for no-show inventory triggered when turnaround departure is within 45 minutes.")
    
    clearance_items = []
    for _, r in df_res.iterrows():
        if r["TAT"] > 0:
            unsold_seats = int(np.clip(r["TotalSeats"] * 0.03 + (r["Delay"] // 15), 2, 8))
            discount = int(np.clip(25 + (r["Delay"] * 0.5), 20, 60))
            offer_fare = int(r["BaseFare"] * (1 - discount / 100))
            salvaged = offer_fare * unsold_seats
            clearance_items.append({
                "Flight": r["Flight"],
                "Origin": r["Origin"],
                "Assigned Stand": r["Gate"],
                "Unsold / No-Show Seats": unsold_seats,
                "Original Fare ($)": r["BaseFare"],
                "Flash Clearance Fare ($)": offer_fare,
                "Discount": f"{discount}% OFF",
                "Salvaged Revenue ($)": salvaged
            })
            
    df_comm = pd.DataFrame(clearance_items)
    
    c_m1, c_m2, c_m3 = st.columns(3)
    c_m1.metric("Clearance Eligible Flights", len(df_comm))
    c_m2.metric("Total Standby Seats Released", int(df_comm["Unsold / No-Show Seats"].sum()))
    c_m3.metric("Projected Recovered Revenue", f"${df_comm['Salvaged Revenue ($)'].sum():,}")
    
    st.markdown("---")
    st.dataframe(df_comm, use_container_width=True)

# ==========================================
# TAB 5: GANTT SCHEDULE & SOLVER TELEMETRY
# ==========================================
with tab_gantt:
    st.markdown("### Gate Assignment Gantt Schedule")
    if not df_res.empty:
        base_t = pd.Timestamp("2026-11-04 08:00:00")
        df_res["Start"] = df_res["Arrival"].apply(lambda t: base_t + pd.Timedelta(minutes=t))
        df_res["End"] = df_res["Departure"].apply(lambda t: base_t + pd.Timedelta(minutes=t))
        
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
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No valid gate assignment possible.")
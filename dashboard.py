import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from ortools.sat.python import cp_model
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
import json
import base64
import os
import time
import io
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="AEROT | AOCC & Commercial Intelligence",
    page_icon="🛫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Base64 Logo Helper ---
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return ""

# --- Session State Initializations ---
if "splash_complete" not in st.session_state:
    st.session_state.splash_complete = False

if "active_page" not in st.session_state:
    st.session_state.active_page = "Executive Command Deck"

# ==============================================================================
# SPLASH SCREEN CONTROLLER
# ==============================================================================
if not st.session_state.splash_complete:
    logo_file = "logo.png" if os.path.exists("logo.png") else ("logo.jpg" if os.path.exists("logo.jpg") else "")
    bg_file = "splash_bg.jpg" if os.path.exists("splash_bg.jpg") else ""
    
    logo_b64 = get_base64_image(logo_file) if logo_file else ""
    bg_b64 = get_base64_image(bg_file) if bg_file else ""

    bg_css_rule = (
        f"background: linear-gradient(180deg, rgba(11, 15, 25, 0.4) 0%, rgba(11, 15, 25, 0.85) 100%), "
        f"url('data:image/jpeg;base64,{bg_b64}') no-repeat center center fixed !important; "
        f"background-size: cover !important;"
    ) if bg_b64 else "background: radial-gradient(circle at 50% 35%, #1e293b 0%, #0b0f19 85%) !important;"

    splash_css = f"""
    <style>
        header, footer, [data-testid="stSidebar"], [data-testid="stToolbar"] {{
            display: none !important;
        }}
        .stApp {{
            {bg_css_rule}
            overflow: hidden;
        }}
        .sky-container {{
            position: fixed;
            top: 0; left: 0; width: 100vw; height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            z-index: 999999;
            backdrop-filter: blur(2px);
        }}
        .radar-sweep {{
            position: absolute;
            width: 580px;
            height: 580px;
            border-radius: 50%;
            border: 1px solid rgba(56, 189, 248, 0.15);
            box-shadow: inset 0 0 50px rgba(56, 189, 248, 0.08);
            background: conic-gradient(from 0deg, rgba(56, 189, 248, 0.2) 0deg, transparent 70deg, transparent 360deg);
            animation: radarRotate 4.5s linear infinite;
            pointer-events: none;
        }}
        .radar-ring-inner {{
            position: absolute;
            width: 340px;
            height: 340px;
            border-radius: 50%;
            border: 1px dashed rgba(56, 189, 248, 0.25);
            pointer-events: none;
        }}
        .brand-box {{
            position: relative;
            z-index: 10;
            display: flex;
            flex-direction: column;
            align-items: center;
            animation: heroEntrance 1.2s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }}
        /* Transparent Logo: inverts dark font and removes background border/boxes */
        .brand-logo {{
            width: 320px;
            max-width: 82vw;
            filter: invert(1) brightness(1.3) contrast(1.15) drop-shadow(0 0 25px rgba(56, 189, 248, 0.45));
            mix-blend-mode: screen;
            background: transparent !important;
            margin-bottom: 8px;
        }}
        .brand-tagline {{
            color: #e2e8f0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            font-size: 0.85rem;
            letter-spacing: 0.35em;
            text-transform: uppercase;
            margin-top: 14px;
            font-weight: 500;
            text-shadow: 0 2px 10px rgba(0,0,0,0.8);
        }}
        .loader-track {{
            margin-top: 36px;
            width: 260px;
            height: 4px;
            background: rgba(255, 255, 255, 0.15);
            border-radius: 999px;
            overflow: hidden;
            position: relative;
            box-shadow: 0 0 10px rgba(0,0,0,0.5);
        }}
        .loader-bar {{
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, #38bdf8, #818cf8, #38bdf8);
            box-shadow: 0 0 15px #38bdf8;
            border-radius: 999px;
            animation: fillBar 2.3s cubic-bezier(0.4, 0, 0.2, 1) forwards;
        }}
        @keyframes radarRotate {{
            from {{ transform: rotate(0deg); }}
            to {{ transform: rotate(360deg); }}
        }}
        @keyframes heroEntrance {{
            0% {{ opacity: 0; transform: scale(0.92) translateY(18px); }}
            100% {{ opacity: 1; transform: scale(1) translateY(0); }}
        }}
        @keyframes fillBar {{
            0% {{ transform: translateX(-100%); }}
            60% {{ transform: translateX(-25%); }}
            100% {{ transform: translateX(0%); }}
        }}
    </style>
    """
    st.markdown(splash_css, unsafe_allow_html=True)

    logo_tag = f'<img src="data:image/png;base64,{logo_b64}" class="brand-logo" alt="AEROT" />' if logo_b64 else '<h1 style="color:#f8fafc; font-size:3rem; letter-spacing:0.15em;">AEROT</h1>'

    splash_html = f"""
    <div class="sky-container">
        <div class="radar-sweep"></div>
        <div class="radar-ring-inner"></div>
        <div class="brand-box">
            {logo_tag}
            <div class="brand-tagline">Initializing AOCC & Digital Twin Telemetry...</div>
            <div class="loader-track">
                <div class="loader-bar"></div>
            </div>
        </div>
    </div>
    """
    st.markdown(splash_html, unsafe_allow_html=True)

    time.sleep(2.4)
    st.session_state.splash_complete = True
    st.rerun()

# ==============================================================================
# MAIN AOCC APPLICATION (APPEARS AFTER SPLASH)
# ==============================================================================

# Executive Dark Theme Styling
st.markdown("""
<style>
    .stApp { background-color: #0b0f19; color: #f3f4f6; }
    div[data-testid="stSidebar"] { background-color: #0d1322; border-right: 1px solid rgba(59, 130, 246, 0.15); }
    .hud-title {
        font-family: 'Inter', -apple-system, sans-serif;
        color: #f8fafc;
        font-weight: 800;
        font-size: 2.1rem;
        margin-bottom: 2px;
    }
    .hud-sub {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 15px;
    }
    .kpi-title { font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }
    .kpi-val { font-size: 1.8rem; font-weight: 700; color: #f8fafc; margin: 4px 0; }
    .kpi-sub { font-size: 0.78rem; color: #38bdf8; }
</style>
""", unsafe_allow_html=True)

# 1. Machine Learning: Turnaround Prediction Model
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

# 2. Sidebar Navigation & Global Controls
st.sidebar.markdown("## 🧭 AEROT Navigation")

nav_options = [
    "Executive Command Deck",
    "3D Airside Digital Twin",
    "Inbound Route Network",
    "Commercial Seat Clearance",
    "Gate Optimization Telemetry"
]

selected_page = st.sidebar.radio(
    "Go To Module",
    nav_options,
    index=nav_options.index(st.session_state.active_page)
)
st.session_state.active_page = selected_page

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ Live Disruption Controls")

weather_profile = st.sidebar.selectbox(
    "A-CDM Meteorological Profile",
    ["Clear Skies (Nominal)", "Monsoon Downpour (+25% TAT)", "Severe Fog / CAT III (+40% TAT)"]
)

if "Clear" in weather_profile:
    weather_multiplier = 1.0
    weather_taxi_add = 0
    weather_badge = "🟢 NOMINAL"
elif "Monsoon" in weather_profile:
    weather_multiplier = 1.25
    weather_taxi_add = 5
    weather_badge = "🟡 ADVERSE WEATHER"
else:
    weather_multiplier = 1.40
    weather_taxi_add = 10
    weather_badge = "🔴 CAT III GROUND-STOP"

congestion = st.sidebar.slider("Apron Congestion Index", 5, 35, 14)
buffer_mins = st.sidebar.slider("Wake & Separation Buffer (mins)", 10, 30, 15)

st.sidebar.markdown("---")
with st.sidebar.expander("⏱️ Inbound Flight Delays", expanded=False):
    flight_meta = [
        {"flight_id": "AI-101", "class": "Wide-body", "origin": "London (LHR)", "lat": 51.47, "lon": -0.45, "base_arr": 10, "taxi": 8, "hour": 8, "seats": 296, "fare": 450},
        {"flight_id": "6E-205", "class": "Narrow-body", "origin": "Delhi (DEL)", "lat": 28.55, "lon": 77.10, "base_arr": 20, "taxi": 6, "hour": 8, "seats": 180, "fare": 95},
        {"flight_id": "SG-302", "class": "Narrow-body", "origin": "Mumbai (BOM)", "lat": 19.08, "lon": 72.87, "base_arr": 25, "taxi": 10, "hour": 8, "seats": 186, "fare": 80},
        {"flight_id": "AI-409", "class": "Narrow-body", "origin": "Bengaluru (BLR)", "lat": 13.19, "lon": 77.70, "base_arr": 45, "taxi": 5, "hour": 8, "seats": 160, "fare": 70},
        {"flight_id": "IX-551", "class": "Regional-Jet", "origin": "Dubai (DXB)", "lat": 25.25, "lon": 55.36, "base_arr": 50, "taxi": 5, "hour": 8, "seats": 78, "fare": 160},
    ]

    live_flights = []
    for f in flight_meta:
        d = st.number_input(f"{f['flight_id']} ({f['class']}) Delay", min_value=-15, max_value=120, value=0, step=5, key=f"d_{f['flight_id']}")
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

# 3. Model Inference & OR-Tools Optimization
df_in = pd.DataFrame(live_flights)
df_in["aircraft_class"] = df_in["aircraft_class"].astype("category")
preds = model.predict(df_in[["aircraft_class", "arr_delay_mins", "taxi_in_mins", "hour_of_day", "apron_congestion_index"]])

for i, fl in enumerate(live_flights):
    fl["predicted_tat"] = int(np.round(preds[i] * weather_multiplier))

gates = [
    {"gate_id": "Gate-01 (Heavy)", "allowed_class": "Wide-body", "is_contact": True, "x": -26, "z": -12},
    {"gate_id": "Gate-02 (Contact)", "allowed_class": "Narrow-body", "is_contact": True, "x": -6, "z": -12},
    {"gate_id": "Gate-03 (Contact)", "allowed_class": "Narrow-body", "is_contact": True, "x": 14, "z": -12},
    {"gate_id": "Stand-R1 (Remote)", "allowed_class": "Narrow-body", "is_contact": False, "x": -16, "z": 24},
    {"gate_id": "Stand-R2 (Remote)", "allowed_class": "Narrow-body", "is_contact": False, "x": 10, "z": 24},
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

penalties = [x[i, g] * (0 if gates[g]["is_contact"] else 60) for i in range(n_f) for g in range(n_g)]
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

# ==============================================================================
# SUB-MODULE ROUTING
# ==============================================================================
if st.session_state.active_page == "Executive Command Deck":
    st.markdown('<div class="hud-title">🛫 AEROT Executive Command Deck</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hud-sub">A-CDM Operational Telemetry • Weather State: <b>{weather_badge}</b></div>', unsafe_allow_html=True)

    remote_overflows = int((df_res["Type"] == "Remote Stand").sum()) if not df_res.empty else 0
    delayed_flights = int((df_res["Delay"] > 0).sum()) if not df_res.empty else 0
    avg_tat = int(df_res["TAT"].mean()) if not df_res.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="kpi-card"><div class="kpi-title">Active Aircraft</div><div class="kpi-val">{len(df_res)} Movements</div><div class="kpi-sub">Peak Inbound Wave</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi-card"><div class="kpi-title">Mean Predicted TAT</div><div class="kpi-val">{avg_tat} mins</div><div class="kpi-sub">LightGBM Machine Learning</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi-card"><div class="kpi-title">Remote Stand Demotions</div><div class="kpi-val" style="color:{"#ef4444" if remote_overflows > 0 else "#10b981"};">{remote_overflows} Flights</div><div class="kpi-sub">Bus operations required</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="kpi-card"><div class="kpi-title">Network OTP</div><div class="kpi-val" style="color:{"#10b981" if delayed_flights == 0 else "#f59e0b"};">{int(((len(df_res) - delayed_flights)/len(df_res))*100) if len(df_res) else 0}%</div><div class="kpi-sub">{delayed_flights} delayed arrival(s)</div></div>', unsafe_allow_html=True)

    st.markdown("### 🎛️ Operational Console Shortcuts")
    q1, q2, q3, q4 = st.columns(4)
    if q1.button("🏢 Launch 3D Digital Twin", use_container_width=True):
        st.session_state.active_page = "3D Airside Digital Twin"
        st.rerun()
    if q2.button("🌐 Open Inbound Radar Map", use_container_width=True):
        st.session_state.active_page = "Inbound Route Network"
        st.rerun()
    if q3.button("🏷️ Open Seat Clearance Feed", use_container_width=True):
        st.session_state.active_page = "Commercial Seat Clearance"
        st.rerun()
    if q4.button("📈 Open Solver Diagnostics", use_container_width=True):
        st.session_state.active_page = "Gate Optimization Telemetry"
        st.rerun()

    st.markdown("---")
    left_c, right_c = st.columns([3, 2])
    with left_c:
        st.markdown("#### 📋 Live Gate Allocation Manifest")
        st.dataframe(df_res[["Flight", "Class", "Origin", "Gate", "Type", "Arrival", "TAT", "Departure", "Delay"]], use_container_width=True)
    with right_c:
        st.markdown("#### 📊 Stand Utilization Mix")
        if not df_res.empty:
            pie = px.pie(df_res, names="Type", color="Type", color_discrete_map={"Contact Bridge": "#3b82f6", "Remote Stand": "#ef4444"}, hole=0.55)
            pie.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=260, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc"))
            st.plotly_chart(pie, use_container_width=True)

elif st.session_state.active_page == "3D Airside Digital Twin":
    st.markdown('<div class="hud-title">🏢 3D Airside Terminal Digital Twin</div>', unsafe_allow_html=True)
    st.markdown('<div class="hud-sub">Interactive spatial twin showing real-time aircraft models, jet bridges, dynamic ground hazard rings, and live TAT countdowns.</div>', unsafe_allow_html=True)

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
      <style>body {{ margin: 0; overflow: hidden; background-color: #0b0f19; font-family: sans-serif; }}</style>
      <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
      <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    </head>
    <body>
      <div id="canvas-container" style="width: 100vw; height: 580px;"></div>
      <script>
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0b0f19);
        scene.fog = new THREE.FogExp2(0x0b0f19, 0.007);
        const camera = new THREE.PerspectiveCamera(45, window.innerWidth / 580, 1, 1000);
        camera.position.set(0, 50, 75);
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, 580);
        renderer.shadowMap.enabled = true;
        document.getElementById('canvas-container').appendChild(renderer.domElement);
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;

        const ambientLight = new THREE.AmbientLight(0xffffff, 0.65);
        scene.add(ambientLight);
        const sun = new THREE.DirectionalLight(0x60a5fa, 0.9);
        sun.position.set(30, 60, 20);
        sun.castShadow = true;
        scene.add(sun);

        const tarmac = new THREE.Mesh(new THREE.PlaneGeometry(180, 130), new THREE.MeshStandardMaterial({{ color: 0x1e293b, roughness: 0.85 }}));
        tarmac.rotation.x = -Math.PI / 2;
        scene.add(tarmac);

        const term = new THREE.Mesh(new THREE.BoxGeometry(85, 12, 16), new THREE.MeshStandardMaterial({{ color: 0x334155, roughness: 0.2 }}));
        term.position.set(-6, 6, -24);
        scene.add(term);

        [-26, -6, 14].forEach(x => {{
            const jb = new THREE.Mesh(new THREE.BoxGeometry(2.4, 3, 10), new THREE.MeshStandardMaterial({{ color: 0x64748b }}));
            jb.position.set(x, 2.5, -13);
            scene.add(jb);
        }});

        const flights = {twin_json};
        flights.forEach(f => {{
            let ringColor = 0x10b981;
            if (f.delay > 0 && f.delay <= 15) ringColor = 0xf59e0b;
            if (f.delay > 15 || f.is_remote) ringColor = 0xef4444;

            const halo = new THREE.Mesh(new THREE.RingGeometry(5.2, 6.6, 36), new THREE.MeshBasicMaterial({{ color: ringColor, side: THREE.DoubleSide }}));
            halo.rotation.x = -Math.PI / 2;
            halo.position.set(f.x, 0.08, f.z);
            scene.add(halo);

            const scale = (f.class === "Wide-body") ? 1.6 : (f.class === "Regional-Jet" ? 0.85 : 1.1);
            const fuse = new THREE.Mesh(new THREE.CylinderGeometry(0.75 * scale, 0.85 * scale, 8 * scale, 16), new THREE.MeshStandardMaterial({{ color: 0xf1f5f9 }}));
            fuse.rotation.x = Math.PI / 2;
            fuse.position.set(f.x, 1.3 * scale, f.z);

            const wings = new THREE.Mesh(new THREE.BoxGeometry(9.2 * scale, 0.16, 2.2 * scale), new THREE.MeshStandardMaterial({{ color: 0x94a3b8 }}));
            wings.position.set(0, 0, 0.5 * scale);
            fuse.add(wings);

            const tail = new THREE.Mesh(new THREE.BoxGeometry(0.2, 2.2 * scale, 1.6 * scale), new THREE.MeshStandardMaterial({{ color: 0x2563eb }}));
            tail.position.set(0, 1.3 * scale, -3.2 * scale);
            fuse.add(tail);

            scene.add(fuse);
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
    components.html(threejs_code, height=600)

elif st.session_state.active_page == "Inbound Route Network":
    st.markdown('<div class="hud-title">🌐 Macro Air Traffic Radar & Corridors</div>', unsafe_allow_html=True)
    st.markdown('<div class="hud-sub">Visualizing regional & international flight paths converging on destination hub.</div>', unsafe_allow_html=True)

    hub_lat, hub_lon = 10.15, 76.40
    fig_map = go.Figure()

    for _, r in df_res.iterrows():
        color = "#10b981" if r["Delay"] == 0 else "#ef4444"
        fig_map.add_trace(go.Scattergeo(
            lon=[r["Origin_Lon"], hub_lon],
            lat=[r["Origin_Lat"], hub_lat],
            mode="lines",
            line=dict(width=2.5, color=color),
            opacity=0.85,
            hoverinfo="none",
            name=f"{r['Flight']} Arc"
        ))

    fig_map.add_trace(go.Scattergeo(
        lon=df_res["Origin_Lon"],
        lat=df_res["Origin_Lat"],
        mode="markers+text",
        text=df_res["Flight"],
        textposition="top center",
        marker=dict(size=9, color="#38bdf8", symbol="circle"),
        hoverinfo="text",
        hovertext=[f"Flight: {r['Flight']}<br>Origin: {r['Origin']}<br>Assigned: {r['Gate']}<br>Delay: +{r['Delay']}m" for _, r in df_res.iterrows()],
        name="Inbound Waypoints"
    ))

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
        showcountries=True,
        countrycolor="#334155",
        bgcolor="rgba(0,0,0,0)",
        fitbounds="locations"
    )
    fig_map.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc"))
    st.plotly_chart(fig_map, use_container_width=True)

elif st.session_state.active_page == "Commercial Seat Clearance":
    st.markdown('<div class="hud-title">🏷️ Commercial Revenue & Standby Seat Clearance</div>', unsafe_allow_html=True)
    st.markdown('<div class="hud-sub">Monetizing no-shows created by ground delays. Dynamic algorithmic pricing triggers during turnaround.</div>', unsafe_allow_html=True)

    clearance_items = []
    for _, r in df_res.iterrows():
        unsold_seats = int(np.clip(r["TotalSeats"] * 0.03 + (r["Delay"] // 15), 2, 8))
        discount = int(np.clip(20 + (r["Delay"] * 0.6), 20, 65))
        offer_fare = int(r["BaseFare"] * (1 - discount / 100))
        salvaged = offer_fare * unsold_seats
        clearance_items.append({
            "Flight": r["Flight"],
            "Origin": r["Origin"],
            "Assigned Stand": r["Gate"],
            "Unsold Seats": unsold_seats,
            "Original Fare ($)": r["BaseFare"],
            "Flash Price ($)": offer_fare,
            "Discount": f"{discount}% OFF",
            "Potential Ancillary ($)": salvaged
        })
    df_comm = pd.DataFrame(clearance_items)

    c1, c2, c3 = st.columns(3)
    c1.metric("Eligible Departing Flights", len(df_comm))
    c2.metric("Total Standby Inventory", f"{df_comm['Unsold Seats'].sum()} Seats")
    c3.metric("Projected Salvaged Revenue", f"${df_comm['Potential Ancillary ($)'].sum():,}")

    st.markdown("---")
    st.markdown("#### 🎫 Live Standby Deals Feed (Passenger Portal Simulation)")
    for idx, row in df_comm.iterrows():
        with st.container():
            p1, p2, p3, p4, p5 = st.columns([1.5, 2, 1.5, 1.5, 1.5])
            p1.markdown(f"**{row['Flight']}**  \n`{row['Origin']}`")
            p2.markdown(f"**Gate:** {row['Assigned Stand']}  \n*{row['Unsold Seats']} seats available*")
            p3.markdown(f"~~${row['Original Fare ($)']}~~  \n<span style='color:#10b981;font-weight:700;'>${row['Flash Price ($)']}</span>", unsafe_allow_html=True)
            p4.markdown(f"<span style='background:#047857;padding:3px 8px;border-radius:4px;font-size:0.8rem;font-weight:600;'>{row['Discount']}</span>", unsafe_allow_html=True)
            if p5.button(f"Claim Seat", key=f"book_{row['Flight']}"):
                st.success(f"Seat on {row['Flight']} claimed at ${row['Flash Price ($)']}! Ancillary revenue captured.")
            st.markdown("<hr style='border:0.5px solid rgba(255,255,255,0.05); margin:8px 0;' />", unsafe_allow_html=True)

elif st.session_state.active_page == "Gate Optimization Telemetry":
    st.markdown('<div class="hud-title">📈 Gate Optimization Diagnostics & Gantt Timeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="hud-sub">Mathematical schedule intervals from Google OR-Tools CP-SAT solver.</div>', unsafe_allow_html=True)

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
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc"))
        st.plotly_chart(fig, use_container_width=True)

        t1, t2, t3 = st.columns(3)
        t1.metric("Solver Status", "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE")
        t2.metric("Wall Time", f"{solver.WallTime():.3f}s")
        t3.metric("Remote Penalty Objective", f"{int(solver.ObjectiveValue())} pts")
    else:
        st.error("No valid gate assignment possible.")
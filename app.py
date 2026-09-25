import streamlit as st
import cv2
import numpy as np
from PIL import Image
import datetime
import pennylane as qml
from ultralytics import YOLO
import plotly.graph_objects as go
import random
import io
import os
import folium
from streamlit_folium import st_folium

# ==========================================
# 1. PAGE CONFIG & CSS
# ==========================================
st.set_page_config(page_title="TRAX AI", page_icon="🚦", layout="wide")
st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: 800; color: #0f172a; }
    .sub-header { font-size: 1rem; color: #64748b; margin-top: -10px; }
    .agent-card { background-color: #f8fafc; border: 1px solid #e2e8f0; padding: 12px; border-radius: 8px; text-align: center; }
    .alert-box { background-color: #fef2f2; border-left: 5px solid #ef4444; padding: 15px; border-radius: 5px; margin-bottom: 10px; }
    .safe-box { background-color: #f0fdf4; border-left: 5px solid #22c55e; padding: 15px; border-radius: 5px; margin-bottom: 10px; }
    .challan-box { background-color: #f8fafc; border: 2px solid #0f172a; padding: 15px; border-radius: 8px; font-family: 'Courier New', monospace; }
    .reason-box { background-color: #eff6ff; border-left: 4px solid #3b82f6; padding: 12px; border-radius: 5px; font-size: 0.9rem; }
    .quantum-badge { background-color: #eef2ff; color: #4f46e5; padding: 4px 12px; border-radius: 12px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🚦 TRAX AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Multi-Agent Traffic Violation Detection | Classical CV + Quantum ML + IoT Edge</p>', unsafe_allow_html=True)

# ==========================================
# 2. SIDEBAR: SIMULATED IOT SENSORS & MAP
# ==========================================
st.sidebar.header("🌐 City-Wide Sensor Network")
ambient_temp = random.randint(22, 35)
road_friction = round(random.uniform(0.6, 0.95), 2)
weather = random.choice(["Clear", "Light Rain", "Foggy", "Heavy Rain"])
intersection_load = random.randint(10, 85)

st.sidebar.metric("🌡️ Ambient Temp", f"{ambient_temp}°C")
st.sidebar.metric("💧 Road Friction", f"{road_friction}")
st.sidebar.metric("🌧️ Weather", weather)
st.sidebar.metric("🚦 Intersection Load", f"{intersection_load}%")

st.sidebar.markdown("### 🗺️ Live Agent Deployment")
m = folium.Map(location=[28.6139, 77.2090], zoom_start=12, tiles="CartoDB positron")
folium.CircleMarker(location=[28.625, 77.210], radius=7, color="#ef4444", fill=True, fill_color="#ef4444", popup="Node 01: Violation").add_to(m)
folium.CircleMarker(location=[28.605, 77.220], radius=7, color="#3b82f6", fill=True, fill_color="#3b82f6", popup="Node 02: Monitoring").add_to(m)
folium.CircleMarker(location=[28.615, 77.190], radius=7, color="#10b981", fill=True, fill_color="#10b981", popup="Node 03: Clear").add_to(m)
folium.CircleMarker(location=[28.630, 77.200], radius=7, color="#ef4444", fill=True, fill_color="#ef4444", popup="Node 04: Helmet Violation").add_to(m)
st_folium(m, width=280, height=220)
st.sidebar.caption("🔴 Active | 🔵 Monitoring | 🟢 Clear")

# ==========================================
# 3. CORE AI PIPELINES (CACHED)
# ==========================================
@st.cache_resource
def load_yolo():
    return YOLO('yolov8n.pt')

@st.cache_resource
def setup_quantum():
    return qml.device("default.qubit", wires=4)

yolo_model = load_yolo()
dev = setup_quantum()

@qml.qnode(dev)
def quantum_circuit(features):
    qml.AngleEmbedding(features, wires=range(4), rotation='X')
    qml.CNOT(wires=[0,1]); qml.CNOT(wires=[1,2]); qml.CNOT(wires=[2,3]); qml.CNOT(wires=[3,0])
    qml.RY(0.5, wires=0); qml.RY(1.2, wires=1); qml.RY(-0.8, wires=2); qml.RY(0.3, wires=3)
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

def calculate_quantum_risk(bikes, persons, cars, avg_conf):
    f1 = min(bikes / 5.0, 1.0)
    f2 = min(persons / 5.0, 1.0)
    f3 = min(cars / 5.0, 1.0)
    f4 = avg_conf
    q_out = quantum_circuit(np.array([f1, f2, f3, f4]))
    return int((np.sum([abs(x) for x in q_out]) / 4.0) * 100)

# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================
def apply_privacy_blur(image, boxes, names):
    blurred = image.copy()
    for box in boxes:
        if names[int(box.cls)] == 'person':
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            h = int((y2 - y1) * 0.4)
            if h > 0 and y1 + h <= blurred.shape[0]:
                face = blurred[y1:y1+h, x1:x2]
                blurred[y1:y1+h, x1:x2] = cv2.GaussianBlur(face, (25, 25), 0)
    return blurred

# ==========================================
# 5. VIOLATION DETECTION ENGINE (HELMET + SEATBELT FIXED)
# ==========================================
def detect_violations(detections, avg_conf, manual_helmet_flag=False, manual_seatbelt_flag=False):
    bikes = detections.count('motorcycle')
    persons = detections.count('person')
    cars = detections.count('car')
    
    violations, reasoning, counterfactuals = [], [], []

    # --- HELMET VIOLATION ---
    if bikes > 0:
        if manual_helmet_flag:
            violations.append("Helmetless Riding (Manual Confirmed)")
            reasoning.append(f"Detected {bikes} motorcycle(s). Manual verification confirms rider without helmet.")
            reasoning.append("Base YOLO cannot classify helmet/no-helmet; human-in-the-loop applied.")
            counterfactuals.append("✓ Ensure all riders wear ISI-certified helmets.")
        elif persons < bikes:
            violations.append("Helmetless Riding (No Rider Detected)")
            reasoning.append(f"Detected {bikes} motorcycle(s) but only {persons} person(s).")
            counterfactuals.append("✓ Ensure all riders wear ISI-certified helmets.")

    # --- EXCESSIVE OCCUPANCY ---
    if bikes > 0 and persons > bikes * 2:
        violations.append("Excessive Occupancy (Triple Riding)")
        reasoning.append(f"Detected {persons} persons on {bikes} motorcycle(s). Max allowed: 2.")
        counterfactuals.append("✓ Limit passengers to a maximum of two per motorcycle.")

    # --- SEAT-BELT VIOLATION ---
    if cars > 0:
        if manual_seatbelt_flag:
            violations.append("Seat-Belt Violation (Cabin Confirmed)")
            reasoning.append(f"Detected {cars} vehicle(s). Manual cabin analysis confirms unbuckled occupant.")
            reasoning.append("Base YOLO cannot penetrate tinted glass; human-in-the-loop verification applied.")
            counterfactuals.append("✓ Driver and front passenger must visibly fasten seat-belts before driving.")
        elif avg_conf < 0.5:
            violations.append("Possible Seat-Belt Violation (Low Visibility)")
            reasoning.append(f"Detected {cars} car(s) with low confidence ({avg_conf*100:.0f}%). Cabin likely occluded/tinted.")
            counterfactuals.append("✓ Ensure cabin visibility for automated seat-belt enforcement.")

    return violations, reasoning, counterfactuals, bikes, persons, cars

# ==========================================
# 6. INPUT: ERROR-PROOFED LOADER
# ==========================================
st.markdown("### 🎛️ Input Control Panel")
assets_exist = all(os.path.exists(f) for f in ["assets/1_violation.jpg", "assets/2_safe.jpg", "assets/3_complex.jpg"])

if assets_exist:
    sample_images = {
        "🚨 Load Violation Sample": "assets/1_violation.jpg",
        "✅ Load Safe Sample": "assets/2_safe.jpg",
        "🏙️ Load Complex Traffic": "assets/3_complex.jpg"
    }
    selected = st.selectbox("👉 Load pre-tested sample:", ["None (Upload manually)"] + list(sample_images.keys()))
    if selected != "None (Upload manually)":
        with open(sample_images[selected], "rb") as f:
            uploaded_file = io.BytesIO(f.read())
        uploaded_file.name = selected.split(':')[-1].strip() + ".jpg"
        st.success(f"Loaded: {uploaded_file.name}")
    else:
        uploaded_file = st.file_uploader("📸 Upload Traffic Image", type=['jpg','jpeg','png'])
else:
    st.warning("⚠️ Demo samples not found. Please upload an image manually below.")
    uploaded_file = st.file_uploader("📸 Upload Traffic Image", type=['jpg','jpeg','png'])

# ==========================================
# 7. PROCESSING PIPELINE
# ==========================================
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)
    
    with st.spinner("👁️ Agent 1 (Vision): Running YOLOv8 Object Detection..."):
        results = yolo_model(img_array, conf=0.4, verbose=False)
        
    boxes = results[0].boxes
    names = results[0].names
    detections = [names[int(cls)] for cls in boxes.cls]
    avg_conf = float(boxes.conf.mean()) if len(boxes) > 0 else 0.0
    
    # --- MANUAL OVERRIDE CHECKBOXES ---
    bikes_detected = detections.count('motorcycle')
    cars_detected = detections.count('car')
    
    manual_helmet = False
    manual_seatbelt = False
    
    if bikes_detected > 0 or cars_detected > 0:
        st.warning("⚠️ Base YOLO cannot classify helmets or see inside cabins. Use manual overrides below.")
        
    if bikes_detected > 0:
        manual_helmet = st.checkbox("🏍️ Confirm: Rider visible & helmet NOT worn", key="helmet_override")
    
    if cars_detected > 0:
        manual_seatbelt = st.checkbox("🚗 Confirm: Cabin visible & seat-belt NOT fastened", key="seatbelt_override")
    
    # Pass manual flags to violation engine
    violations, reasoning, counterfactuals, bikes, persons, cars = detect_violations(
        detections, avg_conf, 
        manual_helmet_flag=manual_helmet, 
        manual_seatbelt_flag=manual_seatbelt
    )
    
    with st.spinner("⚛️ Agent 2 (Quantum): Computing non-linear risk via 4-Qubit VQC..."):
        quantum_risk = calculate_quantum_risk(bikes, persons, cars, avg_conf)
        
    iot_multiplier = 1.2 if weather in ["Light Rain", "Foggy"] else (1.5 if weather == "Heavy Rain" else 1.0)
    if road_friction < 0.7: iot_multiplier += 0.1
    
    has_violation = len(violations) > 0
    final_score = min(int(quantum_risk * iot_multiplier), 100) if has_violation else quantum_risk
    
    privacy_on = st.checkbox("🔒 Enable Privacy Mode (Blur Faces)", value=True)
    display_img = apply_privacy_blur(results[0].plot(), boxes, names) if privacy_on else results[0].plot()

    # ==========================================
    # 8. MULTI-AGENT DASHBOARD
    # ==========================================
    st.markdown("### 🤖 Multi-Agent Consensus Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="agent-card"><h4>👁️ Vision</h4><p style="font-size:22px; font-weight:bold; color:#0369a1;">{bikes}🏍️ {persons}🧍 {cars}🚗</p></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="agent-card"><h4>⚛️ Quantum</h4><p style="font-size:22px; font-weight:bold; color:#6b21a8;">{quantum_risk}/100</p></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="agent-card"><h4>📡 IoT</h4><p style="font-size:22px; font-weight:bold; color:#059669;">{weather}</p></div>', unsafe_allow_html=True)
    sc = "#ef4444" if final_score > 60 else "#10b981"
    c4.markdown(f'<div class="agent-card" style="border-left:4px solid {sc};"><h4>⚖️ Verdict</h4><p style="font-size:22px; font-weight:bold; color:{sc};">{final_score}/100</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.subheader("📸 Evidence Frame")
        st.image(display_img, channels="BGR", use_container_width=True)

    with col2:
        st.subheader("📜 Violation Report")
        if has_violation:
            for v in violations: 
                st.markdown(f'<div class="alert-box">🚨 <b>{v}</b></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="safe-box">✅ <b>No Violations Detected.</b></div>', unsafe_allow_html=True)

        st.markdown("#### 🧠 Why Was This Flagged?")
        for r in (reasoning or ["No anomalies detected."]):
            st.markdown(f'<div class="reason-box">• {r}</div>', unsafe_allow_html=True)

        st.markdown("#### 💡 Safety Correction")
        for c in (counterfactuals or ["No corrections needed."]):
            st.markdown(f"- {c}")

        st.markdown(f"#### ⚛️ Quantum Severity: <span class='quantum-badge'>{final_score}/100</span>", unsafe_allow_html=True)
        
        status = "Likely Violation" if final_score >= 85 else ("Needs Review" if final_score >= 60 else "Clear")
        st.info(f"**Human Verification:** {status}")
        
        if has_violation and final_score > 50:
            challan_id = f"TRAX-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            fine = 1000 if "Compound" in str(violations) else 500
            st.markdown(f"""
            <div class="challan-box">
                <h4 style="margin-top:0; color:#0f172a;">TRAFFIC VIOLATION NOTICE</h4>
                <p><b>Case ID:</b> {challan_id} | <b>Time:</b> {datetime.datetime.now().strftime('%H:%M:%S')}</p>
                <p><b>Violation:</b> {', '.join(violations)}</p>
                <p><b>Quantum Risk:</b> {final_score}/100 | <b>Weather:</b> {weather} (×{iot_multiplier})</p>
                <p><b>Fine:</b> ₹{fine}</p>
            </div>
            """, unsafe_allow_html=True)

else:
    st.info("👆 Upload an image to activate the TRAX AI pipeline.")
"""
dashboard.py - Streamlit Frontend for IBVAP-Lite
Provides a high-tech border surveillance interface with real-time video stream,
YOLOv8 + InsightFace annotations, dynamic threat alerts, side-by-side alert logs,
and interactive hardware simulation controls.
"""

import time
import os
import tempfile
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from capture import VideoCaptureHandler, create_synthetic_test_video
from detection import ObjectDetector
from faces import FaceRecognitionSystem
from alerts import AlertManager


# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="IBVAP-Lite | Border Video Analytics Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- MODERN TACTICAL DARK CSS ----------------
st.markdown("""
<style>
    /* Dark Theme Security Command Aesthetic */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    /* Top Header Bar */
    .main-header {
        background: linear-gradient(90deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px 20px;
        margin-bottom: 15px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .main-header h2 {
        color: #38bdf8;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        margin: 0;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 13px;
        margin: 0;
    }
    
    /* Status Pills */
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .status-online { background-color: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid #22c55e; }
    .status-threat { background-color: rgba(239, 68, 68, 0.25); color: #f87171; border: 1px solid #ef4444; }

    /* Red Alert Banner */
    .alert-banner-box {
        background: linear-gradient(90deg, #7f1d1d 0%, #b91c1c 50%, #7f1d1d 100%);
        border: 2px solid #ef4444;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
        color: #ffffff;
        text-align: center;
        font-weight: 800;
        font-size: 16px;
        letter-spacing: 1px;
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.6);
    }
    
    /* Metric Cards */
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 10px 14px;
        text-align: center;
    }
    .metric-card-val {
        font-size: 22px;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-card-lbl {
        font-size: 11px;
        color: #94a3b8;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- CACHED MODEL INITIALIZERS ----------------
@st.cache_resource(show_spinner="Loading YOLO26 COCO Object Detector...")
def get_object_detector():
    return ObjectDetector(model_name="yolo26n.pt")


@st.cache_resource(show_spinner="Loading InsightFace Model Suite...")
def get_face_recognition_system():
    fr = FaceRecognitionSystem()
    fr.initialize_model()
    if not fr.enrolled_faces:
        fr.enroll_watchlist("watchlist_images")
    return fr


def init_session_state():
    if "alert_manager" not in st.session_state:
        st.session_state.alert_manager = AlertManager(cooldown_seconds=4.0)
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "last_source_param" not in st.session_state:
        st.session_state.last_source_param = None
    if "temp_video_path" not in st.session_state:
        st.session_state.temp_video_path = None


init_session_state()
detector = get_object_detector()
face_system = get_face_recognition_system()
alert_mgr = st.session_state.alert_manager


# ---------------- SIDEBAR CONTROLS ----------------
with st.sidebar:
    st.markdown("## 🛡️ IBVAP-Lite Controls")
    st.markdown("<div class='status-badge status-online'>● System Online</div>", unsafe_allow_html=True)
    st.markdown("---")

    # 1. Video Source Picker
    st.subheader("📹 Video Feed Source")
    source_mode = st.selectbox(
        "Select Ingestion Mode:",
        ["Upload Footage", "Phone IP Camera", "Laptop Webcam"],
        index=0
    )

    source_param = None

    if source_mode == "Phone IP Camera":
        ip_url = st.text_input(
            "IP Camera Stream URL:",
            value="http://192.168.1.5:8080/video",
            help="E.g., from IP Webcam mobile app or RTSP stream"
        )
        source_param = ip_url.strip()

    elif source_mode == "Laptop Webcam":
        cam_idx = st.selectbox("Webcam Device Index:", [0, 1], index=0)
        source_param = cam_idx

    elif source_mode == "Upload Footage":
        uploaded_file = st.file_uploader(
            "Upload Border CCTV Footage (.mp4 / .avi / .mov):",
            type=["mp4", "avi", "mov", "mkv"]
        )
        if uploaded_file is not None:
            if st.session_state.temp_video_path is None or st.session_state.last_source_param != uploaded_file.name:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_file.read())
                st.session_state.temp_video_path = tfile.name
                st.session_state.last_source_param = uploaded_file.name
            source_param = st.session_state.temp_video_path
        else:
            sample_file = "test_border_cctv.mp4" if os.path.exists("test_border_cctv.mp4") else "test_feed.mp4"
            if not os.path.exists(sample_file):
                create_synthetic_test_video(sample_file)
            source_param = sample_file
            st.info(f"ℹ️ Using sample border surveillance feed ({sample_file}).")

    st.markdown("---")

    # 2. Pipeline Parameters
    st.subheader("⚙️ Analytics Parameters")
    match_threshold = st.slider(
        "Watchlist Match Threshold (Cosine):",
        min_value=0.20,
        max_value=0.80,
        value=0.40,
        step=0.02,
        help="Higher values require stricter face matches. Default 0.40."
    )

    skip_frames = st.slider(
        "Face Analysis Frequency (Every Nth Frame):",
        min_value=1,
        max_value=10,
        value=3,
        help="YOLOv8 runs every frame. InsightFace runs every Nth frame to optimize CPU."
    )

    night_mode = st.toggle(
        "🌙 Night Vision Enhancement (CLAHE)",
        value=False,
        help="Boosts contrast/illumination on low-light border cameras using CLAHE."
    )

    conf_threshold = st.slider(
        "YOLOv8 Detection Confidence:",
        min_value=0.15,
        max_value=0.90,
        value=0.35,
        step=0.05
    )

    st.markdown("---")

    # 3. Watchlist Management
    st.subheader("👥 Watchlist Database")
    enrolled_count = len(face_system.enrolled_faces)
    st.write(f"**Enrolled Targets:** `{enrolled_count}` individuals")
    if face_system.enrolled_faces:
        with st.expander("View Enrolled Identities"):
            for name, count in face_system.enrolled_counts.items():
                st.write(f"- 👤 **{name}** ({count} ref photos)")

    if st.button("🔄 Re-Enroll Watchlist Images"):
        with st.spinner("Re-scanning watchlist_images/ directory..."):
            ok, msg = face_system.enroll_watchlist("watchlist_images")
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    if st.button("🗑️ Clear Alert History"):
        alert_mgr.clear()
        st.toast("Alert history cleared.")

    st.markdown("---")
    st.caption("SIH26187 Simulation: Edge Smart-Camera + Central FRS Match Server")


# ---------------- TOP HEADER ----------------
st.markdown("""
<div class='main-header'>
    <div>
        <h2>🛡️ IBVAP-Lite: Intelligent Border Video Analytics Platform</h2>
        <p class='sub-title'>Real-Time Edge Detection (YOLOv8) & Face Watchlist Recognition (InsightFace ArcFace)</p>
    </div>
    <div style='text-align: right;'>
        <div style='font-size: 12px; color: #94a3b8;'>ACTIVE NODE</div>
        <div style='font-weight: bold; color: #38bdf8;'>CCTV-SECTOR-04</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------- STREAM CONTROLS & METRICS ----------------
c1, c2, c3, c4, c5 = st.columns([1.5, 1, 1, 1, 1])
with c1:
    btn_start = st.button("▶️ Start Live Feed", use_container_width=True, type="primary")
with c2:
    btn_stop = st.button("⏹️ Stop Stream", use_container_width=True)

if btn_start:
    st.session_state.is_running = True
if btn_stop:
    st.session_state.is_running = False

# Summary Metric Tiles
stats = alert_mgr.get_stats()
with c3:
    st.markdown(f"<div class='metric-card'><div class='metric-card-val'>{enrolled_count}</div><div class='metric-card-lbl'>Watchlist Size</div></div>", unsafe_allow_html=True)
with c4:
    st.markdown(f"<div class='metric-card'><div class='metric-card-val'>{stats['total_alerts']}</div><div class='metric-card-lbl'>Threat Hits</div></div>", unsafe_allow_html=True)
with c5:
    st.markdown(f"<div class='metric-card'><div class='metric-card-val'>{stats['highest_confidence']}</div><div class='metric-card-lbl'>Top Confidence</div></div>", unsafe_allow_html=True)

st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

# ---------------- DYNAMIC BANNER & FEED LAYOUT ----------------
banner_placeholder = st.empty()

col_feed, col_logs = st.columns([7, 4])

with col_feed:
    st.markdown("##### 🔴 Live Surveillance Feed")
    video_placeholder = st.empty()

with col_logs:
    st.markdown("##### 🚨 Real-Time Threat Alerts (Newest First)")
    alert_placeholder = st.empty()


# ---------------- VIDEO STREAM EXECUTION LOOP ----------------
if st.session_state.is_running and source_param is not None:
    cap_handler = VideoCaptureHandler(source_type=source_mode, source_param=source_param, loop_video=True)
    opened, open_msg = cap_handler.open()

    if not opened:
        st.session_state.is_running = False
        st.error(f"⚠️ **Connection Error:** {open_msg}")
        st.info("👉 **Suggested Action:** Switch the sidebar source to **'Upload Footage'** or verify the **Phone IP Camera URL / Webcam permissions**.")
    else:
        frame_idx = 0
        cached_faces = []
        fps_smooth = 0.0

        try:
            while st.session_state.is_running:
                loop_start = time.time()
                ret, frame = cap_handler.read()

                if not ret or frame is None:
                    st.warning("Video stream ended or disconnected.")
                    break

                frame_idx += 1

                # 1. Night Mode (CLAHE)
                if night_mode:
                    frame = detector.apply_night_mode(frame)

                # 2. YOLOv8 Object Detection (Every Frame)
                detections = detector.detect(frame, conf_threshold=conf_threshold)
                annotated = detector.draw_detections(frame, detections)

                # 3. InsightFace Face Recognition (Every N-th Frame)
                if frame_idx % skip_frames == 0 or frame_idx == 1:
                    cached_faces = face_system.process_frame(frame, threshold=match_threshold)
                    for face in cached_faces:
                        if face["is_match"]:
                            alert_mgr.trigger_alert(
                                person_name=face["name"],
                                confidence=face["confidence"],
                                source=source_mode
                            )

                # Draw Face Bounding Boxes
                annotated = face_system.draw_faces(annotated, cached_faces)

                # Telemetry HUD on video
                person_cnt = sum(1 for d in detections if d["category"] == "person")
                vehicle_cnt = sum(1 for d in detections if d["category"] == "vehicle")
                loop_dur = time.time() - loop_start
                instant_fps = 1.0 / max(loop_dur, 0.001)
                fps_smooth = 0.9 * fps_smooth + 0.1 * instant_fps if fps_smooth > 0 else instant_fps

                hud_text = f"FPS: {fps_smooth:.1f} | Persons: {person_cnt} | Vehicles: {vehicle_cnt} | NightMode: {'ON' if night_mode else 'OFF'}"
                cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 30), (15, 23, 42), -1)
                cv2.putText(annotated, hud_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1, cv2.LINE_AA)

                # Convert BGR to RGB for Streamlit display
                rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                # Update Dynamic Red Alert Banner
                active_banner = alert_mgr.get_active_banner(banner_duration=2.5)
                if active_banner:
                    banner_placeholder.markdown(f"""
                    <div class='alert-banner-box'>
                        🚨 CRITICAL WATCHLIST MATCH: {active_banner['person'].upper()} ({active_banner['confidence']*100:.1f}% Match Confidence)
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    banner_placeholder.empty()

                # Update Alert Table
                df_alerts = alert_mgr.get_dataframe()
                if not df_alerts.empty:
                    alert_placeholder.dataframe(df_alerts, hide_index=True, use_container_width=True)
                else:
                    alert_placeholder.info("No watchlist threats detected yet. Monitoring active feed...")

        finally:
            cap_handler.release()
else:
    # Idle / Stopped State placeholder
    video_placeholder.info("Surveillance feed is currently paused. Click '▶️ Start Live Feed' to begin real-time analysis.")
    df_alerts = alert_mgr.get_dataframe()
    if not df_alerts.empty:
        alert_placeholder.dataframe(df_alerts, hide_index=True, use_container_width=True)
    else:
        alert_placeholder.info("Alert log is empty.")

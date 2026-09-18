# IBVAP-Lite: Intelligent Border Video Analytics Platform
**Prototype Module: Real-Time Object Detection (YOLOv8) + Watchlist Face Recognition (InsightFace ArcFace)**

---

## 🎯 System Overview & Real-World Hardware Simulation (SIH26187)

In real-world border security and perimeter surveillance deployments (e.g., SIH26187 specifications), video analytics is distributed across two primary physical layers:

```
[ Tier 1: Border Smart-Camera Edge Node ]
   ├── High-FPS Video Stream Ingestion (CCTV / RTSP / Thermal / Optical)
   ├── Low-latency Object Detection & Tracking (YOLOv8 Edge Model)
   │     └── Filters: "person", "car", "truck", "bus"
   └── Low-Light Preprocessing (CLAHE Night Vision Enhancement)
                │
                ▼ (Selected Face Crops / Frame Sample every Nth frame)
[ Tier 2: Central FRS Matching Server ]
   ├── High-Dimensional Feature Extraction (InsightFace 512-d ArcFace Deep Embeddings)
   ├── Fast Cosine-Similarity Match against Enrolled Watchlist Database
   ├── Alert Management & Cooldown Deduplication
   └── Command & Control Security Dashboard (Streamlit Web UI)
```

**IBVAP-Lite simulates this entire ecosystem seamlessly on a single workstation or laptop.**

---

## 🚀 Key Features

1. **3 Interchangeable Video Input Modes**:
   - 📱 **Phone IP Camera Stream**: Connect live mobile streams (e.g., *IP Webcam* Android app or RTSP `http://<phone-ip>:8080/video`). Includes automatic fallback to webcam 0 if unreachable.
   - 💻 **Laptop / USB Webcam**: Selectable device index (`0`, `1`, etc.) with DirectShow hardware acceleration.
   - 📁 **Uploaded CCTV Footage**: Supports `.mp4`, `.avi`, `.mov` with **native FPS pacing** so playback speed matches real live CCTV time rather than fast-forwarding.
2. **YOLOv8 Object Detection**:
   - Detects `person`, `car`, `truck`, and `bus` classes every frame with tactical color-coded bounding boxes.
3. **InsightFace Face Recognition & Watchlist Matching**:
   - Pre-enrolls faces from `watchlist_images/` with multi-photo averaging for higher vector accuracy.
   - Computes cosine similarity matching with configurable threshold (default `0.40`).
   - Frame skipping optimization ($N$-th frame slider, default `3`) keeps CPU usage smooth without video stutter.
4. **Night Vision Mode**:
   - Optional CLAHE (Contrast Limited Adaptive Histogram Equalization) toggle for low-light border cameras.
5. **Real-time Threat Management**:
   - Prominent red alert banner overlay on live video upon watchlist identification.
   - Running threat log table (newest on top) with timestamp, matched person, and confidence score.
   - Cooldown deduplication preventing alert spam.

---

## 🛠️ Installation & Setup Guide

### 1. Prerequisites
- Python 3.10 or 3.11 recommended.

### 2. Create Virtual Environment
```bash
# Windows
py -3.11 -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3.11 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies in Recommended Order
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 📂 Watchlist Enrollment

Organize suspect or target reference photos inside `watchlist_images/`:

```
watchlist_images/
├── John_Doe/
│   ├── photo1.jpg
│   └── photo2.jpg
├── Jane_Smith/
│   ├── front.png
│   └── angle.png
└── Suspect_3.jpg
```
- Multi-photo subfolders are automatically averaged into a single robust L2-normalized vector.
- Single flat images are also supported!

---

## 🖥️ Launching the Application

### Option A: Interactive Streamlit Dashboard (Primary Demo)
```bash
streamlit run dashboard.py
```
Open your browser at `http://localhost:8501`.

### Option B: Standalone CLI / OpenCV Test Mode
```bash
# Run with synthetic/sample CCTV video
python main.py --source-type "Upload Footage" --source "test_feed.mp4"

# Run with local webcam
python main.py --source-type "Webcam" --source 0

# Run with phone IP webcam
python main.py --source-type "IP Camera" --source "http://192.168.1.5:8080/video"
```

---

## 📁 Project Structure

```
.
├── capture.py        # Video ingestion stream handler (IP Cam, Webcam, File with FPS pacing)
├── detection.py      # YOLOv8 object detector (person/vehicles) & CLAHE night vision
├── faces.py          # InsightFace face detection, ArcFace embedding & watchlist matching
├── alerts.py         # Threat logging, deduplication cooldown & telemetry stats
├── dashboard.py      # Streamlit security command center interface
├── main.py           # Standalone CLI / OpenCV pipeline runner
├── watchlist_images/ # Directory containing reference photos for watchlist targets
├── requirements.txt  # Pinned dependency requirements
└── README.md         # Documentation & real-world architecture notes
```

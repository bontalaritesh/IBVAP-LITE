"""
server.py - IBVAP-Lite FastAPI Backend
Real-time border surveillance pipeline server with WebSocket video streaming,
REST APIs for watchlist enrollment, and dynamic alert management.
"""

import asyncio
import csv
import io
import json
import os
import time
import traceback
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request

from alerts import AlertManager, _load_telegram_config, save_telegram_config, test_telegram_connection
from capture import VideoCaptureHandler, create_synthetic_test_video
from detection import ObjectDetector
from faces import FaceRecognitionSystem

# Ensure snapshot evidence directory exists
os.makedirs("captured_threats", exist_ok=True)

# ─── Global State ─────────────────────────────────────────────────────────────
camera_zone_label: str = "Sector 04 — Border Gate 1"
detector: Optional[ObjectDetector] = None
face_system: Optional[FaceRecognitionSystem] = None
alert_manager: Optional[AlertManager] = None

def get_detector():
    global detector
    if detector is None:
        detector = ObjectDetector(model_name="yolo26n.onnx")
    return detector

def get_face_system():
    global face_system
    if face_system is None:
        face_system = FaceRecognitionSystem()
        face_system.initialize_model()
    return face_system

def get_alert_manager():
    global alert_manager
    if alert_manager is None:
        alert_manager = AlertManager(cooldown_seconds=4.0)
    return alert_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load models on startup."""
    print("[IBVAP] Pre-loading models on startup...")
    os.makedirs("static/thumbnails", exist_ok=True)
    os.makedirs("watchlist_images", exist_ok=True)
    os.makedirs("captured_threats", exist_ok=True)
    if not os.path.exists("test_border_cctv.mp4"):
        create_synthetic_test_video("test_border_cctv.mp4", duration_sec=8, fps=25)
    get_alert_manager()
    get_face_system()
    get_detector()
    print("[IBVAP] System ready.")
    yield
    print("[IBVAP] Server shutdown.")

# ─── App Init ─────────────────────────────────────────────────────────────────
app = FastAPI(title="IBVAP-Lite", version="2.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/captured_threats", StaticFiles(directory="captured_threats"), name="captured_threats")

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", response_class=FileResponse)
async def index():
    return FileResponse("templates/index.html")

@app.get("/api/watchlist")
async def get_watchlist():
    fs = get_face_system()
    persons = []
    for pid, meta in fs.enrolled_meta.items():
        persons.append({
            "id": pid,
            "name": meta.get("name", "Unknown"),
            "category": meta.get("category", "Unknown"),
            "notes": meta.get("notes", ""),
            "photo_count": meta.get("photo_count", 1),
            "created_at": meta.get("created_at", ""),
            "thumb_url": meta.get("thumb_url", "/static/thumbnails/default.jpg"),
        })
    return JSONResponse({"persons": persons, "total": len(persons)})

@app.post("/api/enroll")
async def enroll_person(
    name: str = Form(...),
    category: str = Form("Friend / VIP"),
    notes: str = Form(""),
    photos: list[UploadFile] = File(...)
):
    if not name.strip():
        return JSONResponse({"success": False, "message": "Name is required"}, status_code=400)

    fs = get_face_system()
    enrolled_count = 0
    last_meta = None
    errors = []

    for photo in photos:
        try:
            image_bytes = await photo.read()
            ok, msg, meta = fs.enroll_single_photo(
                name=name.strip(),
                category=category,
                notes=notes,
                image_bytes=image_bytes
            )
            if ok:
                enrolled_count += 1
                last_meta = meta
            else:
                errors.append(f"{photo.filename}: {msg}")
        except Exception as e:
            errors.append(f"{photo.filename}: {str(e)}")

    if enrolled_count == 0:
        return JSONResponse({
            "success": False,
            "message": f"No faces detected. Errors: {'; '.join(errors)}"
        }, status_code=400)

    clean_meta = None
    if last_meta:
        clean_meta = {k: v for k, v in last_meta.items() if k != "raw_embeddings"}

    return JSONResponse({
        "success": True,
        "message": f"Enrolled {enrolled_count} photo(s) for '{name}' successfully.",
        "person": clean_meta,
        "errors": errors
    })

@app.put("/api/watchlist/{person_id}")
@app.post("/api/watchlist/{person_id}")
async def update_person_details(
    person_id: str,
    name: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    photos: Optional[list[UploadFile]] = File(None)
):
    fs = get_face_system()
    ok, msg, clean_meta = fs.update_person(person_id, name=name, category=category, notes=notes)
    if not ok:
        return JSONResponse({"success": False, "message": msg}, status_code=404)

    # If additional photos were uploaded for this person, enroll them
    enrolled_photos = 0
    if photos:
        for photo in photos:
            try:
                if photo.filename:
                    image_bytes = await photo.read()
                    if image_bytes:
                        ok_photo, _, meta = fs.enroll_single_photo(
                            name=clean_meta["name"],
                            category=clean_meta["category"],
                            notes=clean_meta.get("notes", ""),
                            image_bytes=image_bytes
                        )
                        if ok_photo:
                            enrolled_photos += 1
            except Exception:
                pass

    if enrolled_photos > 0:
        msg += f" Added {enrolled_photos} new photo(s)."

    return JSONResponse({"success": True, "message": msg, "person": clean_meta})

@app.delete("/api/watchlist/{person_id}")
async def delete_person(person_id: str):
    fs = get_face_system()
    ok, msg = fs.delete_person(person_id)
    return JSONResponse({"success": ok, "message": msg})

@app.get("/api/alerts")
async def get_alerts():
    am = get_alert_manager()
    return JSONResponse({
        "alerts": am.get_recent_alerts(50),
        "stats": am.get_stats()
    })

@app.post("/api/alerts/clear")
async def clear_alerts():
    get_alert_manager().clear()
    return JSONResponse({"success": True})

@app.get("/api/alerts/export")
async def export_alerts_csv():
    """Export full alert history as a downloadable CSV audit log."""
    am = get_alert_manager()
    output = io.StringIO()
    fieldnames = ["id", "date", "timestamp", "person", "match_tier", "confidence_pct",
                  "confidence_raw", "threat_level", "source", "snapshot_path"]
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for alert in reversed(am.alerts):  # oldest first for audit log
        writer.writerow(alert)
    output.seek(0)
    filename = f"ibvap_alert_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.get("/api/telegram/config")
async def get_telegram_config():
    token, chat_id = _load_telegram_config()
    masked_token = (token[:6] + "..." + token[-4:]) if len(token) > 10 else ("Configured" if token else "Not configured")
    return JSONResponse({
        "configured": bool(token and chat_id),
        "bot_token": token or "",
        "bot_token_masked": masked_token,
        "chat_id": chat_id or ""
    })

@app.post("/api/telegram/config")
async def update_telegram_config(
    bot_token: Optional[str] = Form(None),
    token: Optional[str] = Form(None),
    chat_id: str = Form(...)
):
    actual_token = str(bot_token or token or "").strip()
    actual_chat = str(chat_id).strip()

    if not actual_token:
        return JSONResponse({"success": False, "message": "Bot Token is required."}, status_code=400)
    if not actual_chat:
        return JSONResponse({"success": False, "message": "Chat ID is required."}, status_code=400)

    save_telegram_config(actual_token, actual_chat)

    # Also update global alert manager instance
    am = get_alert_manager()
    am.telegram_token = actual_token
    am.telegram_chat_id = actual_chat

    # Send verification ping and fetch bot/chat metadata
    ok, msg, details = test_telegram_connection(actual_token, actual_chat)

    return JSONResponse({
        "success": ok,
        "message": msg if ok else f"Credentials saved locally, but Telegram delivery test failed: {msg}",
        "details": details
    })

@app.post("/api/telegram/test")
async def trigger_telegram_test(
    bot_token: Optional[str] = Form(None),
    token: Optional[str] = Form(None),
    chat_id: Optional[str] = Form(None)
):
    actual_token = str(bot_token or token or "").strip() or None
    actual_chat = str(chat_id or "").strip() or None
    ok, msg, details = test_telegram_connection(actual_token, actual_chat)
    return JSONResponse({"success": ok, "message": msg, "details": details})

@app.get("/api/zone")
async def get_zone():
    global camera_zone_label
    return JSONResponse({"zone": camera_zone_label})

@app.post("/api/zone")
async def set_zone(zone: str = Form(...)):
    global camera_zone_label
    camera_zone_label = zone.strip() or camera_zone_label
    return JSONResponse({"success": True, "zone": camera_zone_label})

@app.post("/api/telegram/config")
async def set_telegram_config(token: str = Form(...), chat_id: str = Form(...)):
    """Configures Telegram Bot integration for live alert dispatches."""
    am = get_alert_manager()
    am.telegram_token = token.strip()
    am.telegram_chat_id = chat_id.strip()
    os.environ["TELEGRAM_BOT_TOKEN"] = am.telegram_token
    os.environ["TELEGRAM_CHAT_ID"] = am.telegram_chat_id
    return JSONResponse({
        "success": True,
        "message": "Telegram Bot alert channel configured successfully.",
        "configured": bool(am.telegram_token and am.telegram_chat_id)
    })

@app.get("/api/telegram/config")
async def get_telegram_config():
    am = get_alert_manager()
    return JSONResponse({
        "configured": bool(am.telegram_token and am.telegram_chat_id),
        "chat_id": am.telegram_chat_id or ""
    })

@app.get("/api/status")
async def system_status():
    global camera_zone_label
    fs = get_face_system()
    am = get_alert_manager()
    return JSONResponse({
        "enrolled_count": len(fs.enrolled_faces),
        "model_status": fs.status_message,
        "alert_count": len(am.alerts),
        "unique_targets": len(set(a["person"] for a in am.alerts)),
        "zone": camera_zone_label,
    })

# ─── WebSocket Video Stream ────────────────────────────────────────────────────
@app.websocket("/ws/feed")
async def websocket_feed(
    websocket: WebSocket,
    source: str = "file:test_border_cctv.mp4",
    threshold: float = 0.40,
    skip: int = 3,
    night_mode: int = 0,
    conf: float = 0.35,
):
    await websocket.accept()
    det = get_detector()
    fs = get_face_system()
    am = get_alert_manager()

    # Parse source
    if source.startswith("webcam:"):
        idx = int(source.split(":")[1])
        cap = VideoCaptureHandler(source_type="Webcam", source_param=idx)
    elif source.startswith("file:"):
        fpath = source[5:]
        cap = VideoCaptureHandler(source_type="Upload Footage", source_param=fpath, loop_video=True)
    else:
        # Assume it's an IP camera URL
        cap = VideoCaptureHandler(source_type="IP Camera", source_param=source)

    ok, msg = cap.open()
    if not ok:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": msg
        }))
        await websocket.close()
        return

    # Send ready signal
    await websocket.send_text(json.dumps({"type": "ready", "fps": cap.fps, "message": msg}))

    # Shared state for background AI worker
    import threading
    ai_running = True
    ai_lock = threading.Lock()
    latest_dets = []
    latest_faces = []
    pending_alerts = []
    ai_fps_smooth = 0.0
    last_ai_t = time.time()

    def ai_worker_func():
        nonlocal latest_dets, latest_faces, pending_alerts, ai_fps_smooth, last_ai_t
        while ai_running:
            try:
                f = None
                if hasattr(cap, 'lock') and hasattr(cap, 'latest_frame'):
                    with cap.lock:
                        if cap.latest_frame is not None:
                            f = cap.latest_frame.copy()
                if f is not None:
                    h, w = f.shape[:2]
                    if w > 960:
                        scale = 960.0 / w
                        f_ai = cv2.resize(f, (960, int(h * scale)), interpolation=cv2.INTER_LINEAR)
                    else:
                        scale = 1.0
                        f_ai = f

                    if night_mode:
                        f_ai = det.apply_night_mode(f_ai)

                    d = det.detect(f_ai, conf_threshold=conf)
                    person_boxes = [obj["bbox"] for obj in d if obj["category"] == "person"]
                    fc = fs.process_frame(f_ai, threshold=threshold, person_bboxes=person_boxes)

                    # Scale boxes back to frame coordinates
                    if scale != 1.0:
                        scale_inv = 1.0 / scale
                        for obj in d:
                            obj["bbox"] = [int(v * scale_inv) for v in obj["bbox"]]
                        for face in fc:
                            face["bbox"] = [int(v * scale_inv) for v in face["bbox"]]
                            if face.get("kps"):
                                face["kps"] = [[int(pt[0] * scale_inv), int(pt[1] * scale_inv)] for pt in face["kps"]]

                    for face in fc:
                        if face["is_match"]:
                            tier = face.get("match_tier", "HIGH")
                            snapshot_path = ""

                            # Capture face evidence snapshot on HIGH matches
                            if tier == "HIGH":
                                try:
                                    x1, y1, x2, y2 = face["bbox"]
                                    pad = 20
                                    fh, fw = f.shape[:2]
                                    sx1 = max(0, x1 - pad)
                                    sy1 = max(0, y1 - pad)
                                    sx2 = min(fw, x2 + pad)
                                    sy2 = min(fh, y2 + pad)
                                    crop = f[sy1:sy2, sx1:sx2]
                                    if crop.size > 0:
                                        safe_name = face["name"].replace(" ", "_").replace("/", "-")
                                        ts = time.strftime('%Y%m%d_%H%M%S')
                                        snap_file = f"captured_threats/{safe_name}_{ts}.jpg"
                                        cv2.imwrite(snap_file, crop)
                                        snapshot_path = snap_file
                                except Exception:
                                    pass

                            is_new, record = am.trigger_alert(
                                person_name=face["name"],
                                confidence=face["confidence"],
                                source=source,
                                match_tier=tier,
                                snapshot_path=snapshot_path
                            )
                            if is_new and record:
                                pending_alerts.append((record, face.get("category", "Unknown")))

                    dt_ai = time.time() - last_ai_t
                    last_ai_t = time.time()
                    if dt_ai > 0:
                        ai_fps_smooth = 0.90 * ai_fps_smooth + 0.10 * (1.0 / dt_ai) if ai_fps_smooth > 0 else 1.0 / dt_ai

                    with ai_lock:
                        latest_dets = d
                        latest_faces = fc
            except Exception as ai_err:
                print(f"[AI WORKER WARN] {ai_err}")
            time.sleep(0.01)

    ai_thread = threading.Thread(target=ai_worker_func, daemon=True)
    ai_thread.start()

    fps_smooth = 0.0
    last_frame_t = time.time()

    try:
        while True:
            t0 = time.time()

            # Check if client disconnected
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
            except asyncio.TimeoutError:
                pass
            except WebSocketDisconnect:
                break

            ret, frame = cap.read()
            if not ret or frame is None:
                await asyncio.sleep(0.01)
                continue

            # Auto-downscale display frame if 1080p/4K to 720p for fast network encoding
            h, w = frame.shape[:2]
            if w > 960:
                scale_disp = 960.0 / w
                frame = cv2.resize(frame, (960, int(h * scale_disp)), interpolation=cv2.INTER_AREA)

            if night_mode:
                frame = det.apply_night_mode(frame)

            # Get latest detections & faces from background AI worker
            with ai_lock:
                dets = list(latest_dets)
                fcs = list(latest_faces)

            # Rescale boxes to display resolution if needed
            h_curr, w_curr = frame.shape[:2]
            scale_x = w_curr / w
            scale_y = h_curr / h
            if scale_x != 1.0 or scale_y != 1.0:
                dets_scaled = []
                for d in dets:
                    d_copy = dict(d)
                    b = d["bbox"]
                    d_copy["bbox"] = [int(b[0]*scale_x), int(b[1]*scale_y), int(b[2]*scale_x), int(b[3]*scale_y)]
                    dets_scaled.append(d_copy)
                dets = dets_scaled

                fcs_scaled = []
                for f_item in fcs:
                    f_copy = dict(f_item)
                    b = f_item["bbox"]
                    f_copy["bbox"] = [int(b[0]*scale_x), int(b[1]*scale_y), int(b[2]*scale_x), int(b[3]*scale_y)]
                    fcs_scaled.append(f_copy)
                fcs = fcs_scaled

            # Flag occluded / masked persons
            for d in dets:
                if d["category"] == "person":
                    px1, py1, px2, py2 = d["bbox"]
                    has_face = False
                    for fc in fcs:
                        fx1, fy1, fx2, fy2 = fc["bbox"]
                        if fx1 >= px1 - 35 and fx2 <= px2 + 35 and fy1 >= py1 - 25 and fy2 <= py1 + (py2 - py1) * 0.60:
                            has_face = True
                            break
                    d["face_occluded"] = not has_face

            # Draw AI annotations
            annotated = det.draw_detections(frame, dets)
            annotated = fs.draw_faces(annotated, fcs)

            person_cnt = sum(1 for d in dets if d["category"] == "person")
            vehicle_cnt = sum(1 for d in dets if d["category"] == "vehicle")

            # Check and send any pending alerts
            while pending_alerts:
                record, cat = pending_alerts.pop(0)
                await websocket.send_text(json.dumps({
                    "type": "alert",
                    "alert": record,
                    "category": cat
                }))

            # FPS calculation for Display Stream
            dt = time.time() - last_frame_t
            last_frame_t = time.time()
            if dt > 0:
                fps_smooth = 0.90 * fps_smooth + 0.10 * (1.0 / dt) if fps_smooth > 0 else 1.0 / dt

            # Periodically transmit telemetry
            if int(time.time() * 10) % 5 == 0:
                try:
                    await websocket.send_text(json.dumps({
                        "type": "telemetry",
                        "stream_fps": round(fps_smooth, 1),
                        "ai_fps": round(ai_fps_smooth, 1),
                        "persons": person_cnt,
                        "vehicles": vehicle_cnt
                    }))
                except Exception:
                    pass

            # Subtle HUD overlay with separated Display vs AI Inference throughput
            h_out, w_out = annotated.shape[:2]
            cv2.rectangle(annotated, (0, 0), (w_out, 30), (14, 16, 20), -1)
            hud = f"IBVAP-Lite  |  {time.strftime('%H:%M:%S')}  |  STREAM: {fps_smooth:.0f} FPS  |  AI: {ai_fps_smooth:.0f} FPS  |  Targets: {person_cnt}  |  Vehicles: {vehicle_cnt}"
            clahe_tag = "  |  LOW-LIGHT CLAHE: ON" if night_mode else ""
            cv2.putText(annotated, hud + clahe_tag, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (218, 227, 234), 1, cv2.LINE_AA)

            # Active banner indicator bar
            banner = am.get_active_banner(banner_duration=2.5)
            if banner:
                cv2.rectangle(annotated, (0, 30), (w_out, 62), (25, 25, 165), -1)
                cv2.rectangle(annotated, (0, 30), (w_out, 62), (45, 45, 225), 1)
                banner_text = f"WATCHLIST MATCH: {banner['person'].upper()}  ({banner['confidence']*100:.0f}%)"
                cv2.putText(annotated, banner_text, (10, 52),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

            # Fast JPEG encode (quality 65 for ultra-low latency & 30-60 FPS)
            ret_enc, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 65])
            if ret_enc:
                await websocket.send_bytes(buf.tobytes())

            # Ultra-low async yield (unlocks up to 60 FPS)
            await asyncio.sleep(0.002)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS ERROR] {e}")
        traceback.print_exc()
    finally:
        ai_running = False
        cap.release()

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)

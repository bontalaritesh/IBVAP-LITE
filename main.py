"""
main.py - IBVAP-Lite Standalone / CLI Video Analytics Runner
Runs full real-time pipeline: Ingestion -> YOLOv8 Detection -> InsightFace Watchlist Matching
-> Alert Triggering -> OpenCV Tactical Overlay.
"""

import time
import argparse
import os
import cv2
import numpy as np

from capture import VideoCaptureHandler, create_synthetic_test_video
from detection import ObjectDetector
from faces import FaceRecognitionSystem
from alerts import AlertManager


def draw_hud_and_banner(frame, alert_manager, fps, person_cnt, vehicle_cnt, night_mode, source_label):
    """
    Draws tactical HUD elements, telemetry info, and red alert banner on frame.
    """
    h, w = frame.shape[:2]
    annotated = frame.copy()

    # 1. Top HUD Bar
    cv2.rectangle(annotated, (0, 0), (w, 42), (20, 20, 20), -1)
    cv2.line(annotated, (0, 42), (w, 42), (60, 60, 60), 1)

    status_text = f"IBVAP-Lite v1.0 | {source_label} | {fps:.1f} FPS | NightMode: {'ON' if night_mode else 'OFF'}"
    cv2.putText(annotated, status_text, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1, cv2.LINE_AA)

    counts_text = f"Persons: {person_cnt} | Vehicles: {vehicle_cnt}"
    (cw, ch), _ = cv2.getTextSize(counts_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(annotated, counts_text, (w - cw - 15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # 2. Check for Active Watchlist Match Banner
    banner = alert_manager.get_active_banner(banner_duration=3.0)
    if banner:
        banner_h = 55
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 44), (w, 44 + banner_h), (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)
        cv2.rectangle(annotated, (2, 44), (w - 2, 44 + banner_h), (0, 0, 255), 2)

        alert_msg = f"CRITICAL ALERT: WATCHLIST MATCH DETECTED - {banner['person'].upper()} ({banner['confidence']*100:.1f}%)"
        (bw, bh), _ = cv2.getTextSize(alert_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        tx = max(10, (w - bw) // 2)
        cv2.putText(annotated, alert_msg, (tx, 44 + banner_h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    return annotated


def run_pipeline(
    source_type="Upload Footage",
    source_param="test_border_cctv.mp4",
    match_threshold=0.40,
    skip_frames=3,
    night_mode=False,
    no_gui=False,
    max_frames=None
):
    print("=" * 65)
    print(" IBVAP-Lite: Intelligent Border Video Analytics Platform")
    print(f" Source: {source_type} ({source_param})")
    print(f" Watchlist Threshold: {match_threshold} | Frame Skip (N): {skip_frames}")
    print(f" Night Mode: {night_mode}")
    print("=" * 65)

    print("[1/4] Initializing Video Capture...")
    cap_handler = VideoCaptureHandler(source_type=source_type, source_param=source_param)
    ok, msg = cap_handler.open()
    if not ok:
        print(f"[ERROR] {msg}")
        return

    print("[2/4] Initializing YOLO26 Object Detector...")
    detector = ObjectDetector(model_name="yolo26n.pt")

    print("[3/4] Initializing InsightFace Watchlist System...")
    face_system = FaceRecognitionSystem()
    face_system.initialize_model()
    if not face_system.enrolled_faces:
        face_system.enroll_watchlist("watchlist_images")

    print("[4/4] Initializing Threat Alert Manager...")
    alert_manager = AlertManager(cooldown_seconds=4.0)

    print("\n[READY] Starting Surveillance Pipeline. Press 'q' to stop.\n")

    frame_idx = 0
    cached_face_results = []
    fps_smooth = 0.0

    try:
        while True:
            ret, frame = cap_handler.read()
            if not ret or frame is None:
                print("[INFO] Video stream ended or completed.")
                break

            frame_idx += 1
            start_loop = time.time()

            # 1. Night Mode Pre-processing (CLAHE)
            processed_frame = frame
            if night_mode:
                processed_frame = detector.apply_night_mode(frame)

            # 2. YOLOv8 Object Detection (Every Frame)
            detections = detector.detect(processed_frame, conf_threshold=0.35)
            annotated_frame = detector.draw_detections(processed_frame, detections)

            # 3. InsightFace Face Recognition (Every N-th Frame)
            if frame_idx % skip_frames == 0 or frame_idx == 1:
                cached_face_results = face_system.process_frame(processed_frame, threshold=match_threshold)
                for face in cached_face_results:
                    if face["is_match"]:
                        alert_manager.trigger_alert(
                            person_name=face["name"],
                            confidence=face["confidence"],
                            source=source_type
                        )

            # Draw cached face boxes
            annotated_frame = face_system.draw_faces(annotated_frame, cached_face_results)

            person_cnt = sum(1 for d in detections if d["category"] == "person")
            vehicle_cnt = sum(1 for d in detections if d["category"] == "vehicle")

            loop_duration = time.time() - start_loop
            instant_fps = 1.0 / max(loop_duration, 0.001)
            fps_smooth = 0.9 * fps_smooth + 0.1 * instant_fps if fps_smooth > 0 else instant_fps

            final_display = draw_hud_and_banner(
                annotated_frame, alert_manager, fps_smooth,
                person_cnt, vehicle_cnt, night_mode, source_type
            )

            if frame_idx % 25 == 0 or frame_idx == 1:
                print(f"[Frame {frame_idx:03d}] FPS: {fps_smooth:.1f} | Persons: {person_cnt} | Vehicles: {vehicle_cnt} | Threat Active: {alert_manager.get_active_banner() is not None}")

            if not no_gui:
                cv2.imshow("IBVAP-Lite Border Surveillance Feed", final_display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("[INFO] User terminated stream.")
                    break
                elif key == ord('n'):
                    night_mode = not night_mode
                    print(f"[INFO] Toggled Night Mode: {night_mode}")

            if max_frames and frame_idx >= max_frames:
                print(f"[INFO] Reached test frame limit of {max_frames} frames.")
                break

    finally:
        cap_handler.release()
        if not no_gui:
            cv2.destroyAllWindows()
        print("\n--- Pipeline Session Statistics ---")
        stats = alert_manager.get_stats()
        print(f" Total Threat Alerts: {stats['total_alerts']}")
        print(f" Unique Watchlist Hits: {stats['unique_targets']}")
        print(f" Highest Confidence: {stats['highest_confidence']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IBVAP-Lite Border Video Analytics Runner")
    parser.add_argument("--source-type", choices=["IP Camera", "Webcam", "Upload Footage"], default="Upload Footage")
    parser.add_argument("--source", default="test_border_cctv.mp4", help="URL, webcam index, or video file path")
    parser.add_argument("--threshold", type=float, default=0.40, help="Face cosine match threshold")
    parser.add_argument("--skip-frames", type=int, default=3, help="Run face analysis every Nth frame")
    parser.add_argument("--night-mode", action="store_true", help="Enable CLAHE night enhancement")
    parser.add_argument("--no-gui", action="store_true", help="Run in headless mode without cv2 window")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to process")
    args = parser.parse_args()

    if args.source == "test_border_cctv.mp4" and not os.path.exists("test_border_cctv.mp4"):
        from capture import create_synthetic_test_video
        create_synthetic_test_video("test_border_cctv.mp4")

    run_pipeline(
        source_type=args.source_type,
        source_param=args.source,
        match_threshold=args.threshold,
        skip_frames=args.skip_frames,
        night_mode=args.night_mode,
        no_gui=args.no_gui,
        max_frames=args.max_frames
    )

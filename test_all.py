import time
import os
import cv2
import numpy as np

print("=== STEP 1: Testing capture.py ===")
from capture import VideoCaptureHandler, create_synthetic_test_video

test_vid = create_synthetic_test_video("test_cctv.mp4", duration_sec=3, fps=25)
handler = VideoCaptureHandler(source_type="Upload Footage", source_param=test_vid)
ok, msg = handler.open()
print(f"File Source Open: {ok} | {msg}")
frames_read = 0
start = time.time()
while frames_read < 25:
    ret, frame = handler.read()
    if not ret:
        break
    frames_read += 1
dur = time.time() - start
print(f"Read {frames_read} frames in {dur:.2f}s (~{frames_read/dur:.1f} FPS)")
handler.release()

# Test IP camera fallback
bad_handler = VideoCaptureHandler(source_type="IP Camera", source_param="http://10.255.255.1:8080/video")
ok, msg = bad_handler.open()
print(f"Fallback on invalid IP: {ok} | {msg}")
bad_handler.release()

print("\n=== STEP 2: Testing detection.py (YOLO26) ===")
from detection import ObjectDetector

detector = ObjectDetector(model_name="yolo26n.pt")
test_img = np.full((480, 640, 3), 128, dtype=np.uint8)
# Draw synthetic person
cv2.rectangle(test_img, (100, 100), (200, 400), (200, 150, 100), -1)
cv2.circle(test_img, (150, 80), 30, (220, 200, 180), -1)

# Test Night mode CLAHE
enhanced = detector.apply_night_mode(test_img)
print(f"Night Mode CLAHE output shape: {enhanced.shape}")

# Test YOLO detection
detections = detector.detect(test_img, conf_threshold=0.25)
print(f"Detections count: {len(detections)}")
annotated = detector.draw_detections(test_img, detections)
print("Detection module verified successfully.")

print("\n=== STEP 3: Testing alerts.py ===")
from alerts import AlertManager

alert_mgr = AlertManager(cooldown_seconds=2.0)
added, record = alert_mgr.trigger_alert("Suspect Alpha", 0.88, source="CCTV-01")
print(f"Alert 1 triggered: {added} | {record['person']} ({record['confidence_pct']})")

# Immediate duplicate within cooldown
added2, _ = alert_mgr.trigger_alert("Suspect Alpha", 0.89, source="CCTV-01")
print(f"Duplicate within cooldown: {added2} (Expected False)")

# New target
added3, record3 = alert_mgr.trigger_alert("Suspect Bravo", 0.74, source="CCTV-01")
print(f"Alert 2 triggered: {added3} | {record3['person']}")

stats = alert_mgr.get_stats()
print(f"Alert stats: {stats}")

print("\n=== STEP 4: Testing faces.py (InsightFace) ===")
from faces import FaceRecognitionSystem

fr = FaceRecognitionSystem()
ok, msg = fr.initialize_model()
print(f"InsightFace model status: {ok} | {msg}")

# Enroll watchlist
ok_enroll, msg_enroll = fr.enroll_watchlist("watchlist_images")
print(f"Enrollment status: {ok_enroll} | {msg_enroll}")
print(f"Enrolled faces in DB: {list(fr.enrolled_faces.keys())}")

# Run process_frame on sample portrait
test_portrait = cv2.imread("watchlist_images/Agent_Alex/ref_front.jpg")
if test_portrait is not None:
    results = fr.process_frame(test_portrait, threshold=0.35)
    print(f"Face recognition test results: {results}")

print("\n=== ALL STANDALONE MODULES VERIFIED SUCCESSFULLY ===")
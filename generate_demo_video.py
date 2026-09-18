import cv2
import numpy as np
import time

def generate_cctv_demo_video(output_path="test_border_cctv.mp4", duration_sec=8, fps=25):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    width, height = 720, 480
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Load enrolled face image
    face_img = cv2.imread("watchlist_images/Suspect_Alpha/ref1.jpg")
    if face_img is None:
        face_crop = np.full((120, 100, 3), 150, dtype=np.uint8)
    else:
        # Crop face from portrait
        face_crop = face_img[180:380, 210:350]
        face_crop = cv2.resize(face_crop, (100, 130))

    total_frames = int(duration_sec * fps)

    for i in range(total_frames):
        # Create CCTV background
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        
        # Border road/checkpoint pavement
        cv2.rectangle(frame, (0, 300), (width, height), (55, 55, 55), -1)
        cv2.line(frame, (0, 300), (width, 300), (80, 80, 80), 2)
        
        # Checkpoint barrier
        cv2.line(frame, (80, 200), (80, 350), (200, 200, 200), 5)
        cv2.line(frame, (80, 250), (320, 270), (0, 0, 220), 4)

        # 1. Moving Person (Suspect Alpha) walking across checkpoint
        p_x = int(50 + (width - 250) * (i / total_frames))
        p_y = 170

        # Person body
        cv2.rectangle(frame, (p_x, p_y + 110), (p_x + 100, p_y + 260), (90, 70, 60), -1)
        # Suspect real face overlay
        fh, fw = face_crop.shape[:2]
        frame[p_y:p_y+fh, p_x:p_x+fw] = face_crop

        # 2. Moving Patrol Vehicle on lower road
        v_x = int(width - 180 - (width - 100) * (i / total_frames))
        v_y = 340
        cv2.rectangle(frame, (v_x, v_y), (v_x + 180, v_y + 70), (70, 110, 150), -1)
        cv2.rectangle(frame, (v_x + 40, v_y - 25), (v_x + 140, v_y), (90, 130, 170), -1)
        cv2.circle(frame, (v_x + 40, v_y + 70), 18, (20, 20, 20), -1)
        cv2.circle(frame, (v_x + 140, v_y + 70), 18, (20, 20, 20), -1)

        # CCTV timestamp and sector label
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S") + f".{i%fps:02d}"
        cv2.putText(frame, f"[IBVAP BORDER CAM-01] {timestamp} [SECTOR 4 CHECKPOINT]", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 1, cv2.LINE_AA)
        out.write(frame)

    out.release()
    print(f"Generated CCTV Demo video: {output_path}")

generate_cctv_demo_video("test_border_cctv.mp4")
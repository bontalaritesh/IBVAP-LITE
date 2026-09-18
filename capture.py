"""
capture.py - Universal Video Capture & Ingestion Module for IBVAP-Lite
Handles Phone IP Camera streams, Webcams (indexes 0/1/etc.), and Uploaded Footage
with native FPS pacing, fast network reachability pre-check, and automatic fallback.
"""

import time
import os
import socket
import threading
import urllib.request
from urllib.parse import urlparse
import cv2
import numpy as np


def is_network_stream_reachable(url, timeout_sec=1.5):
    """
    Performs a fast socket reachability pre-check on IP camera host/port
    to avoid long OpenCV network blocking if the phone IP is unreachable.
    """
    try:
        url_str = str(url).strip()
        if not url_str:
            return False
        if "://" not in url_str:
            url_to_parse = "http://" + url_str
        else:
            url_to_parse = url_str

        parsed = urlparse(url_to_parse)
        host = parsed.hostname
        port = parsed.port
        if not port:
            if parsed.scheme == "rtsp":
                port = 554
            elif parsed.scheme == "https":
                port = 443
            else:
                port = 8080 # Default for most IP webcam apps

        if not host:
            return False

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout_sec)
        result = sock.connect_ex((host, int(port)))
        sock.close()
        return (result == 0)
    except Exception:
        return False


class VideoCaptureHandler:
    """
    Unified video capture handler supporting:
    1. Phone IP Webcam (native pure-Python MJPEG stream reader with auto-reconnect)
    2. Local Webcams (index 0, 1, etc.) with automatic fallback
    3. Uploaded Video Files (.mp4, .avi, .mkv, .mov) with native FPS pacing
    """

    def __init__(self, source_type="Webcam", source_param=0, loop_video=True):
        self.source_type = source_type
        self.source_param = source_param
        self.loop_video = loop_video
        self.cap = None
        self.fps = 30.0
        self.frame_delay = 1.0 / 30.0
        self.last_frame_time = 0.0
        self.is_opened = False
        self.status_message = ""
        self.total_frames = 0
        self.current_frame_idx = 0
        self.thread = None
        self.running = False
        self.lock = threading.Lock()
        self.latest_frame = None

    def open(self):
        """
        Opens the selected video stream with fallback mechanisms.
        Returns: (success: bool, message: str)
        """
        self.release()

        try:
            if self.source_type == "IP Camera":
                url = str(self.source_param).strip()
                if not url:
                    self.status_message = "Error: IP Camera URL is empty."
                    return False, self.status_message

                # Auto-format URL schema if missing
                if not url.startswith("http://") and not url.startswith("https://") and not url.startswith("rtsp://"):
                    url = f"http://{url}"

                # Auto-append /video if user pasted base URL like http://192.168.1.x:8080/
                url_clean = url.rstrip("/")
                if not any(url_clean.endswith(ext) for ext in ["/video", "/mjpeg", ".mjpg", "/shot.jpg", "/videofeed"]):
                    url = f"{url_clean}/video"

                # Fast reachability check
                print(f"[INFO] Checking reachability of IP stream: {url}")
                reachable = is_network_stream_reachable(url, timeout_sec=1.5)
                
                if not reachable:
                    self.status_message = (
                        f"IP Camera at '{url}' is unreachable (device offline or wrong port). "
                        "Falling back to local Webcam 0."
                    )
                    print(f"[WARN] {self.status_message}")
                    # DirectShow backend on Windows
                    self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(0)
                    if self.cap.isOpened():
                        self.source_type = "Webcam"
                        self.source_param = 0
                    else:
                        self.status_message = f"IP Camera '{url}' unreachable and local Webcam 0 unavailable. Please choose 'Upload Footage'."
                        return False, self.status_message
                else:
                    # Pure-Python native MJPEG stream reader (never crashes on packet drops)
                    self.running = True
                    self.is_opened = True
                    self.fps = 30.0
                    self.status_message = f"Stream active (IP Camera @ 30.0 FPS)"
                    self.thread = threading.Thread(target=self._mjpeg_worker, args=(url,), daemon=True)
                    self.thread.start()
                    return True, self.status_message

            elif self.source_type == "Webcam":
                try:
                    idx = int(self.source_param)
                except (ValueError, TypeError):
                    idx = 0

                self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(idx)
                try:
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass

                if not self.cap.isOpened() and idx != 0:
                    print(f"[WARN] Webcam index {idx} failed. Trying fallback to index 0...")
                    self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(0)
                    self.source_param = 0

                if not self.cap.isOpened():
                    self.status_message = f"Webcam {idx} is unavailable. Please check camera permissions or select 'Upload Footage'."
                    return False, self.status_message

            elif self.source_type == "Upload Footage":
                file_path = str(self.source_param)
                if not os.path.exists(file_path):
                    self.status_message = f"Video file not found at: {file_path}"
                    return False, self.status_message

                self.cap = cv2.VideoCapture(file_path)
                if not self.cap.isOpened():
                    self.status_message = f"Failed to decode video file: {file_path}. Please try another .mp4/.avi video."
                    return False, self.status_message

            else:
                self.status_message = f"Unsupported source type: {self.source_type}"
                return False, self.status_message

            raw_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.fps = raw_fps if (raw_fps and 1.0 <= raw_fps <= 120.0) else 30.0
            self.frame_delay = 1.0 / self.fps
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.is_opened = True
            self.last_frame_time = time.time()
            if not self.status_message:
                self.status_message = f"Stream active ({self.source_type} @ {self.fps:.1f} FPS)"

            # Launch background thread for live streams to avoid buffer backlog
            if self.source_type in ["IP Camera", "Webcam"]:
                self.running = True
                self.thread = threading.Thread(target=self._capture_worker, daemon=True)
                self.thread.start()

            return True, self.status_message

        except Exception as e:
            self.status_message = f"Capture initialization error: {str(e)}"
            return False, self.status_message

    def _mjpeg_worker(self, url):
        """High-throughput, zero-lag MJPEG stream reader with auto-reconnect and frame dropping."""
        while self.running:
            stream = None
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                stream = urllib.request.urlopen(req, timeout=4.0)
                buffer = bytearray()
                while self.running:
                    chunk = stream.read(65536)
                    if not chunk:
                        break
                    buffer.extend(chunk)

                    # Keep memory bounded
                    if len(buffer) > 2500000:
                        buffer = buffer[-500000:]

                    # Extract all complete frames; only keep and decode the freshest one
                    latest_jpg = None
                    while True:
                        a = buffer.find(b'\xff\xd8')
                        if a == -1:
                            buffer.clear()
                            break
                        b = buffer.find(b'\xff\xd9', a + 2)
                        if b == -1:
                            if a > 0:
                                del buffer[:a]
                            break

                        latest_jpg = bytes(buffer[a:b+2])
                        del buffer[:b+2]

                    if latest_jpg is not None:
                        frame = cv2.imdecode(np.frombuffer(latest_jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame is not None:
                            h, w = frame.shape[:2]
                            if w > 800:
                                scale = 800.0 / w
                                frame = cv2.resize(frame, (800, int(h * scale)), interpolation=cv2.INTER_LINEAR)
                            with self.lock:
                                self.latest_frame = frame
                time.sleep(0.01)
            except Exception as e:
                time.sleep(0.5)
            finally:
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass

    def _capture_worker(self):
        """Continuously pulls frames into latest_frame buffer with fast pre-scaling."""
        while self.running:
            try:
                if not self.cap or not self.cap.isOpened():
                    time.sleep(0.005)
                    continue
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    # Pre-scale if large HD frame from phone to 800px max width for high FPS
                    h, w = frame.shape[:2]
                    if w > 800:
                        scale = 800.0 / w
                        frame = cv2.resize(frame, (800, int(h * scale)), interpolation=cv2.INTER_LINEAR)
                    with self.lock:
                        self.latest_frame = frame
                else:
                    time.sleep(0.001)
            except Exception:
                time.sleep(0.01)

    def read(self):
        """
        Reads the next video frame.
        For live IP streams / webcams, returns the newest frame instantly.
        In 'Upload Footage' mode, applies real-time FPS delay pacing to simulate live playback.
        Returns: (success: bool, frame: np.ndarray or None)
        """
        if self.source_type in ["IP Camera", "Webcam"]:
            with self.lock:
                if self.latest_frame is not None:
                    self.current_frame_idx += 1
                    return True, self.latest_frame.copy()
            # If worker hasn't grabbed first frame yet, read directly once safely
            if self.cap and self.cap.isOpened():
                try:
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        with self.lock:
                            self.latest_frame = frame
                        return True, frame
                except Exception:
                    pass
            return False, None

        if self.cap is None or not self.cap.isOpened():
            return False, None

        if self.source_type == "Upload Footage":
            elapsed = time.time() - self.last_frame_time
            sleep_time = self.frame_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            self.last_frame_time = time.time()

        ret, frame = self.cap.read()

        if not ret:
            if self.source_type == "Upload Footage" and self.loop_video:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                self.current_frame_idx = 0
            else:
                return False, None

        if ret and frame is not None:
            with self.lock:
                self.latest_frame = frame
            self.current_frame_idx += 1
            return True, frame

        return False, None

    def release(self):
        """Releases video capture hardware / memory and stops worker thread."""
        self.running = False
        if self.thread and self.thread.is_alive():
            try:
                self.thread.join(timeout=0.3)
            except Exception:
                pass
        self.thread = None
        self.latest_frame = None

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.is_opened = False


def create_synthetic_test_video(output_path="test_feed.mp4", duration_sec=6, fps=25):
    """Generates a sample CCTV test video with simulated bounding motion and faces."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width, height = 640, 480
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = int(duration_sec * fps)
    for i in range(total_frames):
        frame = np.full((height, width, 3), 35, dtype=np.uint8)
        for y in range(0, height, 40):
            cv2.line(frame, (0, y), (width, y), (48, 48, 48), 1)
        for x in range(0, width, 40):
            cv2.line(frame, (x, 0), (x, height), (48, 48, 48), 1)
        
        pos_x = int(60 + (width - 240) * (i / total_frames))
        pos_y = 150
        cv2.rectangle(frame, (pos_x, pos_y + 60), (pos_x + 90, pos_y + 220), (140, 120, 90), -1)
        cv2.circle(frame, (pos_x + 45, pos_y + 30), 28, (210, 190, 170), -1)
        cv2.circle(frame, (pos_x + 35, pos_y + 25), 4, (40, 40, 40), -1)
        cv2.circle(frame, (pos_x + 55, pos_y + 25), 4, (40, 40, 40), -1)
        cv2.ellipse(frame, (pos_x + 45, pos_y + 38), (10, 5), 0, 0, 180, (40, 40, 40), 2)

        v_x = int(width - 150 - (width - 100) * (i / total_frames))
        v_y = 320
        cv2.rectangle(frame, (v_x, v_y), (v_x + 160, v_y + 70), (70, 110, 160), -1)
        cv2.circle(frame, (v_x + 35, v_y + 70), 16, (20, 20, 20), -1)
        cv2.circle(frame, (v_x + 125, v_y + 70), 16, (20, 20, 20), -1)

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S") + f".{i%fps:02d}"
        cv2.putText(frame, f"[IBVAP CCTV-CAM-01] {timestamp} [SYNTHETIC TEST FEED]", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        out.write(frame)
        
    out.release()
    return output_path
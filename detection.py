"""
detection.py - YOLOv8 Object Detection, Persistent Tracking & Night Vision Module
Performs high-speed object detection for 'person', 'car', 'truck', 'bus', 'motorcycle'
with Multi-Object Tracking (persistent Track IDs) and CLAHE Night Vision enhancement.
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO


class SimpleTracker:
    """
    Lightweight, fast centroid & IOU multi-object tracker
    assigning persistent tracking IDs across consecutive frames.
    """

    def __init__(self, max_disappeared=15, max_distance=80):
        self.next_object_id = 1
        self.objects = {}  # {id: centroid}
        self.bboxes = {}   # {id: bbox}
        self.disappeared = {}  # {id: frame_count}

    def register(self, centroid, bbox):
        obj_id = self.next_object_id
        self.objects[obj_id] = centroid
        self.bboxes[obj_id] = bbox
        self.disappeared[obj_id] = 0
        self.next_object_id += 1
        return obj_id

    def deregister(self, obj_id):
        self.objects.pop(obj_id, None)
        self.bboxes.pop(obj_id, None)
        self.disappeared.pop(obj_id, None)

    def update(self, detections):
        """
        Updates tracker state with new bounding boxes.
        Returns: list of detections augmented with "track_id".
        """
        if len(detections) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > 15:
                    self.deregister(obj_id)
            return []

        input_centroids = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            input_centroids.append((cx, cy))

        if len(self.objects) == 0:
            for i, det in enumerate(detections):
                obj_id = self.register(input_centroids[i], det["bbox"])
                det["track_id"] = f"TRK-{obj_id:02d}"
            return detections

        # Compute pairwise distance matrix
        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        D = np.zeros((len(object_centroids), len(input_centroids)), dtype=np.float32)
        for r, (ox, oy) in enumerate(object_centroids):
            for c, (ix, iy) in enumerate(input_centroids):
                D[r, c] = np.hypot(ox - ix, oy - iy)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > 100:  # Max distance threshold
                continue

            obj_id = object_ids[row]
            self.objects[obj_id] = input_centroids[col]
            self.bboxes[obj_id] = detections[col]["bbox"]
            self.disappeared[obj_id] = 0
            detections[col]["track_id"] = f"TRK-{obj_id:02d}"

            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(0, D.shape[0])).difference(used_rows)
        unused_cols = set(range(0, D.shape[1])).difference(used_cols)

        for row in unused_rows:
            obj_id = object_ids[row]
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > 15:
                self.deregister(obj_id)

        for col in unused_cols:
            obj_id = self.register(input_centroids[col], detections[col]["bbox"])
            detections[col]["track_id"] = f"TRK-{obj_id:02d}"

        return detections


class ObjectDetector:
    """
    YOLOv8-based object detection engine for border surveillance.
    Detects humans and vehicles, tracks entities, and applies night vision.
    """

    TARGET_CLASSES = {
        0: "person",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck"
    }

    COLORS = {
        "person": (255, 200, 0),     # Cyan / Sky Blue
        "car": (0, 215, 255),        # Amber / Gold
        "truck": (0, 165, 255),      # Orange
        "bus": (255, 120, 0),        # Violet
        "motorcycle": (180, 255, 0), # Lime
        "vehicle": (0, 215, 255)
    }

    def __init__(self, model_name="yolo26n.onnx"):
        if not os.path.exists(model_name):
            for candidate in ["yolo26n.onnx", "yolo26n.pt", "yolo11n.onnx", "yolo11n.pt", "yolov8n.onnx", "yolov8n.pt"]:
                if os.path.exists(candidate):
                    model_name = candidate
                    break
        self.model_name = model_name
        self.model = None
        self.tracker = SimpleTracker()
        self._load_model()

    def _load_model(self):
        try:
            print(f"[INFO] Loading YOLO model '{self.model_name}'...")
            self.model = YOLO(self.model_name)
            print(f"[INFO] YOLO detection engine ready ({self.model_name}).")
        except Exception as e:
            print(f"[ERROR] Failed to load YOLO model: {e}")
            raise

    @staticmethod
    def apply_night_mode(frame, clip_limit=3.0, tile_grid_size=(8, 8)):
        """Enhances low-light CCTV footage using CLAHE."""
        if frame is None:
            return None
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def detect(self, frame, conf_threshold=0.35, track=True):
        """Runs YOLOv8 object detection on the frame."""
        if self.model is None or frame is None:
            return []

        target_ids = list(self.TARGET_CLASSES.keys())
        results = self.model.predict(
            source=frame,
            classes=target_ids,
            conf=conf_threshold,
            imgsz=480,
            verbose=False
        )

        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                
                label = self.TARGET_CLASSES.get(cls_id, "unknown")
                category = "person" if cls_id == 0 else "vehicle"

                detections.append({
                    "class_id": cls_id,
                    "label": label,
                    "confidence": conf,
                    "bbox": xyxy,
                    "category": category,
                    "track_id": "TRK-00"
                })

        if track and detections:
            detections = self.tracker.update(detections)

        return detections

    def draw_detections(self, frame, detections, draw_vehicles=True, draw_persons=True):
        """Draws high-tech tactical bounding brackets and tracking labels."""
        if frame is None:
            return frame

        annotated = frame.copy()
        for det in detections:
            cat = det["category"]
            if cat == "person" and not draw_persons:
                continue
            if cat == "vehicle" and not draw_vehicles:
                continue

            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            conf = det["confidence"]
            trk_id = det.get("track_id", "")
            is_occluded = det.get("face_occluded", False) and label == "person"

            if is_occluded:
                # Flag non-cooperative / masked / turned targets
                tag_text = f"[{trk_id}] PERSON (FACE OCCLUDED) {conf*100:.0f}%" if trk_id else f"PERSON (FACE OCCLUDED) {conf*100:.0f}%"
                color = (35, 140, 220)  # Muted amber warning
            else:
                tag_text = f"[{trk_id}] {label.upper()} {conf*100:.0f}%" if trk_id else f"{label.upper()} {conf*100:.0f}%"
                color = self.COLORS.get(label, (0, 255, 0))

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)

            # Tactical corner brackets
            length = min(16, (x2 - x1) // 4, (y2 - y1) // 4)
            if length > 3:
                cv2.line(annotated, (x1, y1), (x1 + length, y1), color, 2)
                cv2.line(annotated, (x1, y1), (x1 + length, y1), color, 2)
                cv2.line(annotated, (x2, y1), (x2 - length, y1), color, 2)
                cv2.line(annotated, (x2, y1), (x2, y1 + length), color, 2)
                cv2.line(annotated, (x1, y2), (x1 + length, y2), color, 2)
                cv2.line(annotated, (x1, y2), (x1, y2 - length), color, 2)
                cv2.line(annotated, (x2, y2), (x2 - length, y2), color, 2)
                cv2.line(annotated, (x2, y2), (x2, y2 - length), color, 2)

            # Tag
            (w, h), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - h - 6)), (x1 + w + 8, y1), color, -1)
            cv2.putText(annotated, tag_text, (x1 + 4, max(h + 2, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1, cv2.LINE_AA)

        return annotated
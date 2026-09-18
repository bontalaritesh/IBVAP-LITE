"""
faces.py - Advanced Biometric Face Recognition & Watchlist Enrollment System
Powered by InsightFace (ArcFace 512-d deep embeddings & RetinaFace).
Supports dynamic photo uploads for friends/suspects, multi-photo averaging,
live face cropping, category classification, and high-accuracy cosine matching.
"""

import os
import glob
import json
import pickle
import uuid
from datetime import datetime
import cv2
import numpy as np
import time
from collections import deque
import database



class LivenessTracker:
    """
    Real-time facial micro-motion and landmark dynamics analyzer for anti-spoofing.
    Evaluates eye-nose geometry ratios, 3D yaw perspective shifts, and natural micro-jitter
    across consecutive video frames. Flat 2D printed photos or static phone screens lack
    depth parallax and micro-motion, failing the liveness gate.
    """
    def __init__(self, buffer_size=16):
        self.buffer_size = buffer_size
        self.tracks = {}  # {track_key: {"history": deque, "last_t": float}}

    def update(self, bbox, kps):
        if kps is None or len(kps) < 5:
            return True, 1.0, "UNKNOWN"

        # Centroid key for tracking face across frames
        cx = int((bbox[0] + bbox[2]) / 2 / 35) * 35
        cy = int((bbox[1] + bbox[3]) / 2 / 35) * 35
        key = (cx, cy)

        lx, ly = kps[0]
        rx, ry = kps[1]
        nx, ny = kps[2]
        lmx, lmy = kps[3]
        rmx, rmy = kps[4]

        # 1. Inter-pupillary distance
        d_eyes = np.hypot(rx - lx, ry - ly) + 1e-5
        # 2. Nose to eyes distance ratio (sensitive to 3D head yaw rotation)
        d_nose_l = np.hypot(nx - lx, ny - ly)
        d_nose_r = np.hypot(nx - rx, ny - ry)
        yaw_ratio = (d_nose_l - d_nose_r) / d_eyes
        # 3. Vertical eye-mouth aspect
        my_avg = (lmy + rmy) / 2.0
        ey_avg = (ly + ry) / 2.0
        aspect = (my_avg - ey_avg) / d_eyes

        now = time.time()
        if key not in self.tracks:
            self.tracks[key] = {"history": deque(maxlen=self.buffer_size), "last_t": now}
        self.tracks[key]["history"].append((yaw_ratio, aspect))
        self.tracks[key]["last_t"] = now

        # Prune tracks older than 2 seconds
        dead_keys = [k for k, v in self.tracks.items() if now - v["last_t"] > 2.0]
        for k in dead_keys:
            del self.tracks[k]

        hist = self.tracks[key]["history"]
        if len(hist) < 6:
            return True, 0.90, "VERIFYING"

        yaws = [h[0] for h in hist]
        aspects = [h[1] for h in hist]
        var_yaw = float(np.std(yaws))
        var_aspect = float(np.std(aspects))
        total_dynamic = var_yaw + var_aspect

        # If variance is virtually zero across >=6 frames, flag as static spoof
        if total_dynamic < 0.0025:
            return False, total_dynamic, "SPOOF_SUSPECTED"

        return True, total_dynamic, "LIVE"


class FaceRecognitionSystem:
    """
    Military-grade face detection, enrollment, and watchlist matching system.
    """

    def __init__(self, model_name="buffalo_sc", db_path="watchlist_db.pkl", meta_path="watchlist_meta.json", models_root=".insightface"):
        self.model_name = model_name
        self.db_path = db_path
        self.meta_path = meta_path
        self.models_root = os.path.abspath(models_root)
        self.app = None
        self.enrolled_faces = {}  # {person_id: normalized_embedding_vector}
        self.enrolled_meta = {}   # {person_id: {name, category, notes, created_at, photo_count, thumb_path}}
        self.is_ready = False
        self.status_message = ""
        self.liveness = LivenessTracker()

        os.makedirs(self.models_root, exist_ok=True)
        os.makedirs("static/thumbnails", exist_ok=True)
        os.makedirs("watchlist_images", exist_ok=True)
        os.makedirs("captured_threats", exist_ok=True)

        self.load_database()

    def initialize_model(self):
        """Initializes InsightFace RetinaFace detector and ArcFace recognizer."""
        if self.app is not None:
            return True, "Model already active"

        try:
            from insightface.app import FaceAnalysis
            print(f"[INFO] Initializing InsightFace model suite ({self.model_name})...")
            providers = ['CPUExecutionProvider']
            try:
                import onnxruntime as ort
                if 'CUDAExecutionProvider' in ort.get_available_providers():
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            except Exception:
                pass

            self.app = FaceAnalysis(
                name=self.model_name,
                root=self.models_root,
                allowed_modules=['detection', 'recognition'],
                providers=providers
            )
            self.app.prepare(ctx_id=0 if 'CUDAExecutionProvider' in providers else -1, det_size=(640, 640), det_thresh=0.30)
            self.is_ready = True
            self.status_message = f"InsightFace ({self.model_name}) active ({providers[0]})"
            print(f"[INFO] {self.status_message}")
            return True, self.status_message

        except Exception as e:
            print(f"[WARN] Initializing {self.model_name} failed: {e}. Falling back to 'buffalo_sc'...")
            try:
                from insightface.app import FaceAnalysis
                self.app = FaceAnalysis(name="buffalo_sc", root=self.models_root, providers=['CPUExecutionProvider'])
                self.app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.30)
                self.is_ready = True
                self.status_message = "InsightFace (buffalo_sc fallback) active"
                print(f"[INFO] {self.status_message}")
                return True, self.status_message
            except Exception as e2:
                self.is_ready = False
                self.status_message = f"InsightFace failed: {e2}"
                return False, self.status_message

    def enroll_single_photo(self, name, category="VIP / Friend", notes="", image_bytes=None, image_bgr=None, filename=None):
        """
        Enrolls or appends a photo for a given person identity.
        Generates face crop thumbnail, extracts 512-d embedding, and updates database.
        """
        ok, msg = self.initialize_model()
        if not ok:
            return False, msg, None

        if image_bytes is not None:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif image_bgr is not None:
            img = image_bgr
        elif filename is not None and os.path.exists(filename):
            img = cv2.imread(filename)
        else:
            return False, "No valid image data provided", None

        if img is None:
            return False, "Failed to decode image", None

        # Run face detection
        faces = self.app.get(img)
        if not faces:
            return False, "No human face detected in uploaded photo. Please ensure face is clearly visible and well-lit.", None

        # Pick largest/most prominent face
        best_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        emb = best_face.normed_embedding if hasattr(best_face, 'normed_embedding') else best_face.embedding
        emb = emb / np.linalg.norm(emb)

        # Generate face crop thumbnail
        bbox = [int(v) for v in best_face.bbox]
        x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(img.shape[1], bbox[2]), min(img.shape[0], bbox[3])
        # Add slight margin to crop
        pad_x = int((x2 - x1) * 0.15)
        pad_y = int((y2 - y1) * 0.15)
        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(img.shape[1], x2 + pad_x)
        crop_y2 = min(img.shape[0], y2 + pad_y)
        face_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]

        # Check if person already exists by name
        person_id = None
        for pid, meta in self.enrolled_meta.items():
            if meta["name"].strip().lower() == name.strip().lower():
                person_id = pid
                break

        if not person_id:
            person_id = str(uuid.uuid4())[:8]
            thumb_filename = f"static/thumbnails/{person_id}.jpg"
            if face_crop.size > 0:
                cv2.imwrite(thumb_filename, face_crop)

            # Store as multi-angle matrix (1, 512)
            self.enrolled_faces[person_id] = emb.reshape(1, 512)
            self.enrolled_meta[person_id] = {
                "id": person_id,
                "name": name.strip(),
                "category": category,
                "notes": notes,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "photo_count": 1,
                "thumb_url": f"/static/thumbnails/{person_id}.jpg",
                "raw_embeddings": [emb]
            }
        else:
            # Multi-angle exemplar gallery: preserve each angle embedding independently
            meta = self.enrolled_meta[person_id]
            if "raw_embeddings" not in meta:
                existing_embs = self.enrolled_faces[person_id]
                if existing_embs.ndim == 1:
                    meta["raw_embeddings"] = [existing_embs]
                else:
                    meta["raw_embeddings"] = [existing_embs[i] for i in range(len(existing_embs))]

            meta["raw_embeddings"].append(emb)
            # Stack into multi-angle exemplar matrix (N, 512)
            self.enrolled_faces[person_id] = np.vstack(meta["raw_embeddings"])
            meta["photo_count"] = len(meta["raw_embeddings"])
            meta["category"] = category
            if notes:
                meta["notes"] = notes
            # Update thumbnail if not already present or if better face
            thumb_filename = f"static/thumbnails/{person_id}.jpg"
            if face_crop.size > 0 and (not os.path.exists(thumb_filename) or meta["photo_count"] == 1):
                cv2.imwrite(thumb_filename, face_crop)

        # Also save reference photo to watchlist_images folder
        person_dir = os.path.join("watchlist_images", name.replace(" ", "_"))
        os.makedirs(person_dir, exist_ok=True)
        img_save_path = os.path.join(person_dir, f"photo_{int(datetime.now().timestamp())}.jpg")
        cv2.imwrite(img_save_path, img)

        self.save_database()
        return True, f"Successfully enrolled '{name}' ({category}) with {self.enrolled_meta[person_id]['photo_count']} photos.", self.enrolled_meta[person_id]

    def enroll_watchlist(self, root_dir="watchlist_images"):
        """Enrolls reference photos from watchlist directory tree into the database."""
        if not os.path.exists(root_dir):
            return False, f"Directory '{root_dir}' does not exist"

        count = 0
        for item in os.listdir(root_dir):
            item_path = os.path.join(root_dir, item)
            if os.path.isdir(item_path):
                person_name = item.replace("_", " ")
                cat = "Suspect / Threat" if "suspect" in person_name.lower() else "Friend / VIP"
                for root, _, files in os.walk(item_path):
                    for fn in files:
                        if fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                            fpath = os.path.join(root, fn)
                            ok, _, _ = self.enroll_single_photo(person_name, category=cat, filename=fpath)
                            if ok:
                                count += 1
            elif item.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                person_name = os.path.splitext(item)[0].replace("_", " ")
                cat = "Suspect / Threat" if "suspect" in person_name.lower() else "Friend / VIP"
                ok, _, _ = self.enroll_single_photo(person_name, category=cat, filename=item_path)
                if ok:
                    count += 1
        return True, f"Enrolled {count} reference photo(s)"

    def update_person(self, person_id, name=None, category=None, notes=None):
        """Updates metadata (name, category, notes) of an existing enrolled person."""
        if person_id not in self.enrolled_meta:
            return False, "Person not found", None

        meta = self.enrolled_meta[person_id]
        if name and name.strip():
            meta["name"] = name.strip()
        if category:
            meta["category"] = category
        if notes is not None:
            meta["notes"] = notes.strip()

        self.save_database()
        clean_meta = {k: v for k, v in meta.items() if k != "raw_embeddings"}
        return True, f"Updated details for '{meta['name']}'", clean_meta

    def delete_person(self, person_id):
        """Removes an enrolled identity from the database."""
        if person_id in self.enrolled_faces:
            del self.enrolled_faces[person_id]
        if person_id in self.enrolled_meta:
            name = self.enrolled_meta[person_id]["name"]
            thumb_file = f"static/thumbnails/{person_id}.jpg"
            if os.path.exists(thumb_file):
                try:
                    os.remove(thumb_file)
                except Exception:
                    pass
            del self.enrolled_meta[person_id]
            self.save_database()
            return True, f"Deleted person '{name}'"
        return False, "Person ID not found"

    def match_embedding(self, embedding, threshold=0.40, strong_threshold=0.55):
        """
        Matches query embedding against enrolled database with tiered confidence states.
        Returns: (match_name: str, score: float, is_match: bool, match_tier: str, category: str, person_id: str)
        """
        if not self.enrolled_faces:
            return "Unknown", 0.0, False, "UNMATCHED", "Unknown", None

        query_norm = embedding / np.linalg.norm(embedding)
        best_score = -1.0
        best_pid = None

        for pid, enrolled_emb in self.enrolled_faces.items():
            if enrolled_emb.ndim == 1:
                sim = float(np.dot(query_norm, enrolled_emb))
            else:
                sims = np.dot(enrolled_emb, query_norm)
                sim = float(np.max(sims))

            if sim > best_score:
                best_score = sim
                best_pid = pid

        if best_pid and best_score >= threshold:
            meta = self.enrolled_meta.get(best_pid, {})
            name = meta.get("name", "Unknown")
            cat = meta.get("category", "Target")
            tier = "HIGH" if best_score >= strong_threshold else "POSSIBLE"
            return name, max(0.0, best_score), True, tier, cat, best_pid

        return "Unknown", max(0.0, best_score), False, "UNMATCHED", "Unknown", None

    def process_frame(self, frame, threshold=0.38, person_bboxes=None):
        """
        Detects all faces in frame simultaneously and matches each independently.
        Supports dual-stage multi-person detection:
        1. High-resolution full-frame RetinaFace pass (detects 10+ faces across the scene).
        2. YOLO crop-assisted fallback pass: checks upper-body regions of any detected persons
           whose faces were missed in the full-frame pass, ensuring 5 to 6 or more people are recognized.
        """
        if self.app is None:
            ok, _ = self.initialize_model()
            if not ok or self.app is None:
                return []

        if frame is None:
            return []

        try:
            faces = self.app.get(frame, max_num=0)
        except Exception:
            faces = []

        results = []
        covered_centers = []

        for face in faces:
            bbox = face.bbox.astype(int).tolist()
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            covered_centers.append((cx, cy))

            emb = face.normed_embedding if hasattr(face, 'normed_embedding') else face.embedding
            name, score, is_match, tier, cat, pid = self.match_embedding(emb, threshold=threshold)

            # Liveness check based on 3D perspective dynamics
            is_live, liveness_score, liveness_tag = self.liveness.update(bbox, face.kps)
            if is_match and not is_live:
                tier = "SPOOF_SUSPECTED"

            results.append({
                "bbox": bbox,
                "name": name if is_match else "Unidentified Person",
                "confidence": score,
                "is_match": is_match,
                "match_tier": tier,
                "is_live": is_live,
                "liveness_tag": liveness_tag,
                "category": cat,
                "person_id": pid,
                "kps": face.kps.tolist() if hasattr(face, 'kps') and face.kps is not None else None
            })

        # Dual-stage fallback: check any YOLO detected person whose face wasn't caught in the full-frame pass
        if person_bboxes:
            h_frame, w_frame = frame.shape[:2]
            for p_bbox in person_bboxes:
                px1, py1, px2, py2 = p_bbox
                pw = px2 - px1
                ph = py2 - py1
                if pw < 20 or ph < 20:
                    continue

                # Check if this person already has an associated face
                already_has_face = False
                for (cx, cy) in covered_centers:
                    if px1 - 35 <= cx <= px2 + 35 and py1 - 25 <= cy <= py1 + ph * 0.70:
                        already_has_face = True
                        break

                if not already_has_face:
                    # Crop upper 50% of person bounding box (head/torso region)
                    head_y2 = min(h_frame, py1 + int(ph * 0.55))
                    crop = frame[max(0, py1):head_y2, max(0, px1):min(w_frame, px2)]
                    if crop.size > 0 and crop.shape[0] >= 24 and crop.shape[1] >= 24:
                        try:
                            crop_faces = self.app.get(crop, max_num=1)
                            for c_face in crop_faces:
                                c_bbox = c_face.bbox.astype(int).tolist()
                                global_bbox = [
                                    c_bbox[0] + max(0, px1),
                                    c_bbox[1] + max(0, py1),
                                    c_bbox[2] + max(0, px1),
                                    c_bbox[3] + max(0, py1)
                                ]
                                emb = c_face.normed_embedding if hasattr(c_face, 'normed_embedding') else c_face.embedding
                                name, score, is_match, tier, cat, pid = self.match_embedding(emb, threshold=threshold)
                                is_live, liveness_score, liveness_tag = self.liveness.update(global_bbox, c_face.kps)
                                if is_match and not is_live:
                                    tier = "SPOOF_SUSPECTED"

                                global_kps = None
                                if hasattr(c_face, 'kps') and c_face.kps is not None:
                                    global_kps = [[pt[0] + max(0, px1), pt[1] + max(0, py1)] for pt in c_face.kps]

                                results.append({
                                    "bbox": global_bbox,
                                    "name": name if is_match else "Unidentified Person",
                                    "confidence": score,
                                    "is_match": is_match,
                                    "match_tier": tier,
                                    "is_live": is_live,
                                    "liveness_tag": liveness_tag,
                                    "category": cat,
                                    "person_id": pid,
                                    "kps": global_kps
                                })
                                covered_centers.append(((global_bbox[0] + global_bbox[2]) / 2.0, (global_bbox[1] + global_bbox[3]) / 2.0))
                        except Exception:
                            pass

        return results

    def draw_faces(self, frame, face_results):
        """
        Renders military-grade tactical HUD target lock reticles with tiered confidence visuals.
        """
        if frame is None or not face_results:
            return frame

        annotated = frame.copy()
        for res in face_results:
            x1, y1, x2, y2 = res["bbox"]
            is_match = res["is_match"]
            tier = res.get("match_tier", "UNMATCHED")
            name = res["name"]
            conf = res["confidence"]
            cat = res.get("category", "")
            kps = res.get("kps")

            if is_match and tier == "HIGH":
                # High Confidence Verified Match
                if "suspect" in cat.lower() or "threat" in cat.lower():
                    box_color = (45, 45, 225)      # Clean muted red
                    header_tag = f"THREAT: {name.upper()} ({conf*100:.1f}%)"
                elif "auth" in cat.lower():
                    box_color = (70, 180, 60)      # Clean muted green
                    header_tag = f"AUTH: {name.upper()} ({conf*100:.1f}%)"
                else:
                    box_color = (40, 165, 235)     # Clean muted amber/gold
                    header_tag = f"VIP: {name.upper()} ({conf*100:.1f}%)"

                # Solid Reticle + White Corner Brackets
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
                corner_len = min(18, (x2 - x1) // 3, (y2 - y1) // 3)
                if corner_len > 4:
                    cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), (230, 230, 230), 2)
                    cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), (230, 230, 230), 2)
                    cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), (230, 230, 230), 2)
                    cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), (230, 230, 230), 2)
                    cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), (230, 230, 230), 2)
                    cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), (230, 230, 230), 2)
                    cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), (230, 230, 230), 2)
                    cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), (230, 230, 230), 2)

                # Badge Tag
                (w, h), _ = cv2.getTextSize(header_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
                tag_y1 = max(0, y1 - h - 8)
                cv2.rectangle(annotated, (x1, tag_y1), (x1 + w + 10, y1), box_color, -1)
                cv2.putText(annotated, header_tag, (x1 + 5, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

            elif is_match and tier == "SPOOF_SUSPECTED":
                # Anti-Spoofing Gate Triggered
                box_color = (30, 110, 240)  # Amber-orange warning
                header_tag = f"SPOOF ALERT: {name.upper()} (LIVENESS FAILED)"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
                (w, h), _ = cv2.getTextSize(header_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                tag_y1 = max(0, y1 - h - 8)
                cv2.rectangle(annotated, (x1, tag_y1), (x1 + w + 8, y1), (15, 18, 25), -1)
                cv2.rectangle(annotated, (x1, tag_y1), (x1 + w + 8, y1), box_color, 1)
                cv2.putText(annotated, header_tag, (x1 + 4, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, box_color, 1, cv2.LINE_AA)

            elif is_match and tier == "POSSIBLE":
                # Moderate / Possible Match
                box_color = (35, 140, 220) # Muted amber
                header_tag = f"POSSIBLE: {name.upper()} ({conf*100:.1f}%)"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 1)
                (w, h), _ = cv2.getTextSize(header_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                tag_y1 = max(0, y1 - h - 6)
                cv2.rectangle(annotated, (x1, tag_y1), (x1 + w + 8, y1), (17, 20, 25), -1)
                cv2.rectangle(annotated, (x1, tag_y1), (x1 + w + 8, y1), box_color, 1)
                cv2.putText(annotated, header_tag, (x1 + 4, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, box_color, 1, cv2.LINE_AA)

            else:
                # Unidentified Person - Subtle and quiet
                box_color = (100, 110, 120)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 1)
                tag = f"UNIDENTIFIED ({conf*100:.0f}%)" if conf > 0.25 else "UNIDENTIFIED"
                (w, h), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
                tag_y1 = max(0, y1 - h - 6)
                cv2.rectangle(annotated, (x1, tag_y1), (x1 + w + 6, y1), (13, 16, 20), -1)
                cv2.putText(annotated, tag, (x1 + 3, y1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 190, 200), 1, cv2.LINE_AA)

            # Facial landmarks - subtle dots
            if kps and is_match:
                for pt in kps:
                    cv2.circle(annotated, (int(pt[0]), int(pt[1])), 1, (200, 200, 200), -1)

        return annotated

    def save_database(self):
        """Saves enrolled faces and metadata to SQLite and disk."""
        try:
            # Save to defense-grade SQLite database
            for pid, meta in self.enrolled_meta.items():
                if pid in self.enrolled_faces:
                    database.save_person_sqlite(
                        person_id=pid,
                        name=meta.get("name", "Unknown"),
                        category=meta.get("category", "Unknown"),
                        notes=meta.get("notes", ""),
                        embedding=self.enrolled_faces[pid],
                        photo_count=meta.get("photo_count", 1),
                        thumb_url=meta.get("thumb_url", f"/static/thumbnails/{pid}.jpg"),
                        created_at=meta.get("created_at", "")
                    )

            # Also maintain JSON for fast non-blocking reads
            json_meta = {}
            for pid, meta in self.enrolled_meta.items():
                copy_m = meta.copy()
                copy_m.pop("raw_embeddings", None)
                json_meta[pid] = copy_m
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(json_meta, f, indent=2)

            with open(self.db_path, "wb") as f:
                pickle.dump({"enrolled_faces": self.enrolled_faces}, f)

            print(f"[INFO] Watchlist database synced to SQLite ({len(self.enrolled_faces)} targets)")
        except Exception as e:
            print(f"[ERROR] Failed to save database: {e}")

    def load_database(self):
        """Loads database from SQLite (falls back to JSON/pickle)."""
        try:
            database.migrate_existing_data()
            sql_faces, sql_meta = database.load_watchlist_sqlite()
            if len(sql_faces) > 0:
                self.enrolled_faces = sql_faces
                self.enrolled_meta = sql_meta
                for pid, emb_mat in self.enrolled_faces.items():
                    if pid in self.enrolled_meta:
                        if emb_mat.ndim == 1:
                            self.enrolled_meta[pid]["raw_embeddings"] = [emb_mat]
                        else:
                            self.enrolled_meta[pid]["raw_embeddings"] = [emb_mat[i] for i in range(len(emb_mat))]
                        self.enrolled_meta[pid]["photo_count"] = len(self.enrolled_meta[pid]["raw_embeddings"])
                print(f"[INFO] Loaded {len(self.enrolled_faces)} identities from SQLite.")
                return
        except Exception as sql_e:
            print(f"[WARN] SQLite load warning: {sql_e}")

        # Fallback to file storage
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "rb") as f:
                    data = pickle.load(f)
                    self.enrolled_faces = data.get("enrolled_faces", {})
            except Exception as e:
                print(f"[WARN] Error loading {self.db_path}: {e}")

        if os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self.enrolled_meta = json.load(f)
            except Exception as e:
                print(f"[WARN] Error loading {self.meta_path}: {e}")
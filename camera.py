# camera.py
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import threading
import logging
import time
from math import hypot

cv2.setNumThreads(1)  # reduce threading issues on some platforms
logging.basicConfig(level=logging.INFO)


def eye_aspect_ratio(eye):
    """Compute simple EAR from 6 (x,y) points."""
    try:
        A = hypot(eye[1][0] - eye[5][0], eye[1][1] - eye[5][1])
        B = hypot(eye[2][0] - eye[4][0], eye[2][1] - eye[4][1])
        C = hypot(eye[0][0] - eye[3][0], eye[0][1] - eye[3][1])
        if C == 0:
            return 0.0
        return (A + B) / (2.0 * C)
    except Exception:
        return 0.0


class CameraStream:
    """
    Single camera thread that:
     - reads frames continuously into self.latest_frame
     - runs mediapipe face/mesh/pose
     - runs emotion inference per face
     - emits per-frame 'emotion_update' and windowed 'final_review' every window_seconds
     - optionally saves frames/reviews to Mongo (if collections provided)
    """

    def __init__(
        self,
        model,
        socketio,
        camera_index=0,
        max_faces=3,
        window_seconds=10.0,
        mongo_frames_collection=None,
        mongo_reviews_collection=None,
    ):
        self.model = model
        self.socketio = socketio
        self.camera_index = camera_index
        self.max_faces = max_faces
        self.window_seconds = float(window_seconds)
        self.mongo_frames = mongo_frames_collection
        self.mongo_reviews = mongo_reviews_collection

        # runtime state
        self.running = False
        self.thread = None
        self.latest_frame = None            # BGR frame for MJPEG
        self.latest_frame_lock = threading.Lock()
        self.active_uid = None              # optionally set by /set_uid
        self.is_saved_model = hasattr(self.model, "signatures")
        self.infer = None
        if self.is_saved_model:
            # choose serving_default if present
            self.infer = self.model.signatures.get("serving_default", None)
            logging.info("📦 Using SavedModel signatures for inference.")
        else:
            logging.info("🧠 Using Keras .predict for inference.")

        # mediapipe modules
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

        self.mp_mesh = mp.solutions.face_mesh
        self.mesh = self.mp_mesh.FaceMesh(
            max_num_faces=self.max_faces, refine_landmarks=True, min_detection_confidence=0.4, min_tracking_confidence=0.4
        )

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.4, min_tracking_confidence=0.4)

        # analytics window
        self.analytics_lock = threading.Lock()
        self.reset_window_metrics()
        self.window_start = time.time()

        # open camera
        self.cap = cv2.VideoCapture(self.camera_index)
        logging.info("CameraStream initialized (camera_index=%s, max_faces=%s)", camera_index, max_faces)

    # ---------------- analytics ----------------
    def reset_window_metrics(self):
        with threading.Lock():
            self.emotion_counts = {}
            self.confidences = []
            self.eye_open_frames = 0
            self.total_frames = 0
            self.blink_count = 0
            self.last_eye_state_per_face = {}
            self.last_blink_times = {}
            self.posture_scores = []

    # ---------------- lifecycle ----------------
    def start(self):
        if not self.cap.isOpened():
            logging.error("❌ Camera not opened (check permissions and device index).")
            return False
        if self.running:
            logging.info("CameraStream already running.")
            return True
        self.running = True
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()
        logging.info("🎥 CameraStream started")
        return True

    def stop(self):
        self.running = False
        time.sleep(0.05)
        if self.cap and self.cap.isOpened():
            try:
                self.cap.release()
            except Exception:
                pass
        logging.info("🛑 CameraStream stopped")

    # ---------------- utility ----------------
    def _preprocess_face(self, frame_bgr, x, y, w, h, size=(160, 160)):
        H, W = frame_bgr.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(W, x + w)
        y2 = min(H, y + h)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        try:
            crop = cv2.resize(crop, size)
        except Exception:
            return None
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = crop.astype("float32") / 255.0
        return np.expand_dims(crop, axis=0)

    def _predict_emotion(self, face_arr):
        try:
            if self.is_saved_model and self.infer is not None:
                tf_input = tf.convert_to_tensor(face_arr, dtype=tf.float32)
                out = self.infer(tf_input)
                preds = list(out.values())[0].numpy()
            else:
                preds = self.model.predict(face_arr, verbose=0)
            idx = int(np.argmax(preds, axis=-1)[0])
            conf = float(np.max(preds, axis=-1)[0])
            labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
            label = labels[idx] if idx < len(labels) else "unknown"
            return label, round(conf, 3)
        except Exception as e:
            logging.debug("predict_emotion error: %s", e)
            return "unknown", 0.0

    # ---------------- main reader loop ----------------
    def _reader_loop(self):
        """
        Single thread: read camera, process, update latest_frame, emit websockets, build windowed review.
        """
        try:
            while self.running:
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    time.sleep(0.02)
                    continue

                # store latest frame (for MJPEG served to browser)
                with self.latest_frame_lock:
                    self.latest_frame = frame.copy()

                H, W = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # mediapipe detection (face boxes)
                try:
                    face_results = self.detector.process(rgb)
                except Exception as e:
                    logging.debug("face detector error: %s", e)
                    face_results = None

                # mesh & pose (guarded — if face/pose libs fail, continue)
                try:
                    mesh_results = self.mesh.process(rgb)
                except Exception as e:
                    logging.debug("mesh error: %s", e)
                    mesh_results = None

                try:
                    pose_results = self.pose.process(rgb)
                except Exception as e:
                    logging.debug("pose error: %s", e)
                    pose_results = None

                live_emotions = []

                # update total frame count
                with self.analytics_lock:
                    self.total_frames += 1

                # handle detections
                if face_results and face_results.detections:
                    for face_idx, det in enumerate(face_results.detections[: self.max_faces]):
                        bbox = det.location_data.relative_bounding_box
                        x = int(max(0, bbox.xmin * W))
                        y = int(max(0, bbox.ymin * H))
                        width = int(bbox.width * W)
                        height = int(bbox.height * H)

                        face_arr = self._preprocess_face(frame, x, y, width, height)
                        if face_arr is None:
                            continue

                        emotion_label, emotion_conf = self._predict_emotion(face_arr)

                        # eye landmarks (from mesh) -> EAR
                        ear_val = 0.0
                        eye_state = "unknown"
                        if mesh_results and mesh_results.multi_face_landmarks and face_idx < len(mesh_results.multi_face_landmarks):
                            try:
                                lm = mesh_results.multi_face_landmarks[face_idx].landmark
                                left_idx = [33, 160, 158, 133, 153, 144]
                                right_idx = [362, 385, 387, 263, 373, 380]
                                left_eye = [(int(lm[i].x * W), int(lm[i].y * H)) for i in left_idx]
                                right_eye = [(int(lm[i].x * W), int(lm[i].y * H)) for i in right_idx]
                                left_ear = eye_aspect_ratio(left_eye)
                                right_ear = eye_aspect_ratio(right_eye)
                                ear_val = float((left_ear + right_ear) / 2.0)
                                ear_val = round(max(0.0, min(1.0, ear_val)), 3)
                                eye_state = "open" if ear_val > 0.22 else "closed"

                                # detect blinks & open frames for analytics
                                with self.analytics_lock:
                                    prev = self.last_eye_state_per_face.get(face_idx, "open")
                                    now_t = time.time()
                                    if prev == "open" and eye_state == "closed":
                                        last = self.last_blink_times.get(face_idx, 0)
                                        if now_t - last > 0.25:
                                            self.blink_count += 1
                                            self.last_blink_times[face_idx] = now_t
                                    self.last_eye_state_per_face[face_idx] = eye_state
                                    if eye_state == "open":
                                        self.eye_open_frames += 1
                            except Exception:
                                pass

                        # posture scoring (simple shoulders Y-diff)
                        posture_score = 0.5
                        if pose_results and pose_results.pose_landmarks:
                            try:
                                lm = pose_results.pose_landmarks.landmark
                                left_sh = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                                right_sh = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                                left_y = left_sh.y * H
                                right_y = right_sh.y * H
                                diff = abs(left_y - right_y)
                                if diff < 25:
                                    posture_score = 1.0
                                elif diff < 60:
                                    posture_score = 0.7
                                else:
                                    posture_score = 0.35
                                with self.analytics_lock:
                                    self.posture_scores.append(posture_score)
                            except Exception:
                                pass

                        # update analytics aggregates
                        with self.analytics_lock:
                            self.emotion_counts[emotion_label] = self.emotion_counts.get(emotion_label, 0) + 1
                            self.confidences.append(emotion_conf)

                        item = {
                            "emotion": emotion_label,
                            "confidence": round(emotion_conf, 3),
                            "ear": round(ear_val, 3),
                            "eye_state": eye_state,
                            "posture": round(posture_score, 3),
                            "box": [x, y, width, height],
                        }

                        live_emotions.append(item)

                        # optionally save per-frame to mongo (non-blocking best-effort)
                        if self.active_uid and self.mongo_frames:
                            try:
                                doc = {"uid": self.active_uid, "timestamp": int(time.time()), **item}
                                self.mongo_frames.insert_one(doc)
                            except Exception as e:
                                logging.debug("mongo save frame error: %s", e)

                        # draw box & label onto latest_frame copy (so MJPEG shows overlays)
                        try:
                            with self.latest_frame_lock:
                                if self.latest_frame is not None:
                                    cv2.rectangle(self.latest_frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                                    label = f"{emotion_label} {int(emotion_conf*100)}%"
                                    cv2.putText(self.latest_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        except Exception:
                            pass

                # emit per-frame emotion_update (non-blocking)
                if live_emotions:
                    try:
                        # use socketio.emit from background thread; it is thread-safe in threading mode
                        self.socketio.emit("emotion_update", {"emotions": live_emotions})
                    except Exception as e:
                        logging.debug("socket emit error: %s", e)

                # check window elapsed and emit final_review (without stopping camera)
                now_t = time.time()
                elapsed = now_t - self.window_start
                if elapsed >= self.window_seconds:
                    final = self._generate_final_review()
                    try:
                        self.socketio.emit("final_review", final)
                        logging.info("📯 Emitted final_review: %s", final)
                    except Exception as e:
                        logging.debug("socket final_review emit error: %s", e)

                    # save to mongo reviews (best-effort)
                    if self.active_uid and self.mongo_reviews:
                        try:
                            doc = {"uid": self.active_uid, "timestamp": int(time.time()), **final}
                            self.mongo_reviews.insert_one(doc)
                        except Exception as e:
                            logging.debug("mongo save review error: %s", e)

                    # reset window
                    with self.analytics_lock:
                        self.window_start = time.time()
                        self.reset_window_metrics()

                # small sleep to be gentle on CPU (tune if needed)
                time.sleep(0.04)

        except Exception as ex:
            logging.exception("CameraStream reader loop crashed: %s", ex)
            # try to keep thread alive briefly, then stop cleanly
            self.running = False

    # ---------------- final review builder ----------------
    def _generate_final_review(self):
        with self.analytics_lock:
            total = max(1, self.total_frames)
            avg_conf = float(np.mean(self.confidences)) if self.confidences else 0.0
            dominant = max(self.emotion_counts, key=self.emotion_counts.get) if self.emotion_counts else "none"
            eye_open_pct = (self.eye_open_frames / total) * 100.0
            blink_count = int(self.blink_count)
            avg_posture = float(np.mean(self.posture_scores)) if self.posture_scores else 0.5

        engagement_raw = (eye_open_pct * 0.4) + (avg_conf * 100.0 * 0.3) + (avg_posture * 100.0 * 0.3)
        engagement_score = round(max(0.0, min(10.0, engagement_raw / 10.0)), 2)

        return {
            "dominant_emotion": dominant,
            "emotion_counts": self.emotion_counts,
            "avg_confidence": round(avg_conf, 3),
            "eye_open_percent": round(eye_open_pct, 2),
            "blink_count": blink_count,
            "avg_posture_score": round(avg_posture, 3),
            "engagement_score": engagement_score,
            "window_seconds": int(self.window_seconds),
            "timestamp": int(time.time()),
        }


# MJPEG generator (reads latest_frame only)
def generate_video_stream(camera_stream: CameraStream):
    """
    Yield MJPEG frames using camera_stream.latest_frame.
    This avoids reading camera twice.
    """
    while True:
        if not camera_stream.running:
            time.sleep(0.05)
            continue

        with camera_stream.latest_frame_lock:
            frame = None if camera_stream.latest_frame is None else camera_stream.latest_frame.copy()

        if frame is None:
            # no frame yet
            time.sleep(0.02)
            continue

        try:
            ret, buf = cv2.imencode(".jpg", frame)
            if not ret:
                time.sleep(0.02)
                continue
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        except Exception as e:
            logging.debug("mjpeg encode error: %s", e)
            time.sleep(0.02)
            continue

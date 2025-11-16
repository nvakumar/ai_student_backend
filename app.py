# # app.py
# import os
# import logging
# import time
# from flask import Flask, jsonify, Response, request
# from flask_socketio import SocketIO
# import tensorflow as tf
# from dotenv import load_dotenv
# from pymongo import MongoClient

# # application camera module
# from camera import CameraStream, generate_video_stream

# load_dotenv(override=True)

# logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
# logging.info("Starting backend...")
# logging.info("TensorFlow version: %s", tf.__version__)

# app = Flask(__name__)

# socketio = SocketIO(
#     app,
#     cors_allowed_origins="*",
#     async_mode="threading",
#     logger=False,
#     engineio_logger=False,
#     ping_timeout=60,
#     ping_interval=25,
# )

# # --------- optional Mongo ----------
# MONGO_URI = os.getenv("MONGO_URI")
# mongo_client = None
# mongo_db = None
# frames_col = None
# reviews_col = None
# students_col = None

# if MONGO_URI:
#     try:
#         mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
#         mongo_db = mongo_client.get_database(os.getenv("MONGO_DB_NAME", "ai_student"))
#         frames_col = mongo_db.get_collection(os.getenv("MONGO_FRAMES_COLLECTION", "frames"))
#         reviews_col = mongo_db.get_collection(os.getenv("MONGO_REVIEW_COLLECTION", "reviews"))
#         students_col = mongo_db.get_collection(os.getenv("MONGO_STUDENTS_COLLECTION", "students"))
#         # quick ping
#         mongo_client.server_info()
#         logging.info("MongoDB connected")
#     except Exception as e:
#         logging.exception("MongoDB connect failed: %s", e)
#         mongo_client = None

# # --------- load model ----------
# MODEL_PATH = os.getenv("MODEL_PATH", "models/saved_model")
# emotion_model = None
# if MODEL_PATH and os.path.exists(MODEL_PATH):
#     try:
#         emotion_model = tf.keras.models.load_model(MODEL_PATH)
#         logging.info("Emotion model loaded from %s", MODEL_PATH)
#     except Exception as e:
#         logging.exception("Failed to load model: %s", e)
# else:
#     logging.warning("MODEL_PATH not set or missing. Emotion inference will not run.")

# # --------- camera stream ----------
# camera_stream = None
# if emotion_model is not None:
#     camera_stream = CameraStream(
#         model=emotion_model,
#         socketio=socketio,
#         camera_index=int(os.getenv("CAM_INDEX", 0)),
#         max_faces=int(os.getenv("MAX_FACES", 3)),
#         window_seconds=float(os.getenv("WINDOW_SECONDS", 10.0)),
#         mongo_frames_collection=frames_col,
#         mongo_reviews_collection=reviews_col,
#     )
#     started = camera_stream.start()
#     if not started:
#         logging.warning("Camera did not start. Check camera permissions.")
# else:
#     logging.warning("Camera not started because emotion model missing.")

# # ---------- routes ----------
# def require_camera_running():
#     if camera_stream is None:
#         return False, jsonify({"error": "camera not configured"}), 500
#     if not getattr(camera_stream, "running", False):
#         return False, jsonify({"error": "camera not running"}), 500
#     return True, None, None

# @app.route("/health")
# def health():
#     return jsonify({
#         "server": "ok",
#         "model_loaded": emotion_model is not None,
#         "camera_running": camera_stream.running if camera_stream else False,
#         "mongo_connected": mongo_client is not None
#     })

# @app.route("/video_feed")
# def video_feed():
#     ok, resp, code = require_camera_running()
#     if not ok:
#         return resp, code
#     return Response(generate_video_stream(camera_stream), mimetype="multipart/x-mixed-replace; boundary=frame")

# @app.route("/set_uid", methods=["POST"])
# def set_uid():
#     """Set active UID for saving frames/reviews."""
#     if not request.is_json:
#         return jsonify({"error": "expected json body"}), 400
#     uid = request.json.get("uid")
#     if not uid:
#         return jsonify({"error": "missing uid"}), 400
#     if camera_stream is None:
#         return jsonify({"error": "camera not configured"}), 500
#     camera_stream.active_uid = uid
#     logging.info("active_uid set -> %s", uid)
#     return jsonify({"ok": True, "uid": uid})

# @app.route("/restart_analysis", methods=["POST"])
# def restart_analysis():
#     ok, resp, code = require_camera_running()
#     if not ok:
#         return resp, code
#     camera_stream.window_start = time.time()
#     camera_stream.reset_window_metrics()
#     logging.info("Analytics window restarted by API")
#     return jsonify({"ok": True})

# @app.route("/stop_camera", methods=["POST"])
# def stop_camera():
#     if camera_stream:
#         camera_stream.stop()
#         return jsonify({"ok": True})
#     return jsonify({"error": "camera not configured"}), 500

# @app.route("/start_camera", methods=["POST"])
# def start_camera():
#     if camera_stream:
#         camera_stream.start()
#         return jsonify({"ok": True})
#     return jsonify({"error": "camera not configured"}), 500

# @app.route("/last_review/<uid>", methods=["GET"])
# def last_review(uid):
#     if reviews_col is None and mongo_client is None:
#         return jsonify({"error": "mongo not configured"}), 500
#     try:
#         rec = reviews_col.find_one({"uid": uid}, sort=[("timestamp", -1)])
#         if not rec:
#             return jsonify({"message": "no review found"}), 404
#         rec["_id"] = str(rec["_id"])
#         return jsonify(rec)
#     except Exception as e:
#         logging.exception("db read error: %s", e)
#         return jsonify({"error": "db error", "details": str(e)}), 500

# # simple test emit
# @app.route("/test_emit", methods=["GET"])
# def test_emit():
#     try:
#         socketio.emit("emotion_update", {"emotions": [{"emotion": "neutral", "confidence": 0.9}]})
#         return jsonify({"emitted": True})
#     except Exception as e:
#         logging.exception("emit failed: %s", e)
#         return jsonify({"error": str(e)}), 500

# # -------- run ----------
# if __name__ == "__main__":
#     logging.info("Starting server on port %s", os.getenv("PORT", "5001"))
#     try:
#         socketio.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 5001)))
#     except KeyboardInterrupt:
#         logging.info("Shutting down...")
#     finally:
#         if camera_stream:
#             camera_stream.stop()
#         if mongo_client:
#             mongo_client.close()
#         logging.info("Clean exit")

import os
import logging
import base64
import io
import numpy as np
from datetime import datetime

from flask import Flask, jsonify, request
from flask_socketio import SocketIO
from PIL import Image
import tensorflow as tf
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv(override=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")

app = Flask(__name__)

# ---------------- Socket.IO (Render-compatible) ----------------
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="eventlet",   # ✅ required for Render
    ping_timeout=60,
    ping_interval=25,
)

# ---------------- MongoDB (optional) ----------------
MONGO_URI = os.getenv("MONGO_URI")
mongo_client = None
reviews_col = None

if MONGO_URI:
    try:
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        mongo_db = mongo_client.get_database(os.getenv("MONGO_DB_NAME", "ai_student"))
        reviews_col = mongo_db.get_collection(os.getenv("MONGO_REVIEW_COLLECTION", "reviews"))
        mongo_client.server_info()
        logging.info("MongoDB connected")
    except Exception as e:
        logging.exception("MongoDB connect failed: %s", e)
        mongo_client = None

# ---------------- Load TF model ----------------
MODEL_PATH = os.getenv("MODEL_PATH", "models/saved_model")
emotion_model = None

if os.path.exists(MODEL_PATH):
    try:
        emotion_model = tf.keras.models.load_model(MODEL_PATH)
        logging.info("Emotion model loaded")
    except Exception as e:
        logging.exception("Model load failed: %s", e)
else:
    logging.warning("MODEL_PATH not found. Model will not run.")

# ---------------- Utils ----------------
def decode_base64_image(data_url: str):
    """Decode base64 image sent from frontend."""
    try:
        header, b64data = data_url.split(",", 1)
        img_bytes = base64.b64decode(b64data)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return img
    except Exception:
        return None

def preprocess(img: Image.Image):
    """Resize + normalize image for model."""
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

# ---------------- API Routes ----------------
@app.route("/health")
def health():
    return jsonify({
        "server": "ok",
        "model_loaded": emotion_model is not None,
        "mongo_connected": mongo_client is not None,
    })

@app.route("/submit_frame", methods=["POST"])
def submit_frame():
    """
    Accepts a frame from frontend (base64 image)
    Run inference → save review → emit result
    """
    if not request.json or "image" not in request.json:
        return jsonify({"error": "missing base64 image"}), 400

    img = decode_base64_image(request.json["image"])
    uid = request.json.get("uid", "unknown")

    if img is None:
        return jsonify({"error": "invalid image"}), 400

    if emotion_model is None:
        return jsonify({"error": "model not loaded"}), 500

    # run prediction
    arr = preprocess(img)
    pred = emotion_model.predict(arr, verbose=0)
    confidence = float(np.max(pred))
    emotion_idx = int(np.argmax(pred))
    emotion = f"class_{emotion_idx}"

    result = {
        "uid": uid,
        "emotion": emotion,
        "confidence": confidence,
        "timestamp": datetime.utcnow().isoformat(),
    }

    # save to mongo
    if reviews_col:
        try:
            reviews_col.insert_one(result)
        except Exception as e:
            logging.exception("mongo insert error: %s", e)

    # emit via socket.io
    socketio.emit("emotion_update", result)

    return jsonify({"ok": True, "result": result})

@app.route("/last_review/<uid>")
def last_review(uid):
    if not reviews_col:
        return jsonify({"error": "mongo not configured"}), 500

    rec = reviews_col.find_one({"uid": uid}, sort=[("timestamp", -1)])
    if not rec:
        return jsonify({"message": "no review found"}), 404

    rec["_id"] = str(rec["_id"])
    return jsonify(rec)

# ---------------- Run server ----------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    logging.info(f"Starting server on port {port}")
    socketio.run(app, host="0.0.0.0", port=port)




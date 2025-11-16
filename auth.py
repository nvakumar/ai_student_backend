# backend/auth.py
import os
import logging
from functools import wraps
from flask import Blueprint, request, jsonify, g
from dotenv import load_dotenv

# Firebase Admin dependencies
try:
    import firebase_admin
    from firebase_admin import credentials, auth as firebase_auth
except Exception as e:
    firebase_admin = None
    firebase_auth = None
    logging.warning("firebase_admin not installed or failed to import: %s", e)

# Firestore Database
try:
    from firebase_init import db
except Exception as e:
    db = None
    logging.error("Failed to import firebase_init.db: %s", e)

# For face embeddings
import numpy as np

load_dotenv(override=True)


# =====================================================
# 🔥 Initialize Firebase
# =====================================================
def init_firebase():
    """
    Initialize Firebase Admin SDK using:
    - FIREBASE_SERVICE_ACCOUNT_PATH (JSON file), or
    - FIREBASE_PROJECT_ID, FIREBASE_CLIENT_EMAIL, FIREBASE_PRIVATE_KEY
    """
    if firebase_admin is None:
        logging.error("firebase_admin library is missing!")
        return False

    if firebase_admin._apps:
        logging.info("Firebase already initialized.")
        return True

    service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")
    project_id = os.getenv("FIREBASE_PROJECT_ID")
    client_email = os.getenv("FIREBASE_CLIENT_EMAIL")
    private_key = os.getenv("FIREBASE_PRIVATE_KEY")

    try:
        if service_account_path and os.path.exists(service_account_path):
            cred = credentials.Certificate(service_account_path)
            firebase_admin.initialize_app(cred)
            logging.info("Firebase initialized from JSON service account.")
        elif project_id and client_email and private_key:
            # Replace literal "\n" with actual newlines
            private_key = private_key.replace("\\n", "\n")
            info = {
                "type": "service_account",
                "project_id": project_id,
                "client_email": client_email,
                "private_key": private_key,
            }
            cred = credentials.Certificate(info)
            firebase_admin.initialize_app(cred)
            logging.info("Firebase initialized from environment variables.")
        else:
            logging.error("Firebase credentials missing in .env")
            return False

        return True

    except Exception as e:
        logging.exception("❌ Failed to initialize Firebase: %s", e)
        return False


# =====================================================
# 🔐 Verify ID Token
# =====================================================
def verify_id_token(id_token):
    """
    Verify Firebase ID Token.
    Returns decoded token dict OR None on failure.
    """
    if firebase_auth is None:
        logging.error("firebase_auth module missing.")
        return None

    try:
        return firebase_auth.verify_id_token(id_token)
    except Exception as e:
        logging.warning("Failed to verify ID token: %s", e)
        return None


# =====================================================
# 🔰 Auth Middleware Decorator
# =====================================================
def auth_required(f):
    """
    Protect API route using Firebase ID token.
    Header format:
        Authorization: Bearer <idToken>
    """

    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization")

        if not auth_header:
            return jsonify({"error": "Missing Authorization header"}), 401

        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return jsonify({"error": "Invalid Authorization header"}), 401

        id_token = parts[1]
        decoded = verify_id_token(id_token)

        if not decoded:
            return jsonify({"error": "Invalid or expired token"}), 401

        # Attach user info
        g.user = {
            "uid": decoded.get("uid"),
            "email": decoded.get("email"),
            "name": decoded.get("name")
        }

        return f(*args, **kwargs)

    return decorated


# =====================================================
# 📌 Blueprint for Authentication & Face Registration
# =====================================================
auth_routes = Blueprint("auth_routes", __name__)


# -----------------------------------------------------
# POST /register_face
# Save student's face embedding in Firestore
# -----------------------------------------------------
@auth_routes.route("/register_face", methods=["POST"])
@auth_required
def register_face():
    """
    Store student's face embedding.
    Body:
    {
        "embedding": [float, float, ...]
    }
    """
    try:
        body = request.json
        embedding = body.get("embedding")

        if embedding is None:
            return jsonify({"error": "Missing embedding"}), 400

        uid = g.user["uid"]

        # Convert to python list (Firestore JSON compatible)
        embedding_list = np.array(embedding).tolist()

        db.collection("students").document(uid).set(
            {"face_embedding": embedding_list},
            merge=True
        )

        logging.info(f"Face registered for UID: {uid}")

        return jsonify({"message": "Face registered successfully!"})

    except Exception as e:
        logging.exception("Face registration error: %s", e)
        return jsonify({"error": str(e)}), 500

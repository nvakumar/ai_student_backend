import firebase_admin
from firebase_admin import credentials, firestore, auth
import os
from dotenv import load_dotenv

load_dotenv()

FIREBASE_KEY_PATH = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")

if not FIREBASE_KEY_PATH or not os.path.exists(FIREBASE_KEY_PATH):
    raise Exception(f"❌ Firebase service account JSON not found at: {FIREBASE_KEY_PATH}")

# Load credentials
cred = credentials.Certificate(FIREBASE_KEY_PATH)

# initialize firebase only once
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

# Firestore database instance
db = firestore.client()

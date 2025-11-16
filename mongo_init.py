import os
import logging
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv(override=True)

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")

client = None
db = None

try:
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    logging.info("✅ Connected to MongoDB successfully.")
except Exception as e:
    logging.error(f"❌ MongoDB connection failed: {e}")

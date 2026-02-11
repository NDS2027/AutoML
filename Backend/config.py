"""Configuration settings for the AutoML Advisor backend."""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"
MODELS_DIR = BASE_DIR / "trained_models"

# Create directories if they don't exist
for dir_path in [UPLOADS_DIR, PROCESSED_DIR, SAMPLES_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# MongoDB settings
# For MongoDB Atlas, use: mongodb+srv://username:password@cluster.mongodb.net/
# For local MongoDB, use: mongodb://localhost:27017
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
MONGODB_DB_NAME = "automl_advisor"

# ML settings
CHURN_THRESHOLD_DEFAULT = 60  # days
MIN_RECORDS = 50  # Lowered for testing
MIN_CUSTOMERS = 10  # Lowered for testing
TEST_SIZE = 0.2
CV_FOLDS = 5
RANDOM_STATE = 42

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000

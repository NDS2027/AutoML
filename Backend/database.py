"""MongoDB database connection and operations."""

from pymongo import MongoClient
from pymongo.database import Database
from typing import Optional
import config

# Global database connection
_client: Optional[MongoClient] = None
_db: Optional[Database] = None


def get_database() -> Database:
    """Get MongoDB database connection."""
    global _client, _db
    
    if _db is None:
        _client = MongoClient(config.MONGODB_URL)
        _db = _client[config.MONGODB_DB_NAME]
    
    return _db


def close_database():
    """Close MongoDB connection."""
    global _client, _db
    
    if _client:
        _client.close()
        _client = None
        _db = None


# Collection accessors
def get_sessions_collection():
    """Get sessions collection."""
    return get_database()["sessions"]


def get_predictions_collection():
    """Get predictions collection."""
    return get_database()["predictions"]


def get_models_collection():
    """Get models collection."""
    return get_database()["models"]


def get_insights_collection():
    """Get insights collection."""
    return get_database()["insights"]

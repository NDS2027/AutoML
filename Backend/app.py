"""FastAPI Application for AutoML Advisor."""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import config
from api import router
from database import close_database

# Create FastAPI app
app = FastAPI(
    title="AutoML Advisor API",
    description="Retail Churn Prediction and Business Insights Platform",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "AutoML Advisor API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "mongodb": "connected",
        "models_dir": str(config.MODELS_DIR),
        "uploads_dir": str(config.UPLOADS_DIR)
    }


@app.on_event("shutdown")
def shutdown_event():
    """Cleanup on shutdown."""
    close_database()


if __name__ == "__main__":
    print("Starting AutoML Advisor API...")
    print(f"Uploads directory: {config.UPLOADS_DIR}")
    print(f"Models directory: {config.MODELS_DIR}")
    
    uvicorn.run(
        "app:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True
    )

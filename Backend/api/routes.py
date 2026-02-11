"""API Routes for AutoML Advisor."""

import os
import sys
import uuid
import shutil
from pathlib import Path
from typing import Optional
from datetime import datetime
import pandas as pd
import threading

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from database import (
    get_sessions_collection,
    get_predictions_collection,
    get_models_collection,
    get_insights_collection
)
from models.schemas import (
    UploadResponse, ConfigureRequest, ConfigureResponse,
    TrainRequest, TrainStatusResponse, JobStatus,
    ResultsResponse, PredictionsResponse, 
    SimulateRequest, SimulateResponse
)
from core import (
    DataProfiler, FeatureEngineer, 
    AutoMLEngine, ExplainabilityModule, InsightGenerator
)
from utils.helpers import generate_id, load_dataframe, detect_column_types

router = APIRouter(prefix="/api", tags=["automl"])

# In-memory job tracking (for simplicity - could use Redis)
training_jobs = {}


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload a CSV or Excel file for analysis."""
    
    # Validate file type
    if not file.filename:
        raise HTTPException(400, "No filename provided")
        
    suffix = Path(file.filename).suffix.lower()
    if suffix not in [".csv", ".xlsx", ".xls"]:
        raise HTTPException(400, f"Unsupported file type: {suffix}")
    
    # Generate upload ID
    upload_id = generate_id("upload")
    
    # Save file
    file_path = config.UPLOADS_DIR / f"{upload_id}_{file.filename}"
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(500, f"Failed to save file: {str(e)}")
    
    # Load and analyze
    try:
        df = load_dataframe(file_path)
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(400, f"Failed to parse file: {str(e)}")
    
    # Detect column types
    column_types = detect_column_types(df)
    
    # Get preview
    preview = df.head(5).fillna("").to_dict(orient="records")
    
    # Store upload info in MongoDB
    sessions = get_sessions_collection()
    sessions.insert_one({
        "_id": upload_id,
        "filename": file.filename,
        "file_path": str(file_path),
        "rows": len(df),
        "columns": df.columns.tolist(),
        "uploaded_at": datetime.utcnow(),
        "status": "uploaded"
    })
    
    return UploadResponse(
        upload_id=upload_id,
        filename=file.filename,
        rows=len(df),
        columns=df.columns.tolist(),
        preview=preview,
        detected_types=column_types
    )


@router.post("/configure", response_model=ConfigureResponse)
async def configure_analysis(request: ConfigureRequest):
    """Configure column mappings and churn threshold."""
    
    sessions = get_sessions_collection()
    session = sessions.find_one({"_id": request.upload_id})
    
    if not session:
        raise HTTPException(404, "Upload not found")
    
    # Load data
    df = load_dataframe(Path(session["file_path"]))
    
    # Validate mappings
    mapping = {
        "customer_id": request.column_mapping.customer_id,
        "date": request.column_mapping.date,
        "amount": request.column_mapping.amount,
        "product": request.column_mapping.product
    }
    
    # Create profiler and validate
    profiler = DataProfiler(df, mapping)
    validation = profiler.validate()
    
    # Generate session ID
    session_id = generate_id("session")
    
    # Update session
    sessions.update_one(
        {"_id": request.upload_id},
        {"$set": {
            "session_id": session_id,
            "column_mapping": mapping,
            "churn_threshold_days": request.churn_threshold_days,
            "validation_results": validation,
            "status": "configured"
        }}
    )
    
    return ConfigureResponse(
        session_id=session_id,
        status="configured",
        validation_results=validation
    )


def run_training_pipeline(session_id: str, upload_id: str):
    """Background task to run the ML pipeline."""
    
    sessions = get_sessions_collection()
    models_col = get_models_collection()
    predictions_col = get_predictions_collection()
    insights_col = get_insights_collection()
    
    job_id = f"job_{session_id}"
    
    try:
        # Update status
        training_jobs[job_id] = {
            "status": JobStatus.RUNNING,
            "progress": 0,
            "current_step": "loading_data",
            "steps_completed": []
        }
        
        # Get session info
        session = sessions.find_one({"session_id": session_id})
        if not session:
            raise ValueError("Session not found")
        
        # Load and clean data
        df = load_dataframe(Path(session["file_path"]))
        profiler = DataProfiler(df, session["column_mapping"])
        df_clean = profiler.clean()
        
        training_jobs[job_id]["progress"] = 15
        training_jobs[job_id]["current_step"] = "engineering_features"
        training_jobs[job_id]["steps_completed"].append("data_loaded")
        
        # Feature engineering
        engineer = FeatureEngineer(df_clean)
        features = engineer.create_all_features()
        
        # Get churn labels - compute directly from cleaned data
        churn_threshold = session.get("churn_threshold_days", 60)
        reference_date = df_clean["date"].max()
        customer_last_purchase = df_clean.groupby("customer_id")["date"].max()
        days_since_purchase = (reference_date - customer_last_purchase).dt.days
        churn_labels = (days_since_purchase > churn_threshold).astype(int)
        churn_labels.name = "is_churned"
        
        print(f"DEBUG: Features shape: {features.shape}, index sample: {features.index[:5].tolist()}")
        print(f"DEBUG: Churn labels shape: {churn_labels.shape}, index sample: {churn_labels.index[:5].tolist()}")
        
        # Align features and labels
        common_idx = features.index.intersection(churn_labels.index)
        print(f"DEBUG: Common index count: {len(common_idx)}")
        
        if len(common_idx) == 0:
            # Try string conversion for matching
            features.index = features.index.astype(str)
            churn_labels.index = churn_labels.index.astype(str)
            common_idx = features.index.intersection(churn_labels.index)
            print(f"DEBUG: After string conversion, common index count: {len(common_idx)}")
        
        X = features.loc[common_idx]
        y = churn_labels.loc[common_idx]
        
        print(f"DEBUG: Final X shape: {X.shape}, y shape: {y.shape}")
        
        if len(X) == 0:
            raise ValueError(f"No samples after alignment. Check that customer IDs match between features and labels.")
        
        training_jobs[job_id]["progress"] = 30
        training_jobs[job_id]["current_step"] = "training_models"
        training_jobs[job_id]["steps_completed"].append("features_engineered")
        
        # Train models
        automl = AutoMLEngine()
        
        def progress_callback(step, total, message):
            progress = 30 + int((step / total) * 40)
            training_jobs[job_id]["progress"] = progress
            training_jobs[job_id]["current_step"] = message
        
        results = automl.train_all_models(X, y, progress_callback)
        
        training_jobs[job_id]["progress"] = 75
        training_jobs[job_id]["current_step"] = "generating_predictions"
        training_jobs[job_id]["steps_completed"].append("models_trained")
        
        # Save models
        automl.save_models(config.MODELS_DIR, session_id)
        
        # Generate predictions for all customers
        predictions, probabilities = automl.predict(X)
        
        # Create predictions DataFrame
        pred_df = X.copy()
        pred_df["churn_probability"] = probabilities
        pred_df["is_churned_predicted"] = predictions
        pred_df["customer_id"] = pred_df.index
        
        # Get feature importance
        feature_importance = automl.get_feature_importance()
        
        training_jobs[job_id]["progress"] = 85
        training_jobs[job_id]["current_step"] = "generating_insights"
        training_jobs[job_id]["steps_completed"].append("predictions_generated")
        
        # Generate insights
        insight_gen = InsightGenerator(pred_df, feature_importance)
        summary = insight_gen.get_summary_stats()
        segments = insight_gen.get_segment_analysis()
        drivers = insight_gen.get_top_churn_drivers()
        recommendations = insight_gen.generate_recommendations()
        roi = insight_gen.calculate_roi()
        
        # Store results in MongoDB
        champion_name, _ = automl.get_champion()
        
        # Store model metadata
        for model_name, metrics in results.items():
            models_col.insert_one({
                "session_id": session_id,
                "model_name": model_name,
                "is_champion": metrics.get("is_champion", False),
                **metrics,
                "created_at": datetime.utcnow()
            })
        
        # Store predictions (sample for large datasets)
        pred_records = pred_df.head(10000).to_dict(orient="records")
        if pred_records:
            predictions_col.insert_many([
                {"session_id": session_id, **record}
                for record in pred_records
            ])
        
        # Store insights
        insights_col.insert_one({
            "session_id": session_id,
            "summary": summary,
            "segments": {k: {
                "count": v.count,
                "total_clv": v.total_clv,
                "avg_clv": v.avg_clv,
                "avg_churn_prob": v.avg_churn_prob
            } for k, v in segments.items()},
            "drivers": drivers,
            "recommendations": recommendations,
            "roi": roi,
            "feature_importance": [
                {"feature": k, "importance": v}
                for k, v in feature_importance.items()
            ],
            "champion_model": champion_name,
            "created_at": datetime.utcnow()
        })
        
        # Update session status
        sessions.update_one(
            {"session_id": session_id},
            {"$set": {"status": "completed"}}
        )
        
        training_jobs[job_id]["progress"] = 100
        training_jobs[job_id]["status"] = JobStatus.COMPLETED
        training_jobs[job_id]["current_step"] = "completed"
        training_jobs[job_id]["steps_completed"].append("insights_generated")
        
    except Exception as e:
        import traceback
        print("=" * 50)
        print("TRAINING ERROR:")
        traceback.print_exc()
        print("=" * 50)
        
        training_jobs[job_id]["status"] = JobStatus.FAILED
        training_jobs[job_id]["error"] = str(e)
        
        sessions.update_one(
            {"session_id": session_id},
            {"$set": {"status": "failed", "error": str(e)}}
        )


@router.post("/train")
async def start_training(request: TrainRequest, background_tasks: BackgroundTasks):
    """Start model training in background."""
    
    sessions = get_sessions_collection()
    session = sessions.find_one({"session_id": request.session_id})
    
    if not session:
        raise HTTPException(404, "Session not found")
    
    if session.get("status") != "configured":
        raise HTTPException(400, f"Session not ready for training. Status: {session.get('status')}")
    
    job_id = f"job_{request.session_id}"
    
    # Initialize job
    training_jobs[job_id] = {
        "status": JobStatus.PENDING,
        "progress": 0,
        "current_step": "queued",
        "steps_completed": []
    }
    
    # Start background task
    thread = threading.Thread(
        target=run_training_pipeline,
        args=(request.session_id, session["_id"])
    )
    thread.start()
    
    return {
        "job_id": job_id,
        "status": "started",
        "estimated_time_seconds": 120
    }


@router.get("/train/status/{job_id}", response_model=TrainStatusResponse)
async def get_training_status(job_id: str):
    """Get training job status."""
    
    if job_id not in training_jobs:
        raise HTTPException(404, "Job not found")
    
    job = training_jobs[job_id]
    
    # Estimate remaining time
    progress = job.get("progress", 0)
    remaining = None
    if progress > 0 and progress < 100:
        remaining = int((100 - progress) * 1.5)  # Rough estimate
    
    return TrainStatusResponse(
        job_id=job_id,
        status=job.get("status", JobStatus.PENDING),
        progress=progress,
        current_step=job.get("current_step"),
        steps_completed=job.get("steps_completed", []),
        estimated_remaining_seconds=remaining
    )


@router.get("/results/{session_id}")
async def get_results(session_id: str):
    """Get analysis results."""
    
    insights_col = get_insights_collection()
    models_col = get_models_collection()
    
    insights = insights_col.find_one({"session_id": session_id})
    if not insights:
        raise HTTPException(404, "Results not found")
    
    # Get model performance
    models = list(models_col.find({"session_id": session_id}))
    model_performance = [
        {
            "model_name": m["model_name"],
            "f1_score": m.get("f1_score", 0),
            "precision": m.get("precision", 0),
            "recall": m.get("recall", 0),
            "roc_auc": m.get("roc_auc", 0),
            "is_champion": m.get("is_champion", False)
        }
        for m in models
    ]
    
    return {
        "summary": insights["summary"],
        "model_performance": model_performance,
        "feature_importance": insights["feature_importance"][:10],
        "segments": insights["segments"],
        "drivers": insights["drivers"][:5],
        "recommendations": insights["recommendations"],
        "roi": insights["roi"],
        "champion_model": insights["champion_model"]
    }


@router.get("/predictions/{session_id}")
async def get_predictions(
    session_id: str,
    risk_tier: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """Get customer predictions."""
    
    predictions_col = get_predictions_collection()
    
    query = {"session_id": session_id}
    if risk_tier:
        query["risk_tier"] = risk_tier
    
    # Get total count
    total = predictions_col.count_documents(query)
    
    # Get predictions
    cursor = predictions_col.find(query).sort("churn_probability", -1).skip(offset).limit(limit)
    
    customers = []
    for doc in cursor:
        customers.append({
            "customer_id": str(doc.get("customer_id", "")),
            "churn_probability": doc.get("churn_probability", 0),
            "risk_tier": doc.get("risk_tier", "unknown"),
            "clv": doc.get("estimated_clv", 0),
            "days_since_purchase": doc.get("days_since_last_purchase", 0)
        })
    
    return {
        "total": total,
        "page": offset // limit + 1,
        "customers": customers
    }


@router.post("/simulate")
async def simulate_scenario(request: SimulateRequest):
    """Run what-if scenario simulation."""
    
    insights_col = get_insights_collection()
    predictions_col = get_predictions_collection()
    
    insights = insights_col.find_one({"session_id": request.session_id})
    if not insights:
        raise HTTPException(404, "Session not found")
    
    # Get predictions for ROI calculation
    predictions = list(predictions_col.find({"session_id": request.session_id}))
    
    if not predictions:
        raise HTTPException(404, "No predictions found")
    
    pred_df = pd.DataFrame(predictions)
    feature_importance = {
        item["feature"]: item["importance"]
        for item in insights.get("feature_importance", [])
    }
    
    # Calculate ROI with new parameters
    insight_gen = InsightGenerator(pred_df, feature_importance)
    roi = insight_gen.calculate_roi(
        target_count=request.target_count,
        discount_percent=request.discount_percent,
        success_rate=request.success_rate
    )
    
    return SimulateResponse(
        scenario_id=generate_id("scenario"),
        inputs={
            "target_count": request.target_count,
            "discount_percent": request.discount_percent,
            "success_rate": request.success_rate
        },
        outputs=roi
    )


# ======================
# LLM Chat Endpoints
# ======================
from models.schemas import ChatRequest, ChatResponse, InsightRequest, InsightResponse, LLMStatusResponse
from core.llm_service import get_llm_service, create_insights_prompt, create_qa_prompt


@router.get("/llm/status")
async def get_llm_status():
    """Check if LLM service is available."""
    llm = get_llm_service()
    status = llm.check_availability()
    return LLMStatusResponse(**status)


@router.post("/insights/generate")
async def generate_insights(request: InsightRequest):
    """Generate AI-powered insights for the session."""
    
    # Get session data
    insights_col = get_insights_collection()
    insights = insights_col.find_one({"session_id": request.session_id})
    
    if not insights:
        raise HTTPException(404, "Session not found")
    
    # Check LLM availability
    llm = get_llm_service()
    status = llm.check_availability()
    
    if not status.get("available"):
        return InsightResponse(
            insight="⚠️ AI insights not available. Please install Ollama from https://ollama.com and run: ollama pull llama3.2",
            available=False
        )
    
    # Create prompt
    summary = insights.get("summary", {})
    segments = insights.get("segments", {})
    drivers = insights.get("drivers", [])
    recommendations = insights.get("recommendations", [])
    
    prompt = create_insights_prompt(summary, segments, drivers, recommendations)
    
    # Generate insight
    system_prompt = "You are a business analyst expert in customer retention. Be concise and actionable."
    response = llm.generate(prompt, system_prompt)
    
    return InsightResponse(insight=response, available=True)


@router.post("/chat")
async def chat_with_data(request: ChatRequest):
    """Chat with the AI about the churn analysis."""
    
    # Get session data
    insights_col = get_insights_collection()
    insights = insights_col.find_one({"session_id": request.session_id})
    
    if not insights:
        raise HTTPException(404, "Session not found")
    
    # Check LLM availability
    llm = get_llm_service()
    status = llm.check_availability()
    
    if not status.get("available"):
        return ChatResponse(
            response="⚠️ AI chat is not available. Please start Ollama to enable this feature.",
            suggestions=["Install Ollama from https://ollama.com", "Run: ollama pull llama3.2"]
        )
    
    # Create context-aware prompt
    summary = insights.get("summary", {})
    segments = insights.get("segments", {})
    drivers = insights.get("drivers", [])
    
    prompt = create_qa_prompt(request.message, summary, segments, drivers)
    
    # Generate response
    system_prompt = "You are a helpful business analyst. Answer questions about customer churn data clearly and concisely."
    response = llm.generate(prompt, system_prompt)
    
    # Suggest follow-up questions
    suggestions = [
        "What actions should I take first?",
        "Which customers should I prioritize?",
        "How can I reduce churn rate?",
        "What's causing customers to leave?"
    ]
    
    return ChatResponse(response=response, suggestions=suggestions)

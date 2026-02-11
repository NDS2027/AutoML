"""Pydantic models for API request/response validation."""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from datetime import datetime
from enum import Enum


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RiskTier(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# Request Models
class ColumnMapping(BaseModel):
    customer_id: str
    date: str
    amount: str
    product: Optional[str] = None


class ConfigureRequest(BaseModel):
    upload_id: str
    column_mapping: ColumnMapping
    churn_threshold_days: int = 60


class TrainRequest(BaseModel):
    session_id: str


class SimulateRequest(BaseModel):
    session_id: str
    target_count: int = 100
    discount_percent: float = 20.0
    success_rate: float = 0.35


# Response Models
class UploadResponse(BaseModel):
    upload_id: str
    filename: str
    rows: int
    columns: List[str]
    preview: List[Dict[str, Any]]
    detected_types: Dict[str, str]


class ConfigureResponse(BaseModel):
    session_id: str
    status: str
    validation_results: Dict[str, Any]


class TrainStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: int = 0
    current_step: Optional[str] = None
    steps_completed: List[str] = []
    estimated_remaining_seconds: Optional[int] = None


class ModelPerformance(BaseModel):
    model_name: str
    f1_score: float
    precision: float
    recall: float
    roc_auc: float
    is_champion: bool = False


class SegmentInfo(BaseModel):
    count: int
    revenue: float
    avg_clv: float


class Recommendation(BaseModel):
    action: str
    discount_percent: float
    expected_cost: float
    expected_saves: int
    expected_revenue: float
    roi_percent: float


class ResultsResponse(BaseModel):
    summary: Dict[str, Any]
    model_performance: List[ModelPerformance]
    feature_importance: List[Dict[str, Any]]
    segments: Dict[str, SegmentInfo]
    recommendations: Recommendation


class CustomerPrediction(BaseModel):
    customer_id: str
    churn_probability: float
    risk_tier: RiskTier
    clv: float
    days_since_purchase: int
    top_risk_factors: List[Dict[str, Any]]


class PredictionsResponse(BaseModel):
    total: int
    page: int
    customers: List[CustomerPrediction]


class SimulateResponse(BaseModel):
    scenario_id: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]


# Chat Models
class ChatRequest(BaseModel):
    session_id: str
    message: str
    

class ChatResponse(BaseModel):
    response: str
    suggestions: List[str] = []
    

class InsightRequest(BaseModel):
    session_id: str


class InsightResponse(BaseModel):
    insight: str
    available: bool = True


class LLMStatusResponse(BaseModel):
    available: bool
    provider: str
    model: Optional[str] = None
    error: Optional[str] = None

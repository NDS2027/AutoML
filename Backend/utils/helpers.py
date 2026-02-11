"""Utility helper functions."""

import uuid
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


def generate_id(prefix: str = "") -> str:
    """Generate a unique ID with optional prefix."""
    unique_id = str(uuid.uuid4())[:8]
    return f"{prefix}_{unique_id}" if prefix else unique_id


def load_dataframe(file_path: Path) -> pd.DataFrame:
    """Load a CSV or Excel file into a DataFrame."""
    suffix = file_path.suffix.lower()
    
    if suffix == ".csv":
        # Try different encodings
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                return pd.read_csv(file_path, encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not decode CSV file")
    
    elif suffix in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)
    
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """Detect and return column types."""
    type_map = {}
    
    for col in df.columns:
        dtype = df[col].dtype
        
        if pd.api.types.is_datetime64_any_dtype(dtype):
            type_map[col] = "datetime"
        elif pd.api.types.is_numeric_dtype(dtype):
            if pd.api.types.is_integer_dtype(dtype):
                type_map[col] = "integer"
            else:
                type_map[col] = "float"
        elif pd.api.types.is_bool_dtype(dtype):
            type_map[col] = "boolean"
        else:
            # Check if it might be a date string
            sample = df[col].dropna().head(100)
            try:
                pd.to_datetime(sample)
                type_map[col] = "datetime"
            except:
                type_map[col] = "string"
    
    return type_map


def format_currency(value: float) -> str:
    """Format a number as currency."""
    return f"${value:,.2f}"


def format_percentage(value: float) -> str:
    """Format a decimal as percentage."""
    return f"{value * 100:.1f}%"


def calculate_days_between(date1: datetime, date2: datetime) -> int:
    """Calculate days between two dates."""
    return abs((date2 - date1).days)

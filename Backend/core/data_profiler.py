"""Data Profiler - Validates and profiles uploaded data."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class DataProfiler:
    """Validates, cleans, and profiles customer transaction data."""
    
    def __init__(self, df: pd.DataFrame, column_mapping: Dict[str, str]):
        """
        Initialize the profiler.
        
        Args:
            df: Raw transaction DataFrame
            column_mapping: Maps required fields to actual column names
                           {customer_id, date, amount, product (optional)}
        """
        self.df = df.copy()
        self.mapping = column_mapping
        self.quality_issues: List[str] = []
        self.quality_score: float = 10.0
        
    def validate(self) -> Dict[str, Any]:
        """
        Validate the dataset meets requirements.
        
        Returns:
            Validation results dictionary
        """
        results = {
            "is_valid": True,
            "has_required_columns": True,
            "min_records": True,
            "min_customers": True,
            "quality_score": 10.0,
            "issues": []
        }
        
        # Check required columns exist
        for field in ["customer_id", "date", "amount"]:
            col = self.mapping.get(field)
            if not col or col not in self.df.columns:
                results["has_required_columns"] = False
                results["issues"].append(f"Missing required column: {field}")
                results["is_valid"] = False
                
        if not results["has_required_columns"]:
            return results
            
        # Check minimum records
        if len(self.df) < config.MIN_RECORDS:
            results["min_records"] = False
            results["issues"].append(
                f"Need at least {config.MIN_RECORDS} records, got {len(self.df)}"
            )
            self.quality_score -= 2
            
        # Check minimum unique customers
        customer_col = self.mapping["customer_id"]
        unique_customers = self.df[customer_col].nunique()
        if unique_customers < config.MIN_CUSTOMERS:
            results["min_customers"] = False
            results["issues"].append(
                f"Need at least {config.MIN_CUSTOMERS} unique customers, got {unique_customers}"
            )
            self.quality_score -= 2
            
        # Check for missing values
        for field in ["customer_id", "date", "amount"]:
            col = self.mapping[field]
            missing_pct = self.df[col].isna().mean() * 100
            if missing_pct > 20:
                results["issues"].append(
                    f"High missing values in {col}: {missing_pct:.1f}%"
                )
                self.quality_score -= 1
            elif missing_pct > 5:
                results["issues"].append(
                    f"Some missing values in {col}: {missing_pct:.1f}%"
                )
                self.quality_score -= 0.5
                
        results["quality_score"] = max(0, self.quality_score)
        results["is_valid"] = results["quality_score"] >= 5
        
        return results
    
    def clean(self) -> pd.DataFrame:
        """
        Clean the dataset.
        
        Returns:
            Cleaned DataFrame
        """
        df = self.df.copy()
        
        # Get the actual column names from mapping
        cust_col = self.mapping.get("customer_id")
        date_col = self.mapping.get("date")
        amount_col = self.mapping.get("amount")
        product_col = self.mapping.get("product")
        
        print(f"DEBUG CLEAN: Input rows: {len(df)}")
        print(f"DEBUG CLEAN: Columns: {df.columns.tolist()}")
        print(f"DEBUG CLEAN: Mapping - cust:{cust_col}, date:{date_col}, amount:{amount_col}")
        
        # Create new DataFrame with standardized column names
        new_df = pd.DataFrame()
        
        if cust_col and cust_col in df.columns:
            new_df["customer_id"] = df[cust_col].values
            print(f"DEBUG CLEAN: Customer IDs: {new_df['customer_id'].head().tolist()}")
        else:
            raise ValueError(f"Customer ID column '{cust_col}' not found in data. Available: {df.columns.tolist()}")
            
        if date_col and date_col in df.columns:
            new_df["date"] = pd.to_datetime(df[date_col].values, errors="coerce")
            print(f"DEBUG CLEAN: Date sample: {new_df['date'].head().tolist()}")
            print(f"DEBUG CLEAN: Date NaT count: {new_df['date'].isna().sum()}")
        else:
            raise ValueError(f"Date column '{date_col}' not found in data. Available: {df.columns.tolist()}")
            
        if amount_col and amount_col in df.columns:
            new_df["amount"] = pd.to_numeric(df[amount_col].values, errors="coerce")
            print(f"DEBUG CLEAN: Amount sample: {new_df['amount'].head().tolist()}")
            print(f"DEBUG CLEAN: Amount NaN count: {new_df['amount'].isna().sum()}")
        else:
            raise ValueError(f"Amount column '{amount_col}' not found in data. Available: {df.columns.tolist()}")
            
        if product_col and product_col in df.columns:
            new_df["product"] = df[product_col].values
        
        print(f"DEBUG CLEAN: Before dropna: {len(new_df)} rows")
        
        # Remove rows with missing required values
        new_df = new_df.dropna(subset=["customer_id", "date", "amount"])
        print(f"DEBUG CLEAN: After dropna: {len(new_df)} rows")
        
        # Remove negative amounts
        new_df = new_df[new_df["amount"] >= 0]
        print(f"DEBUG CLEAN: After negative filter: {len(new_df)} rows")
        
        # Remove future dates - SKIP THIS CHECK FOR NOW
        # new_df = new_df[new_df["date"] <= pd.Timestamp.now()]
        
        # Sort by customer and date
        new_df = new_df.sort_values(["customer_id", "date"])
        
        print(f"DEBUG CLEAN: Final: {len(new_df)} rows, {new_df['customer_id'].nunique()} customers")
        
        return new_df
    
    def profile(self) -> Dict[str, Any]:
        """
        Generate comprehensive data profile.
        
        Returns:
            Profile statistics dictionary
        """
        df = self.clean()
        
        profile = {
            "record_count": len(df),
            "customer_count": df["customer_id"].nunique(),
            "date_range": {
                "start": df["date"].min().isoformat() if len(df) > 0 else None,
                "end": df["date"].max().isoformat() if len(df) > 0 else None,
                "span_days": (df["date"].max() - df["date"].min()).days if len(df) > 0 else 0
            },
            "transaction_stats": {
                "total_revenue": float(df["amount"].sum()),
                "avg_order_value": float(df["amount"].mean()),
                "median_order_value": float(df["amount"].median()),
                "min_order": float(df["amount"].min()),
                "max_order": float(df["amount"].max())
            },
            "customer_stats": {
                "avg_purchases_per_customer": len(df) / df["customer_id"].nunique() if df["customer_id"].nunique() > 0 else 0,
                "single_purchase_customers": int((df.groupby("customer_id").size() == 1).sum()),
                "repeat_customers": int((df.groupby("customer_id").size() > 1).sum())
            }
        }
        
        # Add product stats if available
        if "product" in df.columns:
            profile["product_stats"] = {
                "unique_products": df["product"].nunique(),
                "top_products": df["product"].value_counts().head(10).to_dict()
            }
            
        return profile
    
    def detect_churn_threshold(self) -> Dict[str, Any]:
        """
        Automatically detect optimal churn threshold.
        
        Returns:
            Threshold recommendation with analysis
        """
        df = self.clean()
        
        # Calculate days between purchases for each customer
        df_sorted = df.sort_values(["customer_id", "date"])
        df_sorted["prev_date"] = df_sorted.groupby("customer_id")["date"].shift(1)
        df_sorted["gap_days"] = (df_sorted["date"] - df_sorted["prev_date"]).dt.days
        
        gaps = df_sorted["gap_days"].dropna()
        
        if len(gaps) == 0:
            return {
                "recommended_threshold": config.CHURN_THRESHOLD_DEFAULT,
                "analysis": "Insufficient data for analysis"
            }
        
        # Calculate percentiles
        percentiles = {
            "p50": float(gaps.quantile(0.50)),
            "p75": float(gaps.quantile(0.75)),
            "p90": float(gaps.quantile(0.90)),
            "p95": float(gaps.quantile(0.95))
        }
        
        # Recommend threshold at ~75-90 percentile
        recommended = int(gaps.quantile(0.80))
        # Bound between 30 and 180 days
        recommended = max(30, min(180, recommended))
        
        # Calculate churn rate at different thresholds
        reference_date = df["date"].max()
        customer_last_purchase = df.groupby("customer_id")["date"].max()
        days_since_purchase = (reference_date - customer_last_purchase).dt.days
        
        churn_rates = {}
        for threshold in [30, 45, 60, 90, 120, 180]:
            churned = (days_since_purchase > threshold).sum()
            total = len(days_since_purchase)
            churn_rates[threshold] = churned / total if total > 0 else 0
            
        return {
            "recommended_threshold": recommended,
            "percentiles": percentiles,
            "churn_rates_by_threshold": churn_rates,
            "analysis": f"Based on purchase gap distribution, recommended threshold: {recommended} days"
        }
    
    def get_churn_labels(self, threshold_days: int) -> pd.Series:
        """
        Create churn labels for each customer.
        
        Args:
            threshold_days: Number of days without purchase to consider churned
            
        Returns:
            Series with customer_id as index and is_churned (0/1) as values
        """
        df = self.clean()
        
        reference_date = df["date"].max()
        customer_last_purchase = df.groupby("customer_id")["date"].max()
        days_since_purchase = (reference_date - customer_last_purchase).dt.days
        
        is_churned = (days_since_purchase > threshold_days).astype(int)
        is_churned.name = "is_churned"
        
        return is_churned

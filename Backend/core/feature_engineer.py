"""Feature Engineering - Creates ML features from transaction data."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class FeatureEngineer:
    """Transforms transaction-level data into customer-level ML features."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with cleaned transaction data.
        
        Args:
            df: Cleaned DataFrame with columns: customer_id, date, amount, product (optional)
        """
        self.df = df.copy()
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.reference_date = self.df["date"].max()
        
    def create_rfm_features(self) -> pd.DataFrame:
        """
        Create RFM (Recency, Frequency, Monetary) features.
        
        Returns:
            DataFrame with customer_id index and RFM features
        """
        df = self.df.copy()
        
        # Group by customer
        rfm = df.groupby("customer_id").agg({
            "date": ["max", "min", "count"],
            "amount": ["sum", "mean", "std", "min", "max"]
        })
        
        # Flatten column names
        rfm.columns = [
            "last_purchase_date", "first_purchase_date", "purchase_count",
            "total_spend", "avg_order_value", "std_order_value", 
            "min_order_value", "max_order_value"
        ]
        
        # Recency features
        rfm["days_since_last_purchase"] = (self.reference_date - rfm["last_purchase_date"]).dt.days
        rfm["days_since_first_purchase"] = (self.reference_date - rfm["first_purchase_date"]).dt.days
        rfm["customer_age_days"] = rfm["days_since_first_purchase"]
        
        # Fill NaN in std with 0 (single purchase customers)
        rfm["std_order_value"] = rfm["std_order_value"].fillna(0)
        
        # Drop date columns (keep derived features)
        rfm = rfm.drop(columns=["last_purchase_date", "first_purchase_date"])
        
        return rfm
    
    def create_temporal_features(self) -> pd.DataFrame:
        """
        Create time-based behavioral features.
        
        Returns:
            DataFrame with temporal features
        """
        df = self.df.copy()
        df = df.sort_values(["customer_id", "date"])
        
        # Calculate gaps between purchases
        df["prev_date"] = df.groupby("customer_id")["date"].shift(1)
        df["gap_days"] = (df["date"] - df["prev_date"]).dt.days
        
        # Aggregate gap statistics
        gap_stats = df.groupby("customer_id")["gap_days"].agg([
            ("avg_days_between_purchases", "mean"),
            ("std_days_between_purchases", "std"),
            ("max_gap_days", "max"),
            ("min_gap_days", "min")
        ])
        
        gap_stats = gap_stats.fillna(0)
        
        # Purchase velocity (purchases per month)
        customer_span = df.groupby("customer_id").agg({
            "date": ["min", "max", "count"]
        })
        customer_span.columns = ["first_date", "last_date", "count"]
        
        span_days = (customer_span["last_date"] - customer_span["first_date"]).dt.days
        span_months = span_days / 30.44  # Average days per month
        
        gap_stats["purchase_velocity"] = customer_span["count"] / (span_months + 1)  # +1 to avoid division by zero
        
        # Recent trend (last 3 months vs previous)
        three_months_ago = self.reference_date - pd.Timedelta(days=90)
        six_months_ago = self.reference_date - pd.Timedelta(days=180)
        
        recent_purchases = df[df["date"] >= three_months_ago].groupby("customer_id").size()
        older_purchases = df[(df["date"] >= six_months_ago) & (df["date"] < three_months_ago)].groupby("customer_id").size()
        
        # Merge and calculate trend
        trend_df = pd.DataFrame({
            "recent_3m_purchases": recent_purchases,
            "older_3m_purchases": older_purchases
        }).fillna(0)
        
        trend_df["purchase_trend"] = trend_df["recent_3m_purchases"] - trend_df["older_3m_purchases"]
        trend_df["is_declining"] = (trend_df["purchase_trend"] < 0).astype(int)
        
        # Merge all temporal features
        temporal = gap_stats.join(trend_df[["recent_3m_purchases", "purchase_trend", "is_declining"]], how="left")
        temporal = temporal.fillna(0)
        
        return temporal
    
    def create_behavioral_features(self) -> pd.DataFrame:
        """
        Create behavioral pattern features.
        
        Returns:
            DataFrame with behavioral features
        """
        df = self.df.copy()
        
        behavioral = pd.DataFrame(index=df["customer_id"].unique())
        
        # Day of week preferences
        df["day_of_week"] = df["date"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        
        weekend_ratio = df.groupby("customer_id")["is_weekend"].mean()
        behavioral["weekend_purchase_ratio"] = weekend_ratio
        
        # Month preferences (seasonality)
        df["month"] = df["date"].dt.month
        monthly_distribution = df.groupby("customer_id")["month"].agg(lambda x: x.value_counts().index[0])
        behavioral["favorite_month"] = monthly_distribution
        
        # Calculate purchase consistency (coefficient of variation of gaps)
        df = df.sort_values(["customer_id", "date"])
        df["prev_date"] = df.groupby("customer_id")["date"].shift(1)
        df["gap_days"] = (df["date"] - df["prev_date"]).dt.days
        
        gap_cv = df.groupby("customer_id")["gap_days"].agg(
            lambda x: x.std() / x.mean() if x.mean() > 0 else 0
        )
        behavioral["purchase_consistency"] = 1 / (gap_cv + 1)  # Higher = more consistent
        
        # Product diversity (if product column exists)
        if "product" in df.columns:
            product_diversity = df.groupby("customer_id")["product"].nunique()
            behavioral["product_diversity"] = product_diversity
            
            # Favorite product
            favorite_product = df.groupby("customer_id")["product"].agg(
                lambda x: x.value_counts().index[0] if len(x) > 0 else "unknown"
            )
            behavioral["favorite_product"] = favorite_product
        
        behavioral = behavioral.fillna(0)
        
        return behavioral
    
    def create_value_features(self) -> pd.DataFrame:
        """
        Create customer value features.
        
        Returns:
            DataFrame with value-based features
        """
        df = self.df.copy()
        
        value = pd.DataFrame(index=df["customer_id"].unique())
        
        # Basic value metrics
        customer_value = df.groupby("customer_id")["amount"].agg(["sum", "mean", "count"])
        customer_value.columns = ["total_spend", "avg_spend", "transaction_count"]
        
        # Customer Lifetime Value estimate (simple model)
        # CLV = (Avg Order Value) × (Purchase Frequency per year) × (Expected Lifespan in years)
        customer_span = df.groupby("customer_id").agg({
            "date": ["min", "max"]
        })
        customer_span.columns = ["first_date", "last_date"]
        
        lifespan_days = (customer_span["last_date"] - customer_span["first_date"]).dt.days
        lifespan_years = lifespan_days / 365.25
        
        yearly_frequency = customer_value["transaction_count"] / (lifespan_years + 0.1)  # Avoid division by zero
        expected_lifespan = 2.0  # Assume 2 years expected lifespan
        
        value["estimated_clv"] = customer_value["avg_spend"] * yearly_frequency * expected_lifespan
        
        # Percentile rank by spend
        value["spend_percentile"] = customer_value["total_spend"].rank(pct=True) * 100
        
        # Value tier (quartiles) - with error handling for small datasets
        try:
            value["value_tier"] = pd.qcut(
                customer_value["total_spend"].dropna(), 
                q=4, 
                labels=["bronze", "silver", "gold", "platinum"],
                duplicates="drop"
            )
        except (ValueError, Exception):
            # Fallback: use median split for small datasets
            median_spend = customer_value["total_spend"].median()
            value["value_tier"] = customer_value["total_spend"].apply(
                lambda x: "gold" if x >= median_spend else "bronze"
            )
        
        # Recent value trend
        three_months_ago = self.reference_date - pd.Timedelta(days=90)
        recent_spend = df[df["date"] >= three_months_ago].groupby("customer_id")["amount"].sum()
        older_spend = df[df["date"] < three_months_ago].groupby("customer_id")["amount"].sum()
        
        spend_comparison = pd.DataFrame({
            "recent_spend": recent_spend,
            "older_spend": older_spend
        }).fillna(0)
        
        # Normalized spend trend
        value["spend_trend_ratio"] = spend_comparison["recent_spend"] / (spend_comparison["older_spend"] + 1)
        
        # Handle categorical column separately before fillna
        if "value_tier" in value.columns:
            value["value_tier"] = value["value_tier"].astype(str).replace("nan", "bronze")
        
        # Fill NaN for numeric columns only
        numeric_cols = value.select_dtypes(include=['float64', 'int64']).columns
        value[numeric_cols] = value[numeric_cols].fillna(0)
        
        return value
    
    def create_all_features(self) -> pd.DataFrame:
        """
        Create all features and combine into single DataFrame.
        
        Returns:
            Complete feature matrix with customer_id as index
        """
        # Generate all feature groups
        rfm = self.create_rfm_features()
        temporal = self.create_temporal_features()
        behavioral = self.create_behavioral_features()
        value = self.create_value_features()
        
        # Combine all features
        features = rfm.join(temporal, how="outer")
        features = features.join(behavioral, how="outer")
        features = features.join(value, how="outer")
        
        # Handle categorical columns - convert to numeric
        categorical_cols = features.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in categorical_cols:
            if col == "value_tier":
                # Ordinal encoding for tiers
                tier_map = {"bronze": 1, "silver": 2, "gold": 3, "platinum": 4, "nan": 1, "0": 1}
                features[col] = features[col].astype(str).map(tier_map).fillna(1).astype(int)
            else:
                # Drop other categorical for now
                features = features.drop(columns=[col], errors='ignore')
        
        # Fill remaining NaN
        features = features.fillna(0)
        
        # Replace infinities
        features = features.replace([np.inf, -np.inf], 0)
        
        # Ensure all columns are numeric
        for col in features.columns:
            features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0)
        
        return features
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions for all features.
        
        Returns:
            Dictionary mapping feature names to descriptions
        """
        return {
            # RFM
            "days_since_last_purchase": "Days since customer's last purchase (Recency)",
            "days_since_first_purchase": "Days since customer's first purchase (Customer Age)",
            "customer_age_days": "Total days as customer",
            "purchase_count": "Total number of purchases (Frequency)",
            "total_spend": "Total amount spent (Monetary)",
            "avg_order_value": "Average order value",
            "std_order_value": "Standard deviation of order values",
            "min_order_value": "Minimum order amount",
            "max_order_value": "Maximum order amount",
            
            # Temporal
            "avg_days_between_purchases": "Average gap between purchases",
            "std_days_between_purchases": "Variation in purchase timing",
            "max_gap_days": "Longest gap between purchases",
            "min_gap_days": "Shortest gap between purchases",
            "purchase_velocity": "Purchases per month",
            "recent_3m_purchases": "Number of purchases in last 3 months",
            "purchase_trend": "Change in purchases (recent vs older)",
            "is_declining": "Flag if purchase frequency is declining",
            
            # Behavioral
            "weekend_purchase_ratio": "Proportion of purchases on weekends",
            "favorite_month": "Month with most purchases",
            "purchase_consistency": "Regularity of purchase timing",
            "product_diversity": "Number of unique products purchased",
            
            # Value
            "estimated_clv": "Estimated Customer Lifetime Value",
            "spend_percentile": "Percentile rank by total spend",
            "value_tier": "Customer value tier (1=bronze to 4=platinum)",
            "spend_trend_ratio": "Ratio of recent to older spending"
        }

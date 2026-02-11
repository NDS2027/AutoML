"""Insight Generator - Business insights and ROI calculations."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class CustomerSegment:
    """Represents a customer risk segment."""
    name: str
    count: int
    total_clv: float
    avg_clv: float
    avg_churn_prob: float
    characteristics: Dict[str, float]


class InsightGenerator:
    """Generates business insights, recommendations, and ROI calculations."""
    
    def __init__(
        self,
        predictions_df: pd.DataFrame,
        feature_importance: Dict[str, float]
    ):
        """
        Initialize insight generator.
        
        Args:
            predictions_df: DataFrame with columns:
                - customer_id, churn_probability, estimated_clv, 
                - plus feature columns
            feature_importance: Dict of feature -> importance score
        """
        self.df = predictions_df.copy()
        self.feature_importance = feature_importance
        self._assign_risk_tiers()
        
    def _assign_risk_tiers(self):
        """Assign risk tiers based on churn probability."""
        self.df["risk_tier"] = pd.cut(
            self.df["churn_probability"],
            bins=[0, 0.4, 0.7, 1.0],
            labels=["low", "medium", "high"]
        )
        
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get high-level summary statistics.
        
        Returns:
            Summary dictionary
        """
        total_customers = len(self.df)
        churned_predicted = (self.df["churn_probability"] >= 0.5).sum()
        churn_rate = churned_predicted / total_customers if total_customers > 0 else 0
        
        high_risk = self.df[self.df["risk_tier"] == "high"]
        revenue_at_risk = high_risk["estimated_clv"].sum() if "estimated_clv" in self.df.columns else 0
        
        return {
            "total_customers": int(total_customers),
            "churn_rate": float(churn_rate),
            "high_risk_count": int(len(high_risk)),
            "medium_risk_count": int((self.df["risk_tier"] == "medium").sum()),
            "low_risk_count": int((self.df["risk_tier"] == "low").sum()),
            "revenue_at_risk": float(revenue_at_risk),
            "avg_churn_probability": float(self.df["churn_probability"].mean())
        }
    
    def get_segment_analysis(self) -> Dict[str, CustomerSegment]:
        """
        Analyze each risk segment.
        
        Returns:
            Dictionary of segment name -> CustomerSegment
        """
        segments = {}
        
        for tier in ["high", "medium", "low"]:
            segment_df = self.df[self.df["risk_tier"] == tier]
            
            if len(segment_df) == 0:
                continue
                
            # Calculate segment characteristics
            characteristics = {}
            for feature in list(self.feature_importance.keys())[:5]:
                if feature in segment_df.columns:
                    characteristics[feature] = float(segment_df[feature].mean())
                    
            clv_col = "estimated_clv" if "estimated_clv" in segment_df.columns else None
            
            segments[tier] = CustomerSegment(
                name=tier,
                count=len(segment_df),
                total_clv=float(segment_df[clv_col].sum()) if clv_col else 0,
                avg_clv=float(segment_df[clv_col].mean()) if clv_col else 0,
                avg_churn_prob=float(segment_df["churn_probability"].mean()),
                characteristics=characteristics
            )
            
        return segments
    
    def get_top_churn_drivers(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get the top factors driving churn.
        
        Args:
            n: Number of top drivers to return
            
        Returns:
            List of driver dictionaries
        """
        # Combine feature importance with segment analysis
        high_risk = self.df[self.df["risk_tier"] == "high"]
        all_customers = self.df
        
        drivers = []
        for feature, importance in sorted(
            self.feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:n]:
            if feature not in self.df.columns:
                continue
                
            high_risk_avg = high_risk[feature].mean() if len(high_risk) > 0 else 0
            overall_avg = all_customers[feature].mean()
            
            # Calculate difference
            diff_pct = ((high_risk_avg - overall_avg) / overall_avg * 100) if overall_avg != 0 else 0
            
            drivers.append({
                "feature": feature,
                "importance": float(importance * 100),
                "high_risk_avg": float(high_risk_avg),
                "overall_avg": float(overall_avg),
                "difference_pct": float(diff_pct),
                "insight": self._generate_driver_insight(feature, high_risk_avg, overall_avg)
            })
            
        return drivers
    
    def _generate_driver_insight(
        self, 
        feature: str, 
        high_risk_avg: float, 
        overall_avg: float
    ) -> str:
        """Generate human-readable insight for a driver."""
        diff = high_risk_avg - overall_avg
        
        if "days_since" in feature.lower():
            if diff > 0:
                return f"High-risk customers average {high_risk_avg:.0f} days since last purchase (vs {overall_avg:.0f} overall)"
        elif "purchase_count" in feature.lower() or "frequency" in feature.lower():
            if diff < 0:
                return f"High-risk customers have fewer purchases ({high_risk_avg:.1f} vs {overall_avg:.1f})"
        elif "spend" in feature.lower() or "clv" in feature.lower():
            return f"Average value: ${high_risk_avg:.0f} for high-risk vs ${overall_avg:.0f} overall"
        elif "declining" in feature.lower():
            return f"{high_risk_avg*100:.0f}% of high-risk customers show declining activity"
            
        return f"High-risk avg: {high_risk_avg:.2f}, Overall avg: {overall_avg:.2f}"
    
    def calculate_roi(
        self,
        target_count: int = 100,
        discount_percent: float = 20,
        success_rate: float = 0.35,
        campaign_cost_per_customer: float = 0.5
    ) -> Dict[str, Any]:
        """
        Calculate ROI for a retention campaign.
        
        Args:
            target_count: Number of customers to target
            discount_percent: Discount to offer (%)
            success_rate: Expected retention success rate
            campaign_cost_per_customer: Cost to reach each customer
            
        Returns:
            ROI calculation breakdown
        """
        # Get top N high-risk customers by CLV
        high_risk = self.df[self.df["risk_tier"] == "high"].copy()
        
        if "estimated_clv" in high_risk.columns:
            high_risk = high_risk.sort_values("estimated_clv", ascending=False)
        
        target_customers = high_risk.head(target_count)
        actual_target_count = len(target_customers)
        
        if actual_target_count == 0:
            return {
                "error": "No high-risk customers found",
                "target_count": 0
            }
            
        # Calculate metrics
        clv_col = "estimated_clv" if "estimated_clv" in target_customers.columns else None
        avg_clv = target_customers[clv_col].mean() if clv_col else 100  # Default CLV
        total_clv_at_risk = target_customers[clv_col].sum() if clv_col else actual_target_count * 100
        
        # Cost calculations
        avg_order = avg_clv / 4  # Assume CLV = 4x avg order
        discount_cost_per_customer = avg_order * (discount_percent / 100)
        total_discount_cost = discount_cost_per_customer * actual_target_count * success_rate
        total_campaign_cost = campaign_cost_per_customer * actual_target_count
        total_cost = total_discount_cost + total_campaign_cost
        
        # Revenue calculations
        expected_saves = int(actual_target_count * success_rate)
        saved_revenue = expected_saves * avg_clv
        
        # ROI
        net_benefit = saved_revenue - total_cost
        roi_percent = (net_benefit / total_cost * 100) if total_cost > 0 else 0
        
        return {
            "target_count": int(actual_target_count),
            "avg_clv": float(avg_clv),
            "total_clv_at_risk": float(total_clv_at_risk),
            "discount_percent": float(discount_percent),
            "success_rate": float(success_rate),
            "campaign_cost": float(total_campaign_cost),
            "discount_cost": float(total_discount_cost),
            "total_cost": float(total_cost),
            "expected_saves": int(expected_saves),
            "saved_revenue": float(saved_revenue),
            "net_benefit": float(net_benefit),
            "roi_percent": float(roi_percent),
            "recommendation": self._get_roi_recommendation(roi_percent)
        }
    
    def _get_roi_recommendation(self, roi_percent: float) -> str:
        """Generate recommendation based on ROI."""
        if roi_percent >= 500:
            return "EXCELLENT - Highly recommended campaign"
        elif roi_percent >= 200:
            return "GOOD - Recommended to proceed"
        elif roi_percent >= 100:
            return "MODERATE - Proceed with caution"
        elif roi_percent >= 0:
            return "LOW - Consider adjusting parameters"
        else:
            return "NEGATIVE - Not recommended"
    
    def generate_recommendations(self) -> List[Dict[str, Any]]:
        """
        Generate actionable recommendations.
        
        Returns:
            List of recommendation dictionaries
        """
        summary = self.get_summary_stats()
        segments = self.get_segment_analysis()
        
        recommendations = []
        
        # High-risk targeting
        if "high" in segments:
            high = segments["high"]
            roi = self.calculate_roi(target_count=min(100, high.count))
            
            recommendations.append({
                "priority": 1,
                "type": "immediate_action",
                "title": "Target High-Risk Customers",
                "description": f"Focus on {high.count} high-risk customers worth ${high.total_clv:,.0f}",
                "expected_roi": roi["roi_percent"],
                "action": f"Send personalized offer to top {min(100, high.count)} customers"
            })
            
        # Medium-risk prevention
        if "medium" in segments:
            medium = segments["medium"]
            recommendations.append({
                "priority": 2,
                "type": "prevention",
                "title": "Engage Medium-Risk Customers",
                "description": f"Prevent {medium.count} medium-risk customers from becoming high-risk",
                "action": "Send engagement emails with product recommendations"
            })
            
        # General insights
        drivers = self.get_top_churn_drivers(n=3)
        if drivers:
            top_driver = drivers[0]
            recommendations.append({
                "priority": 3,
                "type": "insight",
                "title": f"Address Key Driver: {top_driver['feature']}",
                "description": top_driver["insight"],
                "action": "Review business processes related to this factor"
            })
            
        return recommendations
    
    def get_customer_list(
        self,
        risk_tier: Optional[str] = None,
        min_clv: Optional[float] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Get filtered list of customers.
        
        Args:
            risk_tier: Filter by risk tier (high/medium/low)
            min_clv: Minimum CLV filter
            limit: Max customers to return
            offset: Pagination offset
            
        Returns:
            Customer list with pagination info
        """
        df = self.df.copy()
        
        # Apply filters
        if risk_tier:
            df = df[df["risk_tier"] == risk_tier]
            
        if min_clv and "estimated_clv" in df.columns:
            df = df[df["estimated_clv"] >= min_clv]
            
        # Sort by churn probability descending
        df = df.sort_values("churn_probability", ascending=False)
        
        total = len(df)
        df = df.iloc[offset:offset + limit]
        
        # Convert to list of dicts
        customers = []
        for _, row in df.iterrows():
            customer = {
                "customer_id": str(row.get("customer_id", row.name)),
                "churn_probability": float(row["churn_probability"]),
                "risk_tier": str(row["risk_tier"]),
            }
            
            if "estimated_clv" in row:
                customer["clv"] = float(row["estimated_clv"])
            if "days_since_last_purchase" in row:
                customer["days_since_purchase"] = int(row["days_since_last_purchase"])
                
            customers.append(customer)
            
        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "customers": customers
        }

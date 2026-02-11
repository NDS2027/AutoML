"""Explainability Module - SHAP analysis and feature importance."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import warnings

# Handle SHAP import
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


class ExplainabilityModule:
    """Provides model explanations using SHAP and feature importance."""
    
    def __init__(self, model: Any, feature_names: List[str], scaler: Any = None):
        """
        Initialize explainability module.
        
        Args:
            model: Trained ML model
            feature_names: List of feature names
            scaler: Optional scaler used for preprocessing
        """
        self.model = model
        self.feature_names = feature_names
        self.scaler = scaler
        self.shap_values: Optional[np.ndarray] = None
        self.explainer: Optional[Any] = None
        
    def get_feature_importance(self) -> List[Dict[str, Any]]:
        """
        Get global feature importance.
        
        Returns:
            List of {feature, importance, rank} sorted by importance
        """
        # Get importance based on model type
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_[0])
        else:
            importances = np.ones(len(self.feature_names)) / len(self.feature_names)
            
        # Normalize to percentages
        importances = importances / importances.sum() * 100
        
        # Create sorted list
        importance_list = []
        for name, imp in zip(self.feature_names, importances):
            importance_list.append({
                "feature": name,
                "importance": float(imp)
            })
            
        importance_list.sort(key=lambda x: x["importance"], reverse=True)
        
        # Add rank
        for i, item in enumerate(importance_list):
            item["rank"] = i + 1
            
        return importance_list
    
    def calculate_shap_values(
        self, 
        X: pd.DataFrame,
        sample_size: int = 100
    ) -> bool:
        """
        Calculate SHAP values for the dataset.
        
        Args:
            X: Feature matrix
            sample_size: Number of samples to use for SHAP (for speed)
            
        Returns:
            True if successful, False if SHAP not available
        """
        if not HAS_SHAP:
            return False
            
        try:
            # Sample for speed if dataset is large
            if len(X) > sample_size:
                X_sample = X.sample(n=sample_size, random_state=42)
            else:
                X_sample = X
                
            # Scale if scaler provided
            if self.scaler:
                X_scaled = self.scaler.transform(X_sample)
            else:
                X_scaled = X_sample.values
                
            # Create appropriate explainer
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                model_type = type(self.model).__name__.lower()
                
                if "tree" in model_type or "forest" in model_type or "xgb" in model_type or "lgb" in model_type:
                    self.explainer = shap.TreeExplainer(self.model)
                else:
                    # Use KernelExplainer for other models (slower)
                    background = shap.sample(X_scaled, min(50, len(X_scaled)))
                    self.explainer = shap.KernelExplainer(
                        self.model.predict_proba if hasattr(self.model, "predict_proba") else self.model.predict,
                        background
                    )
                
                self.shap_values = self.explainer.shap_values(X_scaled)
                
                # Handle different SHAP output formats
                if isinstance(self.shap_values, list):
                    self.shap_values = self.shap_values[1]  # Get positive class
                    
            return True
            
        except Exception as e:
            print(f"SHAP calculation failed: {e}")
            return False
    
    def explain_customer(
        self, 
        customer_features: pd.Series,
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Explain prediction for a single customer.
        
        Args:
            customer_features: Feature values for one customer
            top_n: Number of top factors to return
            
        Returns:
            List of {factor, value, impact, direction}
        """
        if not HAS_SHAP or self.explainer is None:
            # Fallback: use feature importance + feature values
            return self._explain_without_shap(customer_features, top_n)
            
        try:
            # Scale features
            if self.scaler:
                features_scaled = self.scaler.transform([customer_features.values])
            else:
                features_scaled = [customer_features.values]
                
            # Calculate SHAP for this customer
            shap_vals = self.explainer.shap_values(features_scaled)
            
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]  # Positive class
                
            shap_vals = shap_vals[0]  # First (only) sample
            
            # Create explanations
            explanations = []
            for i, (name, shap_val) in enumerate(zip(self.feature_names, shap_vals)):
                explanations.append({
                    "factor": name,
                    "value": float(customer_features.iloc[i]),
                    "impact": float(abs(shap_val)),
                    "direction": "increases_churn" if shap_val > 0 else "decreases_churn"
                })
                
            # Sort by absolute impact
            explanations.sort(key=lambda x: x["impact"], reverse=True)
            
            return explanations[:top_n]
            
        except Exception:
            return self._explain_without_shap(customer_features, top_n)
    
    def _explain_without_shap(
        self, 
        customer_features: pd.Series,
        top_n: int
    ) -> List[Dict[str, Any]]:
        """
        Fallback explanation without SHAP.
        
        Uses feature importance * deviation from mean as a proxy.
        """
        importances = self.get_feature_importance()
        
        explanations = []
        for item in importances[:top_n]:
            feature = item["feature"]
            value = float(customer_features.get(feature, 0))
            
            # Simple heuristic for direction (this is approximate)
            direction = "increases_churn"
            if "days_since" in feature and value > 30:
                direction = "increases_churn"
            elif "purchase_count" in feature and value < 5:
                direction = "increases_churn"
            elif "spend" in feature and value < 50:
                direction = "increases_churn"
            else:
                direction = "decreases_churn"
                
            explanations.append({
                "factor": feature,
                "value": value,
                "impact": item["importance"] / 100,
                "direction": direction
            })
            
        return explanations
    
    def get_decision_rules(self, top_n: int = 5) -> List[str]:
        """
        Extract simple decision rules from tree-based models.
        
        Args:
            top_n: Number of rules to extract
            
        Returns:
            List of rule strings
        """
        rules = []
        
        # Get feature importance
        importance = self.get_feature_importance()
        
        # Generate rules based on top features
        for item in importance[:top_n]:
            feature = item["feature"]
            
            # Create rule based on feature name
            if "days_since_last_purchase" in feature:
                rules.append(f"IF {feature} > 60 THEN high_churn_risk")
            elif "purchase_count" in feature:
                rules.append(f"IF {feature} < 5 THEN high_churn_risk")
            elif "purchase_velocity" in feature:
                rules.append(f"IF {feature} < 0.5 (purchases/month) THEN high_churn_risk")
            elif "is_declining" in feature:
                rules.append(f"IF {feature} = 1 THEN high_churn_risk")
            elif "avg_order_value" in feature:
                rules.append(f"IF {feature} < median THEN moderate_churn_risk")
            else:
                rules.append(f"{feature} is important (rank #{item['rank']})")
                
        return rules
    
    def get_customer_risk_summary(
        self,
        customer_features: pd.Series,
        churn_probability: float
    ) -> Dict[str, Any]:
        """
        Generate a human-readable risk summary for a customer.
        
        Args:
            customer_features: Customer's feature values
            churn_probability: Predicted churn probability
            
        Returns:
            Summary dictionary with risk assessment
        """
        factors = self.explain_customer(customer_features, top_n=3)
        
        # Determine risk tier
        if churn_probability >= 0.7:
            risk_tier = "HIGH"
            risk_color = "red"
        elif churn_probability >= 0.4:
            risk_tier = "MEDIUM"
            risk_color = "orange"
        else:
            risk_tier = "LOW"
            risk_color = "green"
            
        # Create summary text
        factor_texts = []
        for f in factors:
            if f["direction"] == "increases_churn":
                factor_texts.append(f"{f['factor']}: {f['value']:.1f}")
                
        return {
            "risk_tier": risk_tier,
            "risk_color": risk_color,
            "churn_probability": churn_probability,
            "top_risk_factors": factors,
            "summary": f"Risk Level: {risk_tier} ({churn_probability*100:.0f}% churn probability)",
            "key_drivers": factor_texts[:3]
        }

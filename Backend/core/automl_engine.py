"""AutoML Engine - Trains and evaluates multiple ML models."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Import XGBoost and LightGBM (handle if not installed)
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


class AutoMLEngine:
    """Trains multiple ML models and selects the best performer."""
    
    def __init__(self, random_state: int = config.RANDOM_STATE):
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict] = {}
        self.champion_model: Optional[str] = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        
    def prepare_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        apply_smote: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training.
        
        Args:
            X: Feature matrix
            y: Target labels
            apply_smote: Whether to apply SMOTE for imbalanced data
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        self.feature_names = X.columns.tolist()
        
        # Train-test split
        # Handle edge case: if too few samples in a class, don't stratify
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=config.TEST_SIZE,
                random_state=self.random_state,
                stratify=y
            )
        except ValueError:
            # Fallback without stratification for small datasets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=config.TEST_SIZE,
                random_state=self.random_state
            )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Apply SMOTE if minority class is less than 30% and we have enough samples
        minority_ratio = y_train.mean() if hasattr(y_train, 'mean') else np.mean(y_train)
        min_class_count = min(np.sum(y_train == 0), np.sum(y_train == 1)) if len(y_train) > 0 else 0
        if apply_smote and (minority_ratio < 0.3 or minority_ratio > 0.7) and min_class_count >= 6:
            try:
                smote = SMOTE(random_state=self.random_state, k_neighbors=min(5, min_class_count - 1))
                X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
            except Exception:
                pass  # Skip SMOTE if it fails
            
        return X_train_scaled, X_test_scaled, y_train.values if hasattr(y_train, 'values') else y_train, y_test.values
    
    def get_models(self) -> Dict[str, Any]:
        """
        Get dictionary of models to train.
        
        Returns:
            Dictionary of model name -> model instance
        """
        models = {
            "logistic_regression": LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                class_weight="balanced"
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                random_state=self.random_state,
                class_weight="balanced",
                n_jobs=-1
            ),
            "svm": SVC(
                kernel="rbf",
                probability=True,
                random_state=self.random_state,
                class_weight="balanced"
            )
        }
        
        if HAS_XGBOOST:
            models["xgboost"] = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                eval_metric="logloss"
            )
            
        if HAS_LIGHTGBM:
            models["lightgbm"] = LGBMClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                random_state=self.random_state,
                class_weight="balanced",
                verbose=-1
            )
            
        return models
    
    def train_model(
        self,
        name: str,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Train a single model and evaluate it.
        
        Returns:
            Dictionary with performance metrics
        """
        # Cross-validation
        cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
        
        # Train on full training set
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        # Calculate metrics
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
        }
        
        # ROC-AUC may fail if test set has only one class
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
        except ValueError:
            metrics["roc_auc"] = float(metrics["f1_score"])  # Fallback
        
        # Store model
        self.models[name] = model
        self.results[name] = metrics
        
        return metrics
    
    def train_all_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Dict]:
        """
        Train all models and compare performance.
        
        Args:
            X: Feature matrix
            y: Target labels
            progress_callback: Optional callback(step, total, message)
            
        Returns:
            Dictionary of all model results
        """
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(X, y)
        
        # Get models
        models_to_train = self.get_models()
        total_models = len(models_to_train)
        
        # Train each model
        for i, (name, model) in enumerate(models_to_train.items()):
            if progress_callback:
                progress_callback(i + 1, total_models, f"Training {name}")
                
            self.train_model(name, model, X_train, y_train, X_test, y_test)
            
        # Select champion
        self._select_champion()
        
        return self.results
    
    def _select_champion(self):
        """Select the best performing model."""
        if not self.results:
            return
            
        # Weighted scoring: F1 (40%) + Recall (30%) + ROC-AUC (20%) + Stability (10%)
        scores = {}
        for name, metrics in self.results.items():
            stability = 1 - min(metrics["cv_std"], 0.1)  # Penalize high variance
            scores[name] = (
                0.4 * metrics["f1_score"] +
                0.3 * metrics["recall"] +
                0.2 * metrics["roc_auc"] +
                0.1 * stability
            )
            
        self.champion_model = max(scores, key=scores.get)
        
        # Mark champion in results
        for name in self.results:
            self.results[name]["is_champion"] = (name == self.champion_model)
    
    def get_champion(self) -> Tuple[str, Any]:
        """
        Get the champion model.
        
        Returns:
            Tuple of (model_name, model_instance)
        """
        if not self.champion_model:
            raise ValueError("No models trained yet")
        return self.champion_model, self.models[self.champion_model]
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using champion model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if not self.champion_model:
            raise ValueError("No models trained yet")
            
        model = self.models[self.champion_model]
        X_scaled = self.scaler.transform(X)
        
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]
        
        return predictions, probabilities
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from champion model.
        
        Returns:
            Dictionary of feature -> importance score
        """
        if not self.champion_model:
            raise ValueError("No models trained yet")
            
        model = self.models[self.champion_model]
        
        # Get importance based on model type
        if hasattr(model, "feature_importances_"):
            # Tree-based models
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            # Linear models
            importances = np.abs(model.coef_[0])
        else:
            # Fallback: equal importance
            importances = np.ones(len(self.feature_names)) / len(self.feature_names)
            
        # Normalize
        importances = importances / importances.sum()
        
        return dict(zip(self.feature_names, importances))
    
    def save_models(self, save_dir: Path, session_id: str):
        """
        Save all trained models to disk.
        
        Args:
            save_dir: Directory to save models
            session_id: Session identifier for naming
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save scaler
        joblib.dump(self.scaler, save_dir / f"{session_id}_scaler.pkl")
        
        # Save each model
        for name, model in self.models.items():
            joblib.dump(model, save_dir / f"{session_id}_{name}.pkl")
            
        # Save metadata
        metadata = {
            "champion": self.champion_model,
            "results": self.results,
            "feature_names": self.feature_names
        }
        joblib.dump(metadata, save_dir / f"{session_id}_metadata.pkl")
    
    def load_models(self, save_dir: Path, session_id: str):
        """
        Load trained models from disk.
        
        Args:
            save_dir: Directory containing saved models
            session_id: Session identifier
        """
        # Load scaler
        self.scaler = joblib.load(save_dir / f"{session_id}_scaler.pkl")
        
        # Load metadata
        metadata = joblib.load(save_dir / f"{session_id}_metadata.pkl")
        self.champion_model = metadata["champion"]
        self.results = metadata["results"]
        self.feature_names = metadata["feature_names"]
        
        # Load models
        for name in self.results.keys():
            model_path = save_dir / f"{session_id}_{name}.pkl"
            if model_path.exists():
                self.models[name] = joblib.load(model_path)

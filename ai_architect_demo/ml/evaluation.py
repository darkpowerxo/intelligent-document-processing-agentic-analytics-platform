"""Model evaluation utilities for AI Architect Demo.

This module provides comprehensive model evaluation capabilities including:
- Performance metrics calculation
- Model comparison and benchmarking
- Cross-validation utilities
- Bias detection and fairness metrics
- Automated evaluation reports
"""

import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report,
    roc_auc_score, precision_recall_curve, roc_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
import matplotlib.pyplot as plt
import seaborn as sns

from ai_architect_demo.core.config import settings
from ai_architect_demo.core.logging import get_logger, log_function_call

logger = get_logger(__name__)


class ModelType(Enum):
    """Supported model types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"


class EvaluationMetric(Enum):
    """Available evaluation metrics."""
    # Classification metrics
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ROC_AUC = "roc_auc"
    LOG_LOSS = "log_loss"
    
    # Regression metrics
    MSE = "mse"
    RMSE = "rmse"
    MAE = "mae"
    R2_SCORE = "r2_score"
    MAPE = "mape"
    
    # Custom metrics
    CUSTOM = "custom"


class ModelEvaluationResult(BaseModel):
    """Model evaluation result."""
    model_name: str
    model_type: ModelType
    evaluation_timestamp: datetime
    dataset_info: Dict[str, Any]
    metrics: Dict[str, float]
    confusion_matrix: Optional[List[List[int]]] = None
    feature_importance: Optional[Dict[str, float]] = None
    cross_validation_scores: Optional[Dict[str, List[float]]] = None
    evaluation_time_seconds: float
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ModelComparison(BaseModel):
    """Model comparison report."""
    comparison_timestamp: datetime
    models: List[str]
    metrics_comparison: Dict[str, Dict[str, float]]
    best_model: Dict[str, str]  # metric -> best model name
    statistical_tests: Optional[Dict[str, Any]] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ModelEvaluator:
    """Comprehensive model evaluation and comparison framework."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize model evaluator.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = Path(output_dir) if output_dir else Path("evaluation_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.custom_metrics = {}
        self.evaluation_history = []
    
    def register_custom_metric(self, name: str, func: callable, higher_is_better: bool = True) -> None:
        """Register a custom evaluation metric.
        
        Args:
            name: Metric name
            func: Function that takes (y_true, y_pred) and returns float
            higher_is_better: Whether higher values indicate better performance
        """
        self.custom_metrics[name] = {
            'function': func,
            'higher_is_better': higher_is_better
        }
        logger.info(f"Registered custom metric: {name}")
    
    def evaluate_classification_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
        class_names: Optional[List[str]] = None,
        cross_validate: bool = True,
        cv_folds: int = 5
    ) -> ModelEvaluationResult:
        """Evaluate a classification model.
        
        Args:
            model: Trained model with predict/predict_proba methods
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            class_names: Names of classes
            cross_validate: Whether to perform cross-validation
            cv_folds: Number of CV folds
            
        Returns:
            Evaluation result
        """
        log_function_call("evaluate_classification_model", model_name=model_name)
        
        start_time = time.time()
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = None
        
        if hasattr(model, 'predict_proba'):
            try:
                y_pred_proba = model.predict_proba(X_test)
            except Exception as e:
                logger.warning(f"Could not get prediction probabilities: {e}")
        
        # Calculate metrics
        metrics = self._calculate_classification_metrics(y_test, y_pred, y_pred_proba)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        # Feature importance
        feature_importance = self._get_feature_importance(model, X_test.shape[1])
        
        # Cross-validation
        cv_scores = None
        if cross_validate and X_test.shape[0] > cv_folds:
            cv_scores = self._perform_cross_validation(
                model, X_test, y_test, ModelType.CLASSIFICATION, cv_folds
            )
        
        # Dataset info
        dataset_info = {
            'n_samples': len(y_test),
            'n_features': X_test.shape[1],
            'n_classes': len(np.unique(y_test)),
            'class_distribution': {str(k): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))}
        }
        
        evaluation_time = time.time() - start_time
        
        result = ModelEvaluationResult(
            model_name=model_name,
            model_type=ModelType.CLASSIFICATION,
            evaluation_timestamp=datetime.now(),
            dataset_info=dataset_info,
            metrics=metrics,
            confusion_matrix=cm,
            feature_importance=feature_importance,
            cross_validation_scores=cv_scores,
            evaluation_time_seconds=evaluation_time
        )
        
        # Save result
        self._save_evaluation_result(result)
        self.evaluation_history.append(result)
        
        # Generate plots
        self._generate_classification_plots(result, y_test, y_pred, y_pred_proba, class_names)
        
        return result
    
    def evaluate_regression_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
        cross_validate: bool = True,
        cv_folds: int = 5
    ) -> ModelEvaluationResult:
        """Evaluate a regression model.
        
        Args:
            model: Trained regression model
            X_test: Test features
            y_test: Test targets
            model_name: Name of the model
            cross_validate: Whether to perform cross-validation
            cv_folds: Number of CV folds
            
        Returns:
            Evaluation result
        """
        log_function_call("evaluate_regression_model", model_name=model_name)
        
        start_time = time.time()
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = self._calculate_regression_metrics(y_test, y_pred)
        
        # Feature importance
        feature_importance = self._get_feature_importance(model, X_test.shape[1])
        
        # Cross-validation
        cv_scores = None
        if cross_validate and X_test.shape[0] > cv_folds:
            cv_scores = self._perform_cross_validation(
                model, X_test, y_test, ModelType.REGRESSION, cv_folds
            )
        
        # Dataset info
        dataset_info = {
            'n_samples': len(y_test),
            'n_features': X_test.shape[1],
            'target_mean': float(np.mean(y_test)),
            'target_std': float(np.std(y_test)),
            'target_min': float(np.min(y_test)),
            'target_max': float(np.max(y_test))
        }
        
        evaluation_time = time.time() - start_time
        
        result = ModelEvaluationResult(
            model_name=model_name,
            model_type=ModelType.REGRESSION,
            evaluation_timestamp=datetime.now(),
            dataset_info=dataset_info,
            metrics=metrics,
            feature_importance=feature_importance,
            cross_validation_scores=cv_scores,
            evaluation_time_seconds=evaluation_time
        )
        
        # Save result
        self._save_evaluation_result(result)
        self.evaluation_history.append(result)
        
        # Generate plots
        self._generate_regression_plots(result, y_test, y_pred)
        
        return result
    
    def compare_models(self, evaluation_results: List[ModelEvaluationResult]) -> ModelComparison:
        """Compare multiple model evaluation results.
        
        Args:
            evaluation_results: List of evaluation results to compare
            
        Returns:
            Model comparison report
        """
        log_function_call("compare_models", num_models=len(evaluation_results))
        
        if len(evaluation_results) < 2:
            raise ValueError("Need at least 2 models for comparison")
        
        # Ensure all models are of the same type
        model_types = {result.model_type for result in evaluation_results}
        if len(model_types) > 1:
            raise ValueError("Cannot compare models of different types")
        
        # Build comparison data
        models = [result.model_name for result in evaluation_results]
        metrics_comparison = {}
        
        # Get all available metrics
        all_metrics = set()
        for result in evaluation_results:
            all_metrics.update(result.metrics.keys())
        
        # Build metrics comparison matrix
        for metric in all_metrics:
            metrics_comparison[metric] = {}
            for result in evaluation_results:
                metrics_comparison[metric][result.model_name] = result.metrics.get(metric, np.nan)
        
        # Determine best model for each metric
        best_model = {}
        for metric, model_scores in metrics_comparison.items():
            valid_scores = {model: score for model, score in model_scores.items() if not np.isnan(score)}
            if valid_scores:
                # Determine if higher or lower is better based on common metric knowledge
                higher_is_better = self._is_higher_better(metric)
                if higher_is_better:
                    best_model[metric] = max(valid_scores, key=valid_scores.get)
                else:
                    best_model[metric] = min(valid_scores, key=valid_scores.get)
        
        comparison = ModelComparison(
            comparison_timestamp=datetime.now(),
            models=models,
            metrics_comparison=metrics_comparison,
            best_model=best_model
        )
        
        # Save comparison
        self._save_model_comparison(comparison)
        
        # Generate comparison plots
        self._generate_comparison_plots(comparison)
        
        return comparison
    
    def evaluate_model_bias(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        protected_attributes: Dict[str, np.ndarray],
        model_name: str
    ) -> Dict[str, Any]:
        """Evaluate model for bias and fairness issues.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            protected_attributes: Dict of attribute_name -> attribute_values
            model_name: Name of the model
            
        Returns:
            Bias evaluation results
        """
        log_function_call("evaluate_model_bias", model_name=model_name)
        
        y_pred = model.predict(X_test)
        
        bias_results = {
            'model_name': model_name,
            'evaluation_timestamp': datetime.now().isoformat(),
            'overall_accuracy': accuracy_score(y_test, y_pred),
            'protected_attribute_analysis': {}
        }
        
        for attr_name, attr_values in protected_attributes.items():
            unique_values = np.unique(attr_values)
            attr_analysis = {}
            
            for value in unique_values:
                mask = attr_values == value
                if np.sum(mask) > 0:
                    group_accuracy = accuracy_score(y_test[mask], y_pred[mask])
                    group_precision = precision_score(y_test[mask], y_pred[mask], average='weighted', zero_division=0)
                    group_recall = recall_score(y_test[mask], y_pred[mask], average='weighted', zero_division=0)
                    
                    attr_analysis[str(value)] = {
                        'sample_size': int(np.sum(mask)),
                        'accuracy': float(group_accuracy),
                        'precision': float(group_precision),
                        'recall': float(group_recall)
                    }
            
            # Calculate fairness metrics
            accuracies = [metrics['accuracy'] for metrics in attr_analysis.values()]
            if len(accuracies) > 1:
                attr_analysis['fairness_metrics'] = {
                    'accuracy_difference': float(max(accuracies) - min(accuracies)),
                    'accuracy_ratio': float(min(accuracies) / max(accuracies)) if max(accuracies) > 0 else 0
                }
            
            bias_results['protected_attribute_analysis'][attr_name] = attr_analysis
        
        # Save bias evaluation results
        bias_file = self.output_dir / f"bias_evaluation_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(bias_file, 'w') as f:
            json.dump(bias_results, f, indent=2)
        
        logger.info(f"Bias evaluation completed for {model_name}")
        return bias_results
    
    def _calculate_classification_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate classification metrics."""
        metrics = {}
        
        try:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            if y_pred_proba is not None and y_pred_proba.shape[1] == 2:
                # Binary classification
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            elif y_pred_proba is not None and len(np.unique(y_true)) > 2:
                # Multi-class classification
                try:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
                except ValueError:
                    pass  # Some cases don't support multi-class AUC
            
            # Add custom metrics
            for name, metric_info in self.custom_metrics.items():
                try:
                    custom_score = metric_info['function'](y_true, y_pred)
                    metrics[name] = custom_score
                except Exception as e:
                    logger.warning(f"Custom metric {name} failed: {e}")
        
        except Exception as e:
            logger.error(f"Error calculating classification metrics: {e}")
        
        return metrics
    
    def _calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics."""
        metrics = {}
        
        try:
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2_score'] = r2_score(y_true, y_pred)
            
            # MAPE (Mean Absolute Percentage Error)
            non_zero_mask = y_true != 0
            if np.sum(non_zero_mask) > 0:
                mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
                metrics['mape'] = mape
            
            # Add custom metrics
            for name, metric_info in self.custom_metrics.items():
                try:
                    custom_score = metric_info['function'](y_true, y_pred)
                    metrics[name] = custom_score
                except Exception as e:
                    logger.warning(f"Custom metric {name} failed: {e}")
        
        except Exception as e:
            logger.error(f"Error calculating regression metrics: {e}")
        
        return metrics
    
    def _get_feature_importance(self, model: Any, n_features: int) -> Optional[Dict[str, float]]:
        """Extract feature importance from model."""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_).flatten()
            else:
                return None
            
            if len(importances) == n_features:
                return {f'feature_{i}': float(imp) for i, imp in enumerate(importances)}
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
        
        return None
    
    def _perform_cross_validation(
        self, 
        model: Any, 
        X: np.ndarray, 
        y: np.ndarray, 
        model_type: ModelType, 
        cv_folds: int
    ) -> Dict[str, List[float]]:
        """Perform cross-validation."""
        cv_scores = {}
        
        try:
            if model_type == ModelType.CLASSIFICATION:
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                scoring_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
            else:  # Regression
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                scoring_metrics = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
            
            for metric in scoring_metrics:
                try:
                    scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
                    cv_scores[metric] = scores.tolist()
                except Exception as e:
                    logger.warning(f"Cross-validation failed for metric {metric}: {e}")
        
        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
        
        return cv_scores
    
    def _is_higher_better(self, metric_name: str) -> bool:
        """Determine if higher values are better for a metric."""
        higher_is_better_metrics = {
            'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'r2_score'
        }
        lower_is_better_metrics = {
            'mse', 'rmse', 'mae', 'mape', 'log_loss'
        }
        
        if metric_name.lower() in higher_is_better_metrics:
            return True
        elif metric_name.lower() in lower_is_better_metrics:
            return False
        else:
            # Check custom metrics
            if metric_name in self.custom_metrics:
                return self.custom_metrics[metric_name]['higher_is_better']
            else:
                # Default assumption
                return True
    
    def _save_evaluation_result(self, result: ModelEvaluationResult) -> None:
        """Save evaluation result to file."""
        try:
            filename = f"evaluation_{result.model_name}_{result.evaluation_timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(result.dict(), f, indent=2)
            
            logger.info(f"Evaluation result saved: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save evaluation result: {e}")
    
    def _save_model_comparison(self, comparison: ModelComparison) -> None:
        """Save model comparison to file."""
        try:
            filename = f"comparison_{comparison.comparison_timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(comparison.dict(), f, indent=2)
            
            logger.info(f"Model comparison saved: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save model comparison: {e}")
    
    def _generate_classification_plots(
        self, 
        result: ModelEvaluationResult, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None
    ) -> None:
        """Generate classification visualization plots."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Classification Evaluation: {result.model_name}')
            
            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 0], 
                       xticklabels=class_names, yticklabels=class_names)
            axes[0, 0].set_title('Confusion Matrix')
            axes[0, 0].set_xlabel('Predicted')
            axes[0, 0].set_ylabel('Actual')
            
            # ROC Curve (binary classification only)
            if y_pred_proba is not None and y_pred_proba.shape[1] == 2:
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
                auc_score = result.metrics.get('roc_auc', 0)
                axes[0, 1].plot(fpr, tpr, label=f'ROC (AUC = {auc_score:.3f})')
                axes[0, 1].plot([0, 1], [0, 1], 'k--')
                axes[0, 1].set_xlabel('False Positive Rate')
                axes[0, 1].set_ylabel('True Positive Rate')
                axes[0, 1].set_title('ROC Curve')
                axes[0, 1].legend()
            else:
                axes[0, 1].text(0.5, 0.5, 'ROC Curve\n(Binary Classification Only)', 
                               ha='center', va='center')
                axes[0, 1].set_title('ROC Curve')
            
            # Feature Importance
            if result.feature_importance:
                features = list(result.feature_importance.keys())[:10]  # Top 10
                importances = [result.feature_importance[f] for f in features]
                axes[1, 0].barh(features, importances)
                axes[1, 0].set_title('Top 10 Feature Importances')
                axes[1, 0].set_xlabel('Importance')
            else:
                axes[1, 0].text(0.5, 0.5, 'Feature Importance\nNot Available', 
                               ha='center', va='center')
                axes[1, 0].set_title('Feature Importance')
            
            # Metrics Summary
            metrics_text = '\n'.join([f'{k}: {v:.4f}' for k, v in result.metrics.items()])
            axes[1, 1].text(0.1, 0.9, metrics_text, transform=axes[1, 1].transAxes, 
                           verticalalignment='top', fontfamily='monospace')
            axes[1, 1].set_title('Metrics Summary')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            # Save plot
            plot_filename = f"classification_plots_{result.model_name}_{result.evaluation_timestamp.strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(self.output_dir / plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to generate classification plots: {e}")
    
    def _generate_regression_plots(
        self, 
        result: ModelEvaluationResult, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> None:
        """Generate regression visualization plots."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Regression Evaluation: {result.model_name}')
            
            # Actual vs Predicted
            axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
            axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            axes[0, 0].set_xlabel('Actual Values')
            axes[0, 0].set_ylabel('Predicted Values')
            axes[0, 0].set_title('Actual vs Predicted')
            
            # Residuals Plot
            residuals = y_true - y_pred
            axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
            axes[0, 1].axhline(y=0, color='r', linestyle='--')
            axes[0, 1].set_xlabel('Predicted Values')
            axes[0, 1].set_ylabel('Residuals')
            axes[0, 1].set_title('Residuals Plot')
            
            # Feature Importance
            if result.feature_importance:
                features = list(result.feature_importance.keys())[:10]  # Top 10
                importances = [result.feature_importance[f] for f in features]
                axes[1, 0].barh(features, importances)
                axes[1, 0].set_title('Top 10 Feature Importances')
                axes[1, 0].set_xlabel('Importance')
            else:
                axes[1, 0].text(0.5, 0.5, 'Feature Importance\nNot Available', 
                               ha='center', va='center')
                axes[1, 0].set_title('Feature Importance')
            
            # Metrics Summary
            metrics_text = '\n'.join([f'{k}: {v:.4f}' for k, v in result.metrics.items()])
            axes[1, 1].text(0.1, 0.9, metrics_text, transform=axes[1, 1].transAxes, 
                           verticalalignment='top', fontfamily='monospace')
            axes[1, 1].set_title('Metrics Summary')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            # Save plot
            plot_filename = f"regression_plots_{result.model_name}_{result.evaluation_timestamp.strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(self.output_dir / plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to generate regression plots: {e}")
    
    def _generate_comparison_plots(self, comparison: ModelComparison) -> None:
        """Generate model comparison visualization plots."""
        try:
            # Create comparison DataFrame
            df_data = []
            for metric, model_scores in comparison.metrics_comparison.items():
                for model, score in model_scores.items():
                    if not np.isnan(score):
                        df_data.append({'Metric': metric, 'Model': model, 'Score': score})
            
            if not df_data:
                logger.warning("No data available for comparison plots")
                return
            
            df = pd.DataFrame(df_data)
            
            # Create subplots for each metric
            metrics = df['Metric'].unique()
            n_metrics = len(metrics)
            cols = 2
            rows = (n_metrics + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
            if rows == 1:
                axes = [axes] if cols == 1 else axes
            else:
                axes = axes.flatten()
            
            fig.suptitle('Model Comparison', fontsize=16)
            
            for i, metric in enumerate(metrics):
                metric_data = df[df['Metric'] == metric]
                
                ax = axes[i] if len(axes) > 1 else axes
                bars = ax.bar(metric_data['Model'], metric_data['Score'])
                ax.set_title(f'{metric}')
                ax.set_ylabel('Score')
                
                # Highlight best model
                best_model_name = comparison.best_model.get(metric, '')
                for j, bar in enumerate(bars):
                    if metric_data.iloc[j]['Model'] == best_model_name:
                        bar.set_color('green')
                        bar.set_alpha(0.8)
                
                ax.tick_params(axis='x', rotation=45)
            
            # Hide unused subplots
            for i in range(n_metrics, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            # Save plot
            plot_filename = f"model_comparison_{comparison.comparison_timestamp.strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(self.output_dir / plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to generate comparison plots: {e}")


# Global model evaluator instance
model_evaluator = ModelEvaluator()
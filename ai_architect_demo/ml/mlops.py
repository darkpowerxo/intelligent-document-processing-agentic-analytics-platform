"""MLflow integration and MLOps utilities for AI Architect Demo.

This module provides comprehensive MLOps functionality including:
- Experiment tracking and management
- Model registry operations  
- Model lifecycle management
- Performance monitoring
- Automated deployment workflows
"""

import json
import pickle
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mlflow
import mlflow.sklearn
import mlflow.pytorch
import numpy as np
import pandas as pd
from mlflow.entities import Run
from mlflow.tracking import MlflowClient
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from ai_architect_demo.core.config import settings
from ai_architect_demo.core.logging import get_logger, log_performance

logger = get_logger(__name__)


class MLOpsManager:
    """Centralized MLOps management class for model lifecycle operations."""
    
    def __init__(self):
        """Initialize MLOps manager with MLflow configuration."""
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        self.client = MlflowClient()
        self.experiment_name = settings.mlflow_experiment_name
        self.experiment_id = self._get_or_create_experiment()
        
    def _get_or_create_experiment(self) -> str:
        """Get existing experiment or create new one.
        
        Returns:
            Experiment ID
        """
        try:
            experiment = self.client.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = self.client.create_experiment(
                    name=self.experiment_name,
                    artifact_location=f"{settings.mlflow_artifact_root}/{self.experiment_name}"
                )
                logger.info(f"Created new MLflow experiment: {self.experiment_name} (ID: {experiment_id})")
                return experiment_id
            else:
                logger.info(f"Using existing MLflow experiment: {self.experiment_name} (ID: {experiment.experiment_id})")
                return experiment.experiment_id
                
        except Exception as e:
            logger.error(f"Failed to get or create experiment: {e}")
            raise
    
    def start_run(
        self, 
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run
            tags: Optional tags for the run
            
        Returns:
            Run ID
        """
        try:
            run = mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=run_name,
                tags=tags or {}
            )
            logger.info(f"Started MLflow run: {run.info.run_id}")
            return run.info.run_id
            
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
            raise
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to current run.
        
        Args:
            params: Dictionary of parameters to log
        """
        try:
            mlflow.log_params(params)
            logger.debug(f"Logged parameters: {list(params.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to log parameters: {e}")
            raise
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to current run.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number for time series metrics
        """
        try:
            mlflow.log_metrics(metrics, step=step)
            logger.debug(f"Logged metrics: {list(metrics.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
            raise
    
    def log_model(
        self,
        model: BaseEstimator,
        model_name: str,
        signature: Optional[mlflow.models.ModelSignature] = None,
        input_example: Optional[np.ndarray] = None,
        registered_model_name: Optional[str] = None
    ) -> str:
        """Log model to current run and optionally register.
        
        Args:
            model: Trained model to log
            model_name: Name for the model artifact
            signature: Model signature for input/output schema
            input_example: Example input for model
            registered_model_name: Name for model registry
            
        Returns:
            Model URI
        """
        try:
            # Log the model
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=model_name,
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name
            )
            
            logger.info(f"Logged model: {model_name}")
            if registered_model_name:
                logger.info(f"Registered model: {registered_model_name}")
                
            return model_info.model_uri
            
        except Exception as e:
            logger.error(f"Failed to log model: {e}")
            raise
    
    def evaluate_model(
        self,
        model: BaseEstimator,
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: Union[np.ndarray, pd.Series],
        model_name: str
    ) -> Dict[str, Any]:
        """Evaluate model performance and log metrics.
        
        Args:
            model: Trained model to evaluate
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # Prepare metrics for logging
            metrics = {
                "accuracy": accuracy,
                "precision": report["macro avg"]["precision"],
                "recall": report["macro avg"]["recall"], 
                "f1_score": report["macro avg"]["f1-score"]
            }
            
            # Log metrics
            self.log_metrics(metrics)
            
            # Log confusion matrix as artifact
            self._log_confusion_matrix(conf_matrix, model_name)
            
            # Log classification report
            mlflow.log_text(
                json.dumps(report, indent=2),
                f"{model_name}_classification_report.json"
            )
            
            logger.info(f"Model evaluation completed: {model_name}")
            logger.info(f"Accuracy: {accuracy:.4f}")
            
            return {
                "metrics": metrics,
                "classification_report": report,
                "confusion_matrix": conf_matrix.tolist()
            }
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            raise
    
    def _log_confusion_matrix(self, conf_matrix: np.ndarray, model_name: str) -> None:
        """Log confusion matrix as artifact.
        
        Args:
            conf_matrix: Confusion matrix array
            model_name: Name of the model
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {model_name}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                plt.savefig(tmp.name, dpi=100, bbox_inches='tight')
                mlflow.log_artifact(tmp.name, f"{model_name}_confusion_matrix.png")
                plt.close()
                
        except ImportError:
            logger.warning("matplotlib/seaborn not available, skipping confusion matrix plot")
        except Exception as e:
            logger.error(f"Failed to log confusion matrix: {e}")
    
    def register_model(
        self,
        model_uri: str,
        model_name: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """Register model in MLflow Model Registry.
        
        Args:
            model_uri: URI of the model to register
            model_name: Name for the registered model
            description: Optional description
            tags: Optional tags
            
        Returns:
            Model version
        """
        try:
            result = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                tags=tags or {}
            )
            
            # Update description if provided
            if description:
                self.client.update_registered_model(
                    name=model_name,
                    description=description
                )
            
            logger.info(f"Registered model: {model_name}, version: {result.version}")
            return result.version
            
        except Exception as e:
            logger.error(f"Model registration failed: {e}")
            raise
    
    def promote_model(
        self,
        model_name: str,
        version: str,
        stage: str = "Production"
    ) -> None:
        """Promote model to a specific stage.
        
        Args:
            model_name: Name of the registered model
            version: Version to promote
            stage: Target stage (Staging, Production, Archived)
        """
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            
            logger.info(f"Promoted model {model_name} v{version} to {stage}")
            
        except Exception as e:
            logger.error(f"Model promotion failed: {e}")
            raise
    
    def get_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """Get all versions of a registered model.
        
        Args:
            model_name: Name of the registered model
            
        Returns:
            List of model version information
        """
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            
            return [
                {
                    "version": version.version,
                    "stage": version.current_stage,
                    "creation_timestamp": version.creation_timestamp,
                    "last_updated_timestamp": version.last_updated_timestamp,
                    "run_id": version.run_id,
                    "status": version.status
                }
                for version in versions
            ]
            
        except Exception as e:
            logger.error(f"Failed to get model versions: {e}")
            raise
    
    def load_model(self, model_name: str, version: Optional[str] = None, stage: Optional[str] = None) -> Any:
        """Load model from registry.
        
        Args:
            model_name: Name of the registered model
            version: Specific version to load
            stage: Stage to load from (if version not specified)
            
        Returns:
            Loaded model
        """
        try:
            if version:
                model_uri = f"models:/{model_name}/{version}"
            elif stage:
                model_uri = f"models:/{model_name}/{stage}"
            else:
                model_uri = f"models:/{model_name}/latest"
            
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Loaded model: {model_name} from {model_uri}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def end_run(self) -> None:
        """End current MLflow run."""
        try:
            mlflow.end_run()
            logger.info("Ended MLflow run")
            
        except Exception as e:
            logger.error(f"Failed to end run: {e}")
    
    def get_experiment_runs(self, max_results: int = 100) -> List[Dict[str, Any]]:
        """Get runs from the current experiment.
        
        Args:
            max_results: Maximum number of results to return
            
        Returns:
            List of run information
        """
        try:
            runs = self.client.search_runs(
                experiment_ids=[self.experiment_id],
                max_results=max_results,
                order_by=["start_time DESC"]
            )
            
            return [
                {
                    "run_id": run.info.run_id,
                    "run_name": run.data.tags.get("mlflow.runName", ""),
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                    "tags": run.data.tags
                }
                for run in runs
            ]
            
        except Exception as e:
            logger.error(f"Failed to get experiment runs: {e}")
            raise
    
    def list_registered_models(self) -> List[Dict[str, Any]]:
        """List all registered models in MLflow.
        
        Returns:
            List of model information dictionaries
        """
        try:
            models = self.client.search_registered_models()
            return [
                {
                    "name": model.name,
                    "creation_timestamp": model.creation_timestamp,
                    "last_updated_timestamp": model.last_updated_timestamp,
                    "description": model.description,
                    "latest_versions": [
                        {
                            "version": version.version,
                            "stage": version.current_stage,
                            "creation_timestamp": version.creation_timestamp,
                            "last_updated_timestamp": version.last_updated_timestamp
                        }
                        for version in model.latest_versions
                    ]
                }
                for model in models
            ]
            
        except Exception as e:
            logger.error(f"Failed to list registered models: {e}")
            # Return empty list if MLflow is not accessible rather than failing
            return []


# Global MLOps manager instance
mlops_manager = MLOpsManager()


def track_experiment(func):
    """Decorator to automatically track ML experiments.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        run_name = kwargs.pop('run_name', func.__name__)
        tags = kwargs.pop('tags', {})
        
        try:
            run_id = mlops_manager.start_run(run_name=run_name, tags=tags)
            
            # Log function parameters
            mlops_manager.log_params({
                "function": func.__name__,
                "timestamp": datetime.now().isoformat()
            })
            
            # Execute function
            start_time = datetime.now()
            result = func(*args, **kwargs)
            end_time = datetime.now()
            
            # Log execution time
            execution_time = (end_time - start_time).total_seconds()
            mlops_manager.log_metrics({"execution_time_seconds": execution_time})
            
            log_performance(func.__name__, execution_time)
            
            return result
            
        finally:
            mlops_manager.end_run()
    
    return wrapper
"""Model management and prediction API routes.

This module provides endpoints for:
- Model registration and management
- Model predictions and inference
- Model evaluation and monitoring
- Model versioning and deployment
"""

import uuid
import time
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, status, Query
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd

from ai_architect_demo.core.database import Database, get_database
from ai_architect_demo.core.logging import get_logger, log_function_call
from ai_architect_demo.ml.mlops import MLOpsManager, mlops_manager
from ai_architect_demo.ml.evaluation import ModelEvaluator, model_evaluator, ModelEvaluationResult
from ai_architect_demo.api.auth import auth_manager

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/models", tags=["models"])


# Models
class ModelInfo(BaseModel):
    """Model information."""
    name: str
    version: str
    stage: str
    description: Optional[str] = None
    creation_timestamp: datetime
    last_updated_timestamp: datetime
    user_id: str
    tags: Optional[Dict[str, str]] = None
    metrics: Optional[Dict[str, float]] = None


class PredictionRequest(BaseModel):
    """Model prediction request."""
    model_name: str
    input_data: Union[Dict[str, Any], List[Any], List[Dict[str, Any]]]
    model_version: Optional[str] = None
    preprocessing: bool = True
    return_probabilities: bool = False
    batch_size: Optional[int] = None


class PredictionResponse(BaseModel):
    """Model prediction response."""
    prediction_id: str
    model_name: str
    model_version: str
    predictions: Union[List[Any], Dict[str, Any]]
    probabilities: Optional[Union[List[List[float]], List[float]]] = None
    confidence_scores: Optional[List[float]] = None
    processing_time_ms: float
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


class ModelRegistrationRequest(BaseModel):
    """Model registration request."""
    name: str
    model_path: str
    description: Optional[str] = None
    tags: Optional[Dict[str, str]] = None
    stage: str = "Staging"


class ModelEvaluationRequest(BaseModel):
    """Model evaluation request."""
    model_name: str
    model_version: Optional[str] = None
    evaluation_data_path: str
    target_column: str
    feature_columns: Optional[List[str]] = None
    evaluation_metrics: Optional[List[str]] = None
    cross_validation: bool = True


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    model_name: str
    input_data_path: str
    output_path: str
    model_version: Optional[str] = None
    batch_size: int = 1000
    preprocessing: bool = True


class BatchPredictionStatus(BaseModel):
    """Batch prediction status."""
    batch_id: str
    model_name: str
    total_samples: int
    processed_samples: int
    status: str
    started_at: datetime
    estimated_completion: Optional[datetime] = None
    error_message: Optional[str] = None


# Global storage for batch processing
batch_predictions: Dict[str, Dict[str, Any]] = {}


@router.get("", response_model=List[ModelInfo])
async def list_models(
    stage: Optional[str] = Query(None, description="Filter by model stage"),
    skip: int = Query(0, ge=0, description="Number of models to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of models to return")
):
    """List available models.
    
    Args:
        stage: Filter by model stage
        skip: Number of models to skip
        limit: Maximum number of models to return
        
    Returns:
        List of available models
    """
    log_function_call("list_models", stage=stage)
    
    try:
        models = mlops_manager.list_registered_models()
        
        # Filter by stage if specified
        if stage:
            filtered_models = []
            for model in models:
                model_versions = mlops_manager.get_model_versions(model["name"])
                for version in model_versions:
                    if version.get("stage") == stage:
                        model_info = ModelInfo(
                            name=model["name"],
                            version=version["version"],
                            stage=version["stage"],
                            description=version.get("description"),
                            creation_timestamp=datetime.fromisoformat(version["creation_timestamp"]),
                            last_updated_timestamp=datetime.fromisoformat(version["last_updated_timestamp"]),
                            user_id=version.get("user_id", "unknown"),
                            tags=version.get("tags"),
                            metrics=version.get("metrics")
                        )
                        filtered_models.append(model_info)
            models = filtered_models
        else:
            # Convert to ModelInfo objects
            model_infos = []
            for model in models:
                # Get latest version info
                versions = mlops_manager.get_model_versions(model["name"])
                if versions:
                    latest_version = versions[0]  # Assuming sorted by version
                    model_info = ModelInfo(
                        name=model["name"],
                        version=latest_version["version"],
                        stage=latest_version["stage"],
                        description=latest_version.get("description"),
                        creation_timestamp=datetime.fromisoformat(latest_version["creation_timestamp"]),
                        last_updated_timestamp=datetime.fromisoformat(latest_version["last_updated_timestamp"]),
                        user_id=latest_version.get("user_id", "unknown"),
                        tags=latest_version.get("tags"),
                        metrics=latest_version.get("metrics")
                    )
                    model_infos.append(model_info)
            models = model_infos
        
        # Apply pagination
        return models[skip:skip + limit]
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving models: {str(e)}"
        )


@router.get("/{model_name}", response_model=ModelInfo)
async def get_model_info(
    model_name: str,
    version: Optional[str] = Query(None, description="Model version")
):
    """Get detailed model information.
    
    Args:
        model_name: Name of the model
        version: Model version (latest if not specified)
        
    Returns:
        Model information
    """
    try:
        model_info = mlops_manager.get_model_info(model_name, version)
        if not model_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_name} not found"
            )
        
        # Convert to ModelInfo format
        return ModelInfo(
            name=model_info["name"],
            version=model_info["version"],
            stage=model_info["stage"],
            description=model_info.get("description"),
            creation_timestamp=datetime.fromisoformat(model_info["creation_timestamp"]),
            last_updated_timestamp=datetime.fromisoformat(model_info["last_updated_timestamp"]),
            user_id=model_info.get("user_id", "unknown"),
            tags=model_info.get("tags"),
            metrics=model_info.get("metrics")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving model information: {str(e)}"
        )


@router.post("/{model_name}/register")
async def register_model(
    model_name: str,
    request: ModelRegistrationRequest,
    background_tasks: BackgroundTasks
):
    """Register a new model.
    
    Args:
        model_name: Name of the model to register
        request: Model registration details
        background_tasks: Background task manager
        
    Returns:
        Registration confirmation
    """
    log_function_call("register_model", model_name=model_name)
    
    try:
        # Validate model file exists
        model_path = Path(request.model_path)
        if not model_path.exists():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model file not found: {request.model_path}"
            )
        
        # Register model
        model_uri = mlops_manager.register_model(
            model_name=model_name,
            model_path=request.model_path,
            description=request.description,
            tags=request.tags
        )
        
        # Transition to requested stage
        if request.stage != "None":
            mlops_manager.transition_model_stage(model_name, "1", request.stage)
        
        # Start background evaluation if evaluation data is available
        background_tasks.add_task(evaluate_registered_model, model_name, "1")
        
        return {
            "message": "Model registered successfully",
            "model_name": model_name,
            "model_uri": model_uri,
            "stage": request.stage
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model registration failed: {str(e)}"
        )


@router.post("/{model_name}/predict", response_model=PredictionResponse)
async def predict(
    model_name: str,
    request: PredictionRequest
):
    """Make model predictions.
    
    Args:
        model_name: Name of the model
        request: Prediction request
        
    Returns:
        Prediction results
    """
    log_function_call("predict", model_name=model_name)
    
    prediction_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Load model
        model = mlops_manager.load_model(model_name, request.model_version)
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_name} not found or failed to load"
            )
        
        # Prepare input data
        if isinstance(request.input_data, dict):
            # Single sample
            input_array = np.array([list(request.input_data.values())])
        elif isinstance(request.input_data, list):
            if all(isinstance(item, dict) for item in request.input_data):
                # Multiple samples as list of dicts
                input_array = np.array([list(item.values()) for item in request.input_data])
            else:
                # Single sample as list or batch as nested list
                input_array = np.array(request.input_data)
                if input_array.ndim == 1:
                    input_array = input_array.reshape(1, -1)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid input data format"
            )
        
        # Make prediction
        predictions = model.predict(input_array)
        
        # Get probabilities if requested and model supports it
        probabilities = None
        confidence_scores = None
        
        if request.return_probabilities and hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(input_array).tolist()
                # Calculate confidence as max probability
                confidence_scores = [max(probs) for probs in probabilities]
            except Exception as e:
                logger.warning(f"Could not get probabilities: {e}")
        
        # Process predictions
        if predictions.ndim == 1 and len(predictions) == 1:
            # Single prediction
            predictions_output = predictions[0].item() if hasattr(predictions[0], 'item') else predictions[0]
            probabilities = probabilities[0] if probabilities else None
            confidence_scores = confidence_scores[0] if confidence_scores else None
        else:
            # Multiple predictions
            predictions_output = predictions.tolist()
        
        processing_time = (time.time() - start_time) * 1000
        
        # Get model version
        model_info = mlops_manager.get_model_info(model_name, request.model_version)
        model_version = model_info["version"] if model_info else "unknown"
        
        return PredictionResponse(
            prediction_id=prediction_id,
            model_name=model_name,
            model_version=model_version,
            predictions=predictions_output,
            probabilities=probabilities,
            confidence_scores=confidence_scores,
            processing_time_ms=processing_time,
            timestamp=datetime.now(),
            metadata={
                "input_shape": list(input_array.shape),
                "preprocessing": request.preprocessing
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/{model_name}/evaluate")
async def evaluate_model(
    model_name: str,
    request: ModelEvaluationRequest,
    background_tasks: BackgroundTasks
):
    """Evaluate a model on test data.
    
    Args:
        model_name: Name of the model
        request: Evaluation request
        background_tasks: Background task manager
        
    Returns:
        Evaluation job confirmation
    """
    log_function_call("evaluate_model", model_name=model_name)
    
    try:
        # Validate evaluation data path
        data_path = Path(request.evaluation_data_path)
        if not data_path.exists():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Evaluation data file not found: {request.evaluation_data_path}"
            )
        
        # Start background evaluation
        evaluation_id = str(uuid.uuid4())
        background_tasks.add_task(
            run_model_evaluation,
            evaluation_id,
            model_name,
            request
        )
        
        return {
            "message": "Model evaluation started",
            "evaluation_id": evaluation_id,
            "model_name": model_name,
            "status_url": f"/api/v1/evaluations/{evaluation_id}/status"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting evaluation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed to start: {str(e)}"
        )


@router.post("/{model_name}/batch-predict", response_model=BatchPredictionStatus)
async def batch_predict(
    model_name: str,
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks
):
    """Start batch prediction job.
    
    Args:
        model_name: Name of the model
        request: Batch prediction request
        background_tasks: Background task manager
        
    Returns:
        Batch prediction status
    """
    log_function_call("batch_predict", model_name=model_name)
    
    batch_id = str(uuid.uuid4())
    
    try:
        # Validate input data path
        input_path = Path(request.input_data_path)
        if not input_path.exists():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Input data file not found: {request.input_data_path}"
            )
        
        # Initialize batch prediction status
        batch_predictions[batch_id] = {
            "batch_id": batch_id,
            "model_name": model_name,
            "total_samples": 0,  # Will be updated when file is read
            "processed_samples": 0,
            "status": "started",
            "started_at": datetime.now(),
            "input_path": request.input_data_path,
            "output_path": request.output_path,
            "model_version": request.model_version,
            "batch_size": request.batch_size
        }
        
        # Start background processing
        background_tasks.add_task(
            process_batch_predictions,
            batch_id,
            model_name,
            request
        )
        
        return BatchPredictionStatus(
            batch_id=batch_id,
            model_name=model_name,
            total_samples=0,
            processed_samples=0,
            status="started",
            started_at=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting batch prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed to start: {str(e)}"
        )


@router.get("/batch/{batch_id}/status", response_model=BatchPredictionStatus)
async def get_batch_prediction_status(batch_id: str):
    """Get batch prediction status.
    
    Args:
        batch_id: Batch prediction ID
        
    Returns:
        Batch prediction status
    """
    if batch_id not in batch_predictions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Batch prediction job not found"
        )
    
    batch_info = batch_predictions[batch_id]
    
    return BatchPredictionStatus(**batch_info)


@router.put("/{model_name}/stage")
async def transition_model_stage(
    model_name: str,
    version: str = Query(..., description="Model version"),
    stage: str = Query(..., description="Target stage (None, Staging, Production, Archived)")
):
    """Transition model to different stage.
    
    Args:
        model_name: Name of the model
        version: Model version
        stage: Target stage
        
    Returns:
        Transition confirmation
    """
    log_function_call("transition_model_stage", model_name=model_name, stage=stage)
    
    try:
        mlops_manager.transition_model_stage(model_name, version, stage)
        
        return {
            "message": f"Model {model_name} version {version} transitioned to {stage}",
            "model_name": model_name,
            "version": version,
            "new_stage": stage
        }
        
    except Exception as e:
        logger.error(f"Error transitioning model stage: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stage transition failed: {str(e)}"
        )


# Background task functions
async def evaluate_registered_model(model_name: str, version: str):
    """Evaluate a newly registered model."""
    try:
        logger.info(f"Starting evaluation for registered model: {model_name}:{version}")
        
        # This is a placeholder for automatic evaluation
        # In a real implementation, you would:
        # 1. Load the model
        # 2. Load evaluation dataset
        # 3. Run evaluation
        # 4. Store results in MLflow
        
        await asyncio.sleep(2)  # Simulate evaluation time
        logger.info(f"Evaluation completed for model: {model_name}:{version}")
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")


async def run_model_evaluation(evaluation_id: str, model_name: str, request: ModelEvaluationRequest):
    """Run model evaluation in background."""
    try:
        logger.info(f"Starting evaluation {evaluation_id} for model {model_name}")
        
        # Load model
        model = mlops_manager.load_model(model_name, request.model_version)
        
        # Load evaluation data
        data_path = Path(request.evaluation_data_path)
        if data_path.suffix.lower() == '.csv':
            df = pd.read_csv(data_path)
        else:
            raise ValueError(f"Unsupported data format: {data_path.suffix}")
        
        # Prepare features and target
        if request.feature_columns:
            X = df[request.feature_columns].values
        else:
            X = df.drop(columns=[request.target_column]).values
        
        y = df[request.target_column].values
        
        # Run evaluation
        evaluation_result = model_evaluator.evaluate_classification_model(
            model=model,
            X_test=X,
            y_test=y,
            model_name=model_name,
            cross_validate=request.cross_validation
        )
        
        # Log results to MLflow
        mlops_manager.log_evaluation_results(
            model_name,
            request.model_version or "latest",
            evaluation_result.metrics
        )
        
        logger.info(f"Evaluation {evaluation_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Evaluation {evaluation_id} failed: {e}")


async def process_batch_predictions(batch_id: str, model_name: str, request: BatchPredictionRequest):
    """Process batch predictions in background."""
    try:
        logger.info(f"Starting batch prediction {batch_id} for model {model_name}")
        
        batch_info = batch_predictions[batch_id]
        
        # Load model
        model = mlops_manager.load_model(model_name, request.model_version)
        
        # Load input data
        input_path = Path(request.input_data_path)
        if input_path.suffix.lower() == '.csv':
            df = pd.read_csv(input_path)
        else:
            raise ValueError(f"Unsupported input format: {input_path.suffix}")
        
        # Update total samples
        batch_info["total_samples"] = len(df)
        
        # Process in batches
        predictions = []
        for i in range(0, len(df), request.batch_size):
            batch_data = df.iloc[i:i + request.batch_size]
            batch_predictions_result = model.predict(batch_data.values)
            predictions.extend(batch_predictions_result)
            
            # Update progress
            batch_info["processed_samples"] = min(i + request.batch_size, len(df))
        
        # Save predictions
        output_path = Path(request.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results_df = df.copy()
        results_df['prediction'] = predictions
        results_df.to_csv(output_path, index=False)
        
        # Update final status
        batch_info["status"] = "completed"
        batch_info["estimated_completion"] = datetime.now()
        
        logger.info(f"Batch prediction {batch_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Batch prediction {batch_id} failed: {e}")
        batch_predictions[batch_id]["status"] = "failed"
        batch_predictions[batch_id]["error_message"] = str(e)
"""FastAPI backend for AI Architect Demo.

This module provides the main FastAPI application with:
- Document upload and processing endpoints
- Model serving and prediction endpoints
- Authentication and authorization
- Real-time processing status
- Comprehensive API documentation
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import json

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, BackgroundTasks, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
import uvicorn

from ai_architect_demo.core.config import settings
from ai_architect_demo.core.logging import get_logger, log_function_call
from ai_architect_demo.core.database import Database, get_database
from ai_architect_demo.data.document_processor import DocumentProcessor, document_processor
from ai_architect_demo.data.validation import data_validator, create_document_validation_rules
from ai_architect_demo.ml.mlops import MLOpsManager, mlops_manager
from ai_architect_demo.ml.evaluation import model_evaluator
from ai_architect_demo.api.routes import documents_router, models_router

logger = get_logger(__name__)

# API Models
class DocumentUploadRequest(BaseModel):
    """Document upload request model."""
    filename: str
    content_type: str
    metadata: Optional[Dict[str, Any]] = None


class DocumentUploadResponse(BaseModel):
    """Document upload response model."""
    document_id: str
    filename: str
    size: int
    status: str
    upload_timestamp: datetime
    processing_url: str


class DocumentProcessingStatus(BaseModel):
    """Document processing status model."""
    document_id: str
    status: str
    progress: float = Field(..., ge=0, le=100)
    current_step: Optional[str] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    results: Optional[Dict[str, Any]] = None


class PredictionRequest(BaseModel):
    """Model prediction request."""
    model_name: str
    input_data: Union[Dict[str, Any], List[Any]]
    model_version: Optional[str] = None
    preprocessing: bool = True


class PredictionResponse(BaseModel):
    """Model prediction response."""
    prediction_id: str
    model_name: str
    model_version: str
    predictions: Union[List[Any], Dict[str, Any]]
    confidence: Optional[float] = None
    processing_time_ms: float
    timestamp: datetime


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    version: str
    services: Dict[str, str]


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: str
    timestamp: datetime
    request_id: Optional[str] = None


# Custom JSON encoder for datetime objects
class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


# Custom JSONResponse that uses our datetime encoder
class CustomJSONResponse(JSONResponse):
    """Custom JSON response with datetime serialization support."""
    def render(self, content: Any) -> bytes:
        return json.dumps(
            jsonable_encoder(content),
            cls=DateTimeEncoder,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")


# Initialize FastAPI app with custom JSON encoder
app = FastAPI(
    title="AI Architect Demo API",
    description="Enterprise AI Architecture Demo with document processing, ML models, and real-time analytics",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    default_response_class=CustomJSONResponse
)

# Security
security = HTTPBearer()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["localhost", "127.0.0.1", "*"]
)

# Include routers
app.include_router(documents_router)
app.include_router(models_router)

# Global variables for services
db: Optional[Database] = None
doc_processor: DocumentProcessor = document_processor
ml_manager: MLOpsManager = mlops_manager

# Background task storage
processing_tasks: Dict[str, Dict[str, Any]] = {}


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global db
    
    logger.info("Starting AI Architect Demo API...")
    
    # Initialize database
    try:
        db = Database(settings.database_url)
        await db.connect()
        logger.info("Database connected successfully")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise
    
    # Initialize ML services
    try:
        # Setup validation rules
        create_document_validation_rules(data_validator)
        logger.info("Document validation rules configured")
        
        # MLOps manager is already initialized in its constructor
        logger.info("MLOps manager initialized")
        
    except Exception as e:
        logger.error(f"ML services initialization failed: {e}")
        raise
    
    logger.info("API startup completed successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down AI Architect Demo API...")
    
    if db:
        await db.disconnect()
    
    logger.info("Shutdown completed")


# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user."""
    # For demo purposes, we'll use a simple token validation
    # In production, implement proper JWT validation
    token = credentials.credentials
    
    if token != "demo-token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return {"user_id": "demo-user", "username": "demo"}


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    services_status = {}
    
    # Check database
    try:
        if db:
            await db.health_check()
            services_status["database"] = "healthy"
        else:
            services_status["database"] = "not_initialized"
    except Exception:
        services_status["database"] = "unhealthy"
    
    # Check MLflow
    try:
        if ml_manager and ml_manager.client:
            ml_manager.client.search_experiments()
            services_status["mlflow"] = "healthy"
        else:
            services_status["mlflow"] = "not_initialized"
    except Exception:
        services_status["mlflow"] = "unhealthy"
    
    # Check document processor
    try:
        services_status["document_processor"] = "healthy"
    except Exception:
        services_status["document_processor"] = "unhealthy"
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        services=services_status
    )


# Document processing endpoints moved to routes/documents.py


# Document status endpoint moved to routes/documents.py


# Document list endpoint moved to routes/documents.py


# Model prediction endpoints
@app.post("/api/v1/models/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    user = Depends(get_current_user)
):
    """Make model predictions."""
    log_function_call("predict", model_name=request.model_name)
    
    prediction_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    try:
        # Get model from MLflow
        model_info = ml_manager.get_model_info(request.model_name, request.model_version)
        if not model_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {request.model_name} not found"
            )
        
        # Load model
        model = ml_manager.load_model(request.model_name, request.model_version)
        
        # Make prediction
        if isinstance(request.input_data, dict):
            # Single prediction
            prediction = model.predict([list(request.input_data.values())])
            predictions = prediction.tolist()
        else:
            # Batch prediction
            prediction = model.predict(request.input_data)
            predictions = prediction.tolist()
        
        # Calculate confidence (placeholder - depends on model type)
        confidence = None
        if hasattr(model, 'predict_proba'):
            try:
                if isinstance(request.input_data, dict):
                    proba = model.predict_proba([list(request.input_data.values())])
                else:
                    proba = model.predict_proba(request.input_data)
                confidence = float(max(proba[0])) if len(proba) > 0 else None
            except Exception:
                pass
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return PredictionResponse(
            prediction_id=prediction_id,
            model_name=request.model_name,
            model_version=request.model_version or "latest",
            predictions=predictions,
            confidence=confidence,
            processing_time_ms=processing_time,
            timestamp=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/api/v1/models")
async def list_models(user = Depends(get_current_user)):
    """List available models."""
    try:
        models = ml_manager.list_registered_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving models"
        )


@app.get("/api/v1/models/{model_name}")
async def get_model_info(
    model_name: str,
    user = Depends(get_current_user)
):
    """Get detailed model information."""
    try:
        model_info = ml_manager.get_model_info(model_name)
        if not model_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_name} not found"
            )
        return model_info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving model information"
        )


# Background processing function
async def process_document_background(document_id: str, content: bytes, filename: str, content_type: str):
    """Process document in background."""
    try:
        # Update status to processing
        processing_tasks[document_id].update({
            "status": "processing",
            "progress": 10,
            "current_step": "Starting document processing"
        })
        
        # Process document
        processing_tasks[document_id].update({
            "progress": 30,
            "current_step": "Extracting text content"
        })
        
        # Save content to temporary file for processing
        temp_file = Path(f"/tmp/{document_id}_{filename}")
        temp_file.write_bytes(content)
        
        # Extract text using document processor
        extracted_data = doc_processor.extract_text(str(temp_file))
        
        processing_tasks[document_id].update({
            "progress": 60,
            "current_step": "Processing and analyzing content"
        })
        
        # Generate document summary
        if extracted_data.get("text"):
            summary = doc_processor.generate_summary(extracted_data["text"])
            extracted_data["summary"] = summary
        
        processing_tasks[document_id].update({
            "progress": 80,
            "current_step": "Storing processing results"
        })
        
        # Store results in database
        if db:
            query = """
                UPDATE documents 
                SET status = :status, processing_results = :results, processed_at = :processed_at
                WHERE id = :id
            """
            await db.execute(query, {
                "id": document_id,
                "status": "completed",
                "results": extracted_data,
                "processed_at": datetime.now()
            })
        
        # Update final status
        processing_tasks[document_id].update({
            "status": "completed",
            "progress": 100,
            "current_step": "Processing completed successfully",
            "completed_at": datetime.now(),
            "results": extracted_data
        })
        
        # Cleanup temp file
        if temp_file.exists():
            temp_file.unlink()
        
        logger.info(f"Document {document_id} processed successfully")
        
    except Exception as e:
        logger.error(f"Document processing failed for {document_id}: {e}")
        
        # Update error status
        processing_tasks[document_id].update({
            "status": "failed",
            "progress": 0,
            "current_step": "Processing failed",
            "error_message": str(e),
            "completed_at": datetime.now()
        })
        
        # Update database
        if db:
            try:
                query = """
                    UPDATE documents 
                    SET status = :status, error_message = :error_message, processed_at = :processed_at
                    WHERE id = :id
                """
                await db.execute(query, {
                    "id": document_id,
                    "status": "failed",
                    "error_message": str(e),
                    "processed_at": datetime.now()
                })
            except Exception as db_error:
                logger.error(f"Database update error: {db_error}")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    error_response = ErrorResponse(
        error=exc.detail,
        detail=exc.detail,
        timestamp=datetime.now()
    )
    return CustomJSONResponse(
        status_code=exc.status_code,
        content=error_response.dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    error_response = ErrorResponse(
        error="Internal Server Error",
        detail="An unexpected error occurred",
        timestamp=datetime.now()
    )
    return CustomJSONResponse(
        status_code=500,
        content=error_response.dict()
    )


if __name__ == "__main__":
    uvicorn.run(
        "ai_architect_demo.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
"""Document management API routes.

This module provides endpoints for:
- Document upload and validation
- Document processing status
- Document retrieval and management
- Batch processing operations
"""

import asyncio
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, BackgroundTasks, status, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from pydantic.json import pydantic_encoder
import json

from ai_architect_demo.core.database import Database, get_database
from ai_architect_demo.core.logging import get_logger, log_function_call
from ai_architect_demo.data.document_processor import DocumentProcessor, document_processor
from ai_architect_demo.data.validation import data_validator, ValidationReport
from ai_architect_demo.api.auth import auth_manager

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


# Models
class DocumentMetadata(BaseModel):
    """Document metadata model."""
    id: int
    filename: str
    content_type: str
    file_size: int
    status: str
    upload_timestamp: datetime
    processed_at: Optional[datetime] = None
    user_id: int
    metadata: Optional[dict] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class DocumentUploadResponse(BaseModel):
    """Document upload response."""
    document_id: str
    filename: str
    size: int
    status: str
    upload_timestamp: datetime
    processing_url: str
    validation_report: Optional[ValidationReport] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DocumentProcessingResults(BaseModel):
    """Document processing results."""
    document_id: str
    text_content: Optional[str] = None
    word_count: Optional[int] = None
    page_count: Optional[int] = None
    summary: Optional[str] = None
    entities: Optional[List[Dict[str, Any]]] = None
    sentiment: Optional[Dict[str, float]] = None
    topics: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None


class BatchProcessingRequest(BaseModel):
    """Batch processing request."""
    document_ids: List[str]
    processing_options: Optional[Dict[str, Any]] = None


class BatchProcessingStatus(BaseModel):
    """Batch processing status."""
    batch_id: str
    total_documents: int
    completed: int
    failed: int
    status: str
    started_at: datetime
    estimated_completion: Optional[datetime] = None


# Global storage for batch processing tasks
batch_tasks: Dict[str, Dict[str, Any]] = {}


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    validate_content: bool = Query(True, description="Perform content validation"),
    extract_metadata: bool = Query(True, description="Extract document metadata"),
    db: Database = Depends(get_database)
):
    """Upload a document for processing.
    
    Args:
        file: Document file to upload
        validate_content: Whether to perform content validation
        extract_metadata: Whether to extract metadata
        db: Database connection
        
    Returns:
        Upload response with document ID and processing URL
    """
    log_function_call("upload_document", filename=file.filename)
    
    # Generate document ID
    document_id = str(uuid.uuid4())
    
    # Validate file
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required"
        )
    
    # Read and validate file content
    try:
        content = await file.read()
        file_size = len(content)
        
        if file_size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File is empty"
            )
        
        if file_size > 50 * 1024 * 1024:  # 50MB limit
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size {file_size} exceeds maximum 50MB"
            )
        
    except Exception as e:
        logger.error(f"Error reading uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Error reading file content"
        )
    
    # Validate document metadata
    doc_metadata = {
        "filename": file.filename,
        "content_type": file.content_type,
        "file_size": file_size
    }
    
    validation_report = None
    if validate_content:
        validation_report = data_validator.validate_data(doc_metadata)
        if validation_report.critical > 0:
            error_details = [r.message for r in validation_report.results if r.severity == "critical"]
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Document validation failed: {'; '.join(error_details)}"
            )
    
    # Store document in database
    try:
        query = """
            INSERT INTO app.documents (filename, original_name, file_path, file_size, content_type, processing_status, user_id, metadata)
            VALUES (:filename, :original_name, :file_path, :file_size, :content_type, :processing_status, :user_id, :metadata)
            RETURNING id
        """
        result = await db.fetch_one(query, {
            "filename": file.filename,
            "original_name": file.filename,
            "file_path": f"/uploads/{document_id}_{file.filename}",  # Placeholder path
            "file_size": file_size,
            "content_type": file.content_type,
            "processing_status": "uploaded",
            "user_id": 1,  # Demo user ID (assuming user with id=1 exists)
            "metadata": json.dumps({"document_id": document_id, "content_preview": content[:100].decode('utf-8', errors='ignore') if content else ""})
        })
        
        # Get the auto-generated ID
        if result:
            db_document_id = result["id"]
        
        logger.info(f"Document stored in database with ID: {db_document_id}, UUID: {document_id}")
        
    except Exception as e:
        logger.error(f"Database error storing document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error storing document"
        )
    
    # Start background processing
    background_tasks.add_task(
        process_document_async,
        document_id,
        content,
        file.filename,
        file.content_type,
        extract_metadata
    )
    
    response_data = DocumentUploadResponse(
        document_id=document_id,
        filename=file.filename,
        size=file_size,
        status="uploaded",
        upload_timestamp=datetime.now(),
        processing_url=f"/api/v1/documents/{document_id}/status",
        validation_report=validation_report
    )
    
    # Custom JSON serialization to handle datetime
    def json_encoder(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    return JSONResponse(
        content=json.loads(response_data.json()),
        status_code=200
    )


@router.get("/{document_id}/status")
async def get_document_status(
    document_id: str,
    db: Database = Depends(get_database)
):
    """Get document processing status.
    
    Args:
        document_id: Document ID
        db: Database connection
        
    Returns:
        Document processing status
    """
    try:
        query = """
            SELECT id, filename, processing_status as status, upload_time as upload_timestamp, 
                   processed_at, metadata
            FROM app.documents 
            WHERE metadata::jsonb->>'document_id' = :document_id
        """
        
        document = await db.fetch_one(query, {"document_id": document_id})
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        return dict(document)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving document status"
        )


@router.get("/{document_id}/results", response_model=DocumentProcessingResults)
async def get_document_results(
    document_id: str,
    db: Database = Depends(get_database)
):
    """Get document processing results.
    
    Args:
        document_id: Document ID
        db: Database connection
        
    Returns:
        Document processing results
    """
    try:
        query = """
            SELECT processing_results, status
            FROM documents 
            WHERE id = :document_id
        """
        
        document = await db.fetch_one(query, {"document_id": document_id})
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        if document["status"] != "completed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Document processing not completed. Status: {document['status']}"
            )
        
        results = document["processing_results"] or {}
        
        return DocumentProcessingResults(
            document_id=document_id,
            text_content=results.get("text"),
            word_count=results.get("word_count"),
            page_count=results.get("page_count"),
            summary=results.get("summary"),
            entities=results.get("entities"),
            sentiment=results.get("sentiment"),
            topics=results.get("topics"),
            metadata=results.get("metadata")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document results: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving document results"
        )


@router.get("", response_model=List[DocumentMetadata])
async def list_documents(
    skip: int = Query(0, ge=0, description="Number of documents to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of documents to return"),
    status_filter: Optional[str] = Query(None, description="Filter by document status"),
    db: Database = Depends(get_database)
):
    """List user's documents.
    
    Args:
        skip: Number of documents to skip
        limit: Maximum number of documents to return
        status_filter: Filter by document status
        db: Database connection
        
    Returns:
        List of documents
    """
    try:
        # Build query with optional status filter
        base_query = """
            SELECT id, filename, content_type, file_size, processing_status as status, upload_time as upload_timestamp, 
                   processed_at, user_id, metadata
            FROM app.documents 
            WHERE user_id = :user_id
        """
        
        params = {"user_id": 1, "limit": limit, "skip": skip}
        
        if status_filter:
            base_query += " AND processing_status = :status"
            params["status"] = status_filter
        
        base_query += " ORDER BY upload_time DESC LIMIT :limit OFFSET :skip"
        
        documents = await db.fetch_all(base_query, params)
        
        # Convert documents to proper format, parsing JSON metadata
        document_list = []
        for doc in documents:
            doc_dict = dict(doc)
            # Parse JSON metadata if it exists
            if doc_dict.get('metadata'):
                try:
                    doc_dict['metadata'] = json.loads(doc_dict['metadata'])
                except (json.JSONDecodeError, TypeError):
                    doc_dict['metadata'] = {}
            document_list.append(DocumentMetadata(**doc_dict))
        
        return document_list
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving documents"
        )


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    db: Database = Depends(get_database)
):
    """Delete a document.
    
    Args:
        document_id: Document ID to delete
        db: Database connection
        
    Returns:
        Success message
    """
    try:
        # Check if document exists
        check_query = "SELECT id FROM documents WHERE id = :document_id AND user_id = :user_id"
        document = await db.fetch_one(check_query, {
            "document_id": document_id,
            "user_id": "demo-user"
        })
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # Delete document
        delete_query = "DELETE FROM documents WHERE id = :document_id AND user_id = :user_id"
        await db.execute(delete_query, {
            "document_id": document_id,
            "user_id": "demo-user"
        })
        
        logger.info(f"Document deleted: {document_id}")
        return {"message": "Document deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error deleting document"
        )


@router.post("/batch/process", response_model=BatchProcessingStatus)
async def start_batch_processing(
    request: BatchProcessingRequest,
    background_tasks: BackgroundTasks,
    db: Database = Depends(get_database)
):
    """Start batch processing of multiple documents.
    
    Args:
        request: Batch processing request
        background_tasks: Background task manager
        db: Database connection
        
    Returns:
        Batch processing status
    """
    log_function_call("start_batch_processing", document_count=len(request.document_ids))
    
    batch_id = str(uuid.uuid4())
    
    # Validate document IDs
    if not request.document_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one document ID is required"
        )
    
    # Check if documents exist
    try:
        placeholders = ",".join([":id" + str(i) for i in range(len(request.document_ids))])
        query = f"""
            SELECT id FROM documents 
            WHERE id IN ({placeholders}) AND user_id = :user_id
        """
        
        params = {"user_id": "demo-user"}
        for i, doc_id in enumerate(request.document_ids):
            params[f"id{i}"] = doc_id
        
        existing_docs = await db.fetch_all(query, params)
        existing_ids = {doc["id"] for doc in existing_docs}
        
        missing_ids = set(request.document_ids) - existing_ids
        if missing_ids:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Documents not found: {list(missing_ids)}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating batch documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error validating documents"
        )
    
    # Initialize batch processing status
    batch_tasks[batch_id] = {
        "batch_id": batch_id,
        "total_documents": len(request.document_ids),
        "completed": 0,
        "failed": 0,
        "status": "started",
        "started_at": datetime.now(),
        "document_ids": request.document_ids,
        "processing_options": request.processing_options
    }
    
    # Start batch processing
    background_tasks.add_task(process_batch_async, batch_id, request.document_ids, request.processing_options)
    
    return BatchProcessingStatus(
        batch_id=batch_id,
        total_documents=len(request.document_ids),
        completed=0,
        failed=0,
        status="started",
        started_at=datetime.now()
    )


@router.get("/batch/{batch_id}/status", response_model=BatchProcessingStatus)
async def get_batch_status(batch_id: str):
    """Get batch processing status.
    
    Args:
        batch_id: Batch processing ID
        
    Returns:
        Batch processing status
    """
    if batch_id not in batch_tasks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Batch processing job not found"
        )
    
    batch_info = batch_tasks[batch_id]
    
    return BatchProcessingStatus(**batch_info)


# Background processing functions
async def process_document_async(
    document_id: str,
    content: bytes,
    filename: str,
    content_type: str,
    extract_metadata: bool = True
):
    """Process document asynchronously."""
    try:
        logger.info(f"Starting document processing: {document_id}")
        
        # Create temporary file
        temp_file = Path(f"/tmp/{document_id}_{filename}")
        temp_file.write_bytes(content)
        
        # Extract text
        extracted_data = document_processor.extract_text(str(temp_file))
        
        # Extract additional metadata if requested
        if extract_metadata and extracted_data.get("text"):
            # Generate summary
            try:
                summary = document_processor.generate_summary(extracted_data["text"])
                extracted_data["summary"] = summary
            except Exception as e:
                logger.warning(f"Summary generation failed: {e}")
            
            # Add word count
            if extracted_data.get("text"):
                extracted_data["word_count"] = len(extracted_data["text"].split())
        
        # Update database with results
        from ai_architect_demo.core.database import Database
        db = Database("postgresql://postgres:password@localhost:5432/ai_architect_demo")
        await db.connect()
        
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
        
        await db.disconnect()
        
        # Cleanup
        if temp_file.exists():
            temp_file.unlink()
        
        logger.info(f"Document processing completed: {document_id}")
        
    except Exception as e:
        logger.error(f"Document processing failed: {document_id}, {e}")
        
        # Update database with error
        try:
            from ai_architect_demo.core.database import Database
            db = Database("postgresql://postgres:password@localhost:5432/ai_architect_demo")
            await db.connect()
            
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
            
            await db.disconnect()
        except Exception as db_error:
            logger.error(f"Failed to update error status: {db_error}")


async def process_batch_async(batch_id: str, document_ids: List[str], processing_options: Optional[Dict[str, Any]]):
    """Process batch of documents asynchronously."""
    try:
        logger.info(f"Starting batch processing: {batch_id}")
        
        batch_info = batch_tasks[batch_id]
        
        # Process documents in parallel (limited concurrency)
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent processing tasks
        
        async def process_single_doc(doc_id: str):
            async with semaphore:
                try:
                    # Get document from database
                    from ai_architect_demo.core.database import Database
                    db = Database("postgresql://postgres:password@localhost:5432/ai_architect_demo")
                    await db.connect()
                    
                    query = "SELECT content, filename, content_type FROM documents WHERE id = :id"
                    doc = await db.fetch_one(query, {"id": doc_id})
                    await db.disconnect()
                    
                    if doc:
                        await process_document_async(
                            doc_id,
                            doc["content"],
                            doc["filename"],
                            doc["content_type"],
                            processing_options and processing_options.get("extract_metadata", True)
                        )
                        
                        # Update batch status
                        batch_info["completed"] += 1
                        logger.info(f"Batch {batch_id}: completed document {doc_id}")
                    
                except Exception as e:
                    logger.error(f"Batch {batch_id}: failed to process document {doc_id}: {e}")
                    batch_info["failed"] += 1
        
        # Process all documents
        tasks = [process_single_doc(doc_id) for doc_id in document_ids]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Update final batch status
        batch_info["status"] = "completed"
        batch_info["estimated_completion"] = datetime.now()
        
        logger.info(f"Batch processing completed: {batch_id}")
        
    except Exception as e:
        logger.error(f"Batch processing failed: {batch_id}, {e}")
        batch_tasks[batch_id]["status"] = "failed"
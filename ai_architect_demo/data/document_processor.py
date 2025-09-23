"""Document processing utilities for AI Architect Demo.

This module provides comprehensive document processing capabilities including:
- File validation and type detection
- Text extraction from various formats
- Document parsing and preprocessing
- Metadata extraction
- Error handling and logging
"""

import hashlib
import mimetypes
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from PIL import Image
import pytesseract
from docx import Document as DocxDocument

from ai_architect_demo.core.config import settings
from ai_architect_demo.core.logging import get_logger, log_function_call, log_performance

logger = get_logger(__name__)


class DocumentProcessor:
    """Enterprise-grade document processing with comprehensive error handling."""
    
    def __init__(self):
        """Initialize document processor with supported file types."""
        self.supported_types = settings.allowed_file_types
        self.max_file_size = settings.max_file_size
        
        # Content type mappings
        self.content_type_map = {
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.txt': 'text/plain',
            '.md': 'text/markdown',
            '.csv': 'text/csv',
            '.json': 'application/json',
            '.xml': 'application/xml'
        }
        
    def validate_file(self, file_path: Path, content: Optional[bytes] = None) -> Tuple[bool, str]:
        """Validate file type, size, and content.
        
        Args:
            file_path: Path to the file
            content: Optional file content bytes
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        log_function_call("validate_file", file_path=str(file_path))
        
        try:
            # Check if file exists
            if not file_path.exists() and content is None:
                return False, "File does not exist"
            
            # Check file extension
            file_extension = file_path.suffix.lower()
            if file_extension not in self.supported_types:
                return False, f"Unsupported file type: {file_extension}"
            
            # Check file size
            if file_path.exists():
                file_size = file_path.stat().st_size
            else:
                file_size = len(content) if content else 0
                
            if file_size > self.max_file_size:
                max_size_mb = self.max_file_size / (1024 * 1024)
                return False, f"File too large. Max size: {max_size_mb:.1f}MB"
            
            if file_size == 0:
                return False, "File is empty"
            
            logger.info(f"File validation successful: {file_path.name}")
            return True, ""
            
        except Exception as e:
            logger.error(f"File validation error: {e}")
            return False, f"Validation error: {str(e)}"
    
    def extract_text(self, file_path: Path) -> Dict[str, Any]:
        """Extract text content from document.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        log_function_call("extract_text", file_path=str(file_path))
        
        try:
            file_extension = file_path.suffix.lower()
            
            # Route to appropriate extraction method
            if file_extension == '.pdf':
                return self._extract_from_pdf(file_path)
            elif file_extension == '.docx':
                return self._extract_from_docx(file_path)
            elif file_extension in ['.txt', '.md']:
                return self._extract_from_text(file_path)
            elif file_extension == '.csv':
                return self._extract_from_csv(file_path)
            else:
                raise ValueError(f"Unsupported file type for text extraction: {file_extension}")
                
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return {
                "text": "",
                "error": str(e),
                "metadata": self._get_file_metadata(file_path)
            }
    
    def _extract_from_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Extract text from PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extraction result dictionary
        """
        try:
            import PyPDF2
            
            text_content = []
            metadata = {}
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract metadata
                if pdf_reader.metadata:
                    metadata.update({
                        'title': str(pdf_reader.metadata.get('/Title', '')),
                        'author': str(pdf_reader.metadata.get('/Author', '')),
                        'creator': str(pdf_reader.metadata.get('/Creator', '')),
                        'pages': len(pdf_reader.pages)
                    })
                
                # Extract text from all pages
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_content.append(f"--- Page {page_num + 1} ---\n{page_text}")
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
            
            full_text = "\n\n".join(text_content)
            
            return {
                "text": full_text,
                "word_count": len(full_text.split()),
                "char_count": len(full_text),
                "pages": len(pdf_reader.pages),
                "metadata": {**metadata, **self._get_file_metadata(file_path)}
            }
            
        except ImportError:
            logger.error("PyPDF2 not installed. Cannot process PDF files.")
            raise
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise
    
    def _extract_from_docx(self, file_path: Path) -> Dict[str, Any]:
        """Extract text from DOCX file.
        
        Args:
            file_path: Path to DOCX file
            
        Returns:
            Extraction result dictionary
        """
        try:
            doc = DocxDocument(file_path)
            
            # Extract paragraphs
            paragraphs = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    paragraphs.append(paragraph.text)
            
            full_text = "\n\n".join(paragraphs)
            
            # Extract metadata
            core_props = doc.core_properties
            metadata = {
                'title': core_props.title or '',
                'author': core_props.author or '',
                'created': core_props.created.isoformat() if core_props.created else None,
                'modified': core_props.modified.isoformat() if core_props.modified else None,
                'paragraphs': len(paragraphs)
            }
            
            return {
                "text": full_text,
                "word_count": len(full_text.split()),
                "char_count": len(full_text),
                "paragraphs": len(paragraphs),
                "metadata": {**metadata, **self._get_file_metadata(file_path)}
            }
            
        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            raise
    
    def _extract_from_text(self, file_path: Path) -> Dict[str, Any]:
        """Extract text from plain text or markdown file.
        
        Args:
            file_path: Path to text file
            
        Returns:
            Extraction result dictionary
        """
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        text = file.read()
                        break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Could not decode file with any supported encoding")
            
            lines = text.split('\n')
            non_empty_lines = [line for line in lines if line.strip()]
            
            return {
                "text": text,
                "word_count": len(text.split()),
                "char_count": len(text),
                "lines": len(lines),
                "non_empty_lines": len(non_empty_lines),
                "metadata": self._get_file_metadata(file_path)
            }
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            raise
    
    def _extract_from_csv(self, file_path: Path) -> Dict[str, Any]:
        """Extract text from CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Extraction result dictionary
        """
        try:
            df = pd.read_csv(file_path)
            
            # Convert dataframe to text representation
            text_parts = []
            
            # Add column headers
            text_parts.append("Columns: " + ", ".join(df.columns.tolist()))
            
            # Add sample data (first 5 rows)
            text_parts.append("\nSample Data:")
            text_parts.append(df.head().to_string(index=False))
            
            # Add data summary
            text_parts.append(f"\nData Summary:")
            text_parts.append(f"Rows: {len(df)}")
            text_parts.append(f"Columns: {len(df.columns)}")
            
            full_text = "\n".join(text_parts)
            
            return {
                "text": full_text,
                "word_count": len(full_text.split()),
                "char_count": len(full_text),
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "metadata": self._get_file_metadata(file_path)
            }
            
        except Exception as e:
            logger.error(f"CSV extraction failed: {e}")
            raise
    
    def _get_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract file metadata.
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary of file metadata
        """
        try:
            stat = file_path.stat()
            
            # Generate file hash
            hash_sha256 = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            
            return {
                'filename': file_path.name,
                'file_size': stat.st_size,
                'file_extension': file_path.suffix.lower(),
                'mime_type': mimetypes.guess_type(file_path)[0],
                'created_time': stat.st_ctime,
                'modified_time': stat.st_mtime,
                'file_hash': hash_sha256.hexdigest()
            }
            
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            return {'error': str(e)}
    
    def preprocess_text(self, text: str, options: Optional[Dict[str, bool]] = None) -> str:
        """Preprocess extracted text for analysis.
        
        Args:
            text: Raw text to preprocess
            options: Preprocessing options
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        default_options = {
            'normalize_whitespace': True,
            'remove_empty_lines': True,
            'strip_lines': True,
            'lowercase': False,
            'remove_special_chars': False
        }
        
        options = {**default_options, **(options or {})}
        
        processed_text = text
        
        if options['normalize_whitespace']:
            import re
            processed_text = re.sub(r'\s+', ' ', processed_text)
        
        if options['remove_empty_lines'] or options['strip_lines']:
            lines = processed_text.split('\n')
            if options['strip_lines']:
                lines = [line.strip() for line in lines]
            if options['remove_empty_lines']:
                lines = [line for line in lines if line]
            processed_text = '\n'.join(lines)
        
        if options['lowercase']:
            processed_text = processed_text.lower()
        
        if options['remove_special_chars']:
            import re
            processed_text = re.sub(r'[^\w\s]', ' ', processed_text)
            processed_text = re.sub(r'\s+', ' ', processed_text)
        
        return processed_text.strip()
    
    def get_document_summary(self, file_path: Path) -> Dict[str, Any]:
        """Generate comprehensive document summary.
        
        Args:
            file_path: Path to document
            
        Returns:
            Document summary dictionary
        """
        log_function_call("get_document_summary", file_path=str(file_path))
        
        start_time = pd.Timestamp.now()
        
        try:
            # Validate file
            is_valid, error = self.validate_file(file_path)
            if not is_valid:
                return {"error": error}
            
            # Extract text and metadata
            extraction_result = self.extract_text(file_path)
            
            if "error" in extraction_result:
                return extraction_result
            
            # Generate summary
            summary = {
                "file_info": {
                    "name": file_path.name,
                    "path": str(file_path),
                    "type": file_path.suffix.lower(),
                    "size": file_path.stat().st_size
                },
                "content": {
                    "text_length": len(extraction_result.get("text", "")),
                    "word_count": extraction_result.get("word_count", 0),
                    "char_count": extraction_result.get("char_count", 0)
                },
                "metadata": extraction_result.get("metadata", {}),
                "processing_info": {
                    "processed_at": pd.Timestamp.now().isoformat(),
                    "processing_time_ms": int((pd.Timestamp.now() - start_time).total_seconds() * 1000)
                }
            }
            
            # Add type-specific info
            if "pages" in extraction_result:
                summary["content"]["pages"] = extraction_result["pages"]
            if "paragraphs" in extraction_result:
                summary["content"]["paragraphs"] = extraction_result["paragraphs"]
            if "rows" in extraction_result:
                summary["content"]["rows"] = extraction_result["rows"]
                summary["content"]["columns"] = extraction_result["columns"]
            
            processing_time = (pd.Timestamp.now() - start_time).total_seconds()
            log_performance("get_document_summary", processing_time, 
                          file_type=file_path.suffix, file_size=file_path.stat().st_size)
            
            return summary
            
        except Exception as e:
            logger.error(f"Document summary generation failed: {e}")
            return {"error": str(e)}


# Global document processor instance
document_processor = DocumentProcessor()
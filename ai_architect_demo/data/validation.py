"""Data validation utilities for AI Architect Demo.

This module provides comprehensive data validation capabilities including:
- Schema validation and type checking
- Data quality assessment
- Business rule validation
- Statistical validation
- Error reporting and logging
"""

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum

import pandas as pd
import numpy as np
from pydantic import BaseModel, ValidationError, validator

from ai_architect_demo.core.config import settings
from ai_architect_demo.core.logging import get_logger, log_function_call

logger = get_logger(__name__)


class ValidationSeverity(Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationRule(BaseModel):
    """Validation rule definition."""
    name: str
    field: str
    validator: str
    parameters: Dict[str, Any] = {}
    severity: ValidationSeverity = ValidationSeverity.ERROR
    message: Optional[str] = None
    
    class Config:
        use_enum_values = True


class ValidationResult(BaseModel):
    """Validation result model."""
    field: str
    rule: str
    severity: ValidationSeverity
    message: str
    value: Optional[Any] = None
    expected: Optional[Any] = None
    
    class Config:
        use_enum_values = True


class ValidationReport(BaseModel):
    """Complete validation report."""
    total_checks: int
    passed: int
    failed: int
    warnings: int
    errors: int
    critical: int
    success_rate: float
    results: List[ValidationResult]
    
    @validator('success_rate')
    def calculate_success_rate(cls, v, values):
        if 'total_checks' in values and values['total_checks'] > 0:
            return round(values['passed'] / values['total_checks'] * 100, 2)
        return 0.0


class DataValidator:
    """Enterprise data validation with configurable rules and reporting."""
    
    def __init__(self):
        """Initialize data validator."""
        self.rules = {}
        self.custom_validators = {}
        self._register_built_in_validators()
    
    def _register_built_in_validators(self):
        """Register built-in validation functions."""
        self.custom_validators.update({
            'required': self._validate_required,
            'type': self._validate_type,
            'range': self._validate_range,
            'length': self._validate_length,
            'pattern': self._validate_pattern,
            'email': self._validate_email,
            'url': self._validate_url,
            'date': self._validate_date,
            'numeric': self._validate_numeric,
            'unique': self._validate_unique,
            'not_null': self._validate_not_null,
            'enum': self._validate_enum
        })
    
    def register_validator(self, name: str, func: Callable) -> None:
        """Register a custom validation function.
        
        Args:
            name: Name of the validator
            func: Validation function that returns (bool, str)
        """
        self.custom_validators[name] = func
        logger.info(f"Registered custom validator: {name}")
    
    def add_rule(
        self,
        field: str,
        validator: str,
        params: Optional[Dict[str, Any]] = None,
        severity: ValidationSeverity = ValidationSeverity.ERROR,
        message: Optional[str] = None
    ) -> None:
        """Add a validation rule.
        
        Args:
            field: Field name to validate
            validator: Name of validator function
            params: Parameters for validator
            severity: Severity level
            message: Custom error message
        """
        if field not in self.rules:
            self.rules[field] = []
        
        rule = {
            'validator': validator,
            'params': params or {},
            'severity': severity,
            'message': message or f"{field} failed {validator} validation"
        }
        
        self.rules[field].append(rule)
    
    def validate_data(self, data: Union[Dict[str, Any], pd.DataFrame]) -> ValidationReport:
        """Validate data against configured rules.
        
        Args:
            data: Data to validate
            
        Returns:
            Validation report
        """
        log_function_call("validate_data", data_type=type(data).__name__)
        
        results = []
        
        # Convert DataFrame to dict for validation
        if isinstance(data, pd.DataFrame):
            data_dict = data.to_dict('list')
        else:
            data_dict = data
        
        # Validate each field
        for field, field_rules in self.rules.items():
            field_value = data_dict.get(field)
            
            for rule in field_rules:
                validator_name = rule['validator']
                params = rule['params']
                severity = rule['severity']
                message = rule['message']
                
                if validator_name in self.custom_validators:
                    try:
                        is_valid, error_msg = self.custom_validators[validator_name](
                            field_value, **params
                        )
                        
                        if not is_valid:
                            results.append(ValidationResult(
                                field=field,
                                rule=validator_name,
                                severity=severity,
                                message=error_msg or message,
                                value=field_value
                            ))
                    except Exception as e:
                        logger.error(f"Validation error for {field}.{validator_name}: {e}")
                        results.append(ValidationResult(
                            field=field,
                            rule=validator_name,
                            severity=ValidationSeverity.CRITICAL,
                            message=f"Validator error: {str(e)}",
                            value=field_value
                        ))
                else:
                    logger.warning(f"Unknown validator: {validator_name}")
                    results.append(ValidationResult(
                        field=field,
                        rule=validator_name,
                        severity=ValidationSeverity.CRITICAL,
                        message=f"Unknown validator: {validator_name}",
                        value=field_value
                    ))
        
        # Calculate statistics
        total_checks = len(results) if results else len(self.rules)
        failed_results = [r for r in results if r.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]
        warning_results = [r for r in results if r.severity == ValidationSeverity.WARNING]
        error_results = [r for r in results if r.severity == ValidationSeverity.ERROR]
        critical_results = [r for r in results if r.severity == ValidationSeverity.CRITICAL]
        
        passed = max(0, total_checks - len(results))
        
        success_rate = round(passed / total_checks * 100, 2) if total_checks > 0 else 0.0
        
        return ValidationReport(
            total_checks=total_checks,
            passed=passed,
            failed=len(failed_results),
            warnings=len(warning_results),
            errors=len(error_results),
            critical=len(critical_results),
            success_rate=success_rate,
            results=results
        )
    
    def validate_dataframe(self, df: pd.DataFrame) -> ValidationReport:
        """Validate pandas DataFrame with statistical checks.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Enhanced validation report with statistical information
        """
        # Run standard validation
        report = self.validate_data(df)
        
        # Add statistical validation results
        stats_results = self._validate_dataframe_statistics(df)
        report.results.extend(stats_results)
        
        # Recalculate totals
        report.total_checks += len(stats_results)
        failed_stats = [r for r in stats_results if r.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]
        report.failed += len(failed_stats)
        report.passed = report.total_checks - report.failed
        
        return report
    
    def _validate_dataframe_statistics(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Validate DataFrame statistical properties.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            List of statistical validation results
        """
        results = []
        
        try:
            # Check for empty DataFrame
            if df.empty:
                results.append(ValidationResult(
                    field="dataframe",
                    rule="not_empty",
                    severity=ValidationSeverity.ERROR,
                    message="DataFrame is empty"
                ))
                return results
            
            # Check for missing values
            missing_counts = df.isnull().sum()
            for col, missing_count in missing_counts.items():
                if missing_count > 0:
                    missing_percentage = (missing_count / len(df)) * 100
                    severity = ValidationSeverity.WARNING if missing_percentage < 50 else ValidationSeverity.ERROR
                    
                    results.append(ValidationResult(
                        field=col,
                        rule="missing_values",
                        severity=severity,
                        message=f"Column has {missing_count} missing values ({missing_percentage:.1f}%)",
                        value=missing_count
                    ))
            
            # Check for duplicate rows
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                results.append(ValidationResult(
                    field="dataframe",
                    rule="no_duplicates",
                    severity=ValidationSeverity.WARNING,
                    message=f"Found {duplicate_count} duplicate rows",
                    value=duplicate_count
                ))
            
            # Check data type consistency
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Check for mixed types in object columns
                    types = df[col].dropna().apply(type).unique()
                    if len(types) > 1:
                        results.append(ValidationResult(
                            field=col,
                            rule="consistent_types",
                            severity=ValidationSeverity.WARNING,
                            message=f"Column contains mixed data types: {[t.__name__ for t in types]}",
                            value=str(types)
                        ))
        
        except Exception as e:
            logger.error(f"Statistical validation error: {e}")
            results.append(ValidationResult(
                field="dataframe",
                rule="statistical_validation",
                severity=ValidationSeverity.CRITICAL,
                message=f"Statistical validation failed: {str(e)}"
            ))
        
        return results
    
    # Built-in validator functions
    def _validate_required(self, value: Any) -> tuple[bool, str]:
        """Validate that a value is present and not empty."""
        if value is None:
            return False, "Value is required"
        if isinstance(value, str) and not value.strip():
            return False, "Value cannot be empty"
        if isinstance(value, (list, dict)) and len(value) == 0:
            return False, "Value cannot be empty"
        return True, ""
    
    def _validate_type(self, value: Any, expected_type: str) -> tuple[bool, str]:
        """Validate value type."""
        if value is None:
            return True, ""  # Type validation passes for None
        
        type_map = {
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list': list,
            'dict': dict
        }
        
        expected_python_type = type_map.get(expected_type)
        if not expected_python_type:
            return False, f"Unknown type: {expected_type}"
        
        if not isinstance(value, expected_python_type):
            return False, f"Expected {expected_type}, got {type(value).__name__}"
        
        return True, ""
    
    def _validate_range(self, value: Any, min_val: Optional[float] = None, max_val: Optional[float] = None) -> tuple[bool, str]:
        """Validate numeric range."""
        if value is None:
            return True, ""
        
        try:
            num_value = float(value)
        except (ValueError, TypeError):
            return False, "Value must be numeric for range validation"
        
        if min_val is not None and num_value < min_val:
            return False, f"Value {num_value} is less than minimum {min_val}"
        
        if max_val is not None and num_value > max_val:
            return False, f"Value {num_value} is greater than maximum {max_val}"
        
        return True, ""
    
    def _validate_length(self, value: Any, min_len: Optional[int] = None, max_len: Optional[int] = None) -> tuple[bool, str]:
        """Validate length constraints."""
        if value is None:
            return True, ""
        
        try:
            length = len(value)
        except TypeError:
            return False, "Value does not have length"
        
        if min_len is not None and length < min_len:
            return False, f"Length {length} is less than minimum {min_len}"
        
        if max_len is not None and length > max_len:
            return False, f"Length {length} is greater than maximum {max_len}"
        
        return True, ""
    
    def _validate_pattern(self, value: Any, pattern: str) -> tuple[bool, str]:
        """Validate regex pattern."""
        if value is None:
            return True, ""
        
        if not isinstance(value, str):
            return False, "Pattern validation requires string value"
        
        try:
            if not re.match(pattern, value):
                return False, f"Value does not match pattern: {pattern}"
            return True, ""
        except re.error as e:
            return False, f"Invalid regex pattern: {e}"
    
    def _validate_email(self, value: Any) -> tuple[bool, str]:
        """Validate email format."""
        if value is None:
            return True, ""
        
        if not isinstance(value, str):
            return False, "Email must be a string"
        
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, value):
            return False, "Invalid email format"
        
        return True, ""
    
    def _validate_url(self, value: Any) -> tuple[bool, str]:
        """Validate URL format."""
        if value is None:
            return True, ""
        
        if not isinstance(value, str):
            return False, "URL must be a string"
        
        url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        if not re.match(url_pattern, value, re.IGNORECASE):
            return False, "Invalid URL format"
        
        return True, ""
    
    def _validate_date(self, value: Any, date_format: str = "%Y-%m-%d") -> tuple[bool, str]:
        """Validate date format."""
        if value is None:
            return True, ""
        
        if not isinstance(value, str):
            return False, "Date must be a string"
        
        try:
            datetime.strptime(value, date_format)
            return True, ""
        except ValueError:
            return False, f"Invalid date format. Expected: {date_format}"
    
    def _validate_numeric(self, value: Any) -> tuple[bool, str]:
        """Validate that value is numeric."""
        if value is None:
            return True, ""
        
        try:
            float(value)
            return True, ""
        except (ValueError, TypeError):
            return False, "Value must be numeric"
    
    def _validate_unique(self, values: List[Any]) -> tuple[bool, str]:
        """Validate that all values in list are unique."""
        if not isinstance(values, list):
            return False, "Unique validation requires a list"
        
        if len(values) != len(set(values)):
            return False, "Values are not unique"
        
        return True, ""
    
    def _validate_not_null(self, value: Any) -> tuple[bool, str]:
        """Validate that value is not None or NaN."""
        if value is None:
            return False, "Value cannot be null"
        
        if isinstance(value, float) and np.isnan(value):
            return False, "Value cannot be NaN"
        
        return True, ""
    
    def _validate_enum(self, value: Any, choices: List[Any]) -> tuple[bool, str]:
        """Validate that value is in allowed choices."""
        if value is None:
            return True, ""
        
        if value not in choices:
            return False, f"Value must be one of: {choices}"
        
        return True, ""


# Global data validator instance
data_validator = DataValidator()


# Common validation rule sets
def create_document_validation_rules(validator: DataValidator) -> None:
    """Create validation rules for document processing.
    
    Args:
        validator: DataValidator instance to configure
    """
    # Document metadata validation
    validator.add_rule("filename", "required", severity=ValidationSeverity.ERROR)
    validator.add_rule("filename", "length", {"max_len": 255}, ValidationSeverity.WARNING)
    
    validator.add_rule("file_size", "required", severity=ValidationSeverity.ERROR)
    validator.add_rule("file_size", "range", {"min_val": 1, "max_val": settings.max_file_size}, ValidationSeverity.ERROR)
    
    validator.add_rule("content_type", "required", severity=ValidationSeverity.ERROR)
    validator.add_rule("content_type", "enum", {
        "choices": ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", 
                   "text/plain", "text/markdown", "text/csv"]
    }, ValidationSeverity.ERROR)
    
    # Content validation
    validator.add_rule("text_content", "required", severity=ValidationSeverity.WARNING)
    validator.add_rule("word_count", "range", {"min_val": 1}, ValidationSeverity.INFO)


def create_user_validation_rules(validator: DataValidator) -> None:
    """Create validation rules for user data.
    
    Args:
        validator: DataValidator instance to configure
    """
    validator.add_rule("username", "required", severity=ValidationSeverity.ERROR)
    validator.add_rule("username", "length", {"min_len": 3, "max_len": 50}, ValidationSeverity.ERROR)
    validator.add_rule("username", "pattern", {"pattern": r'^[a-zA-Z0-9_]+$'}, ValidationSeverity.ERROR)
    
    validator.add_rule("email", "required", severity=ValidationSeverity.ERROR)
    validator.add_rule("email", "email", severity=ValidationSeverity.ERROR)
    
    validator.add_rule("password", "required", severity=ValidationSeverity.ERROR)
    validator.add_rule("password", "length", {"min_len": 8}, ValidationSeverity.ERROR)
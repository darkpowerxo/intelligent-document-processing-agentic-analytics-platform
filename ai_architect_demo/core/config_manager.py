"""Configuration management utilities for AI Architect Demo.

This module provides comprehensive configuration management including:
- Environment-specific configurations
- Dynamic configuration updates
- Configuration validation
- Secret management
- Configuration versioning
"""

import os
import json
import yaml
from typing import Any, Dict, List, Optional, Type, Union
from pathlib import Path
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, validator, ValidationError
from pydantic.env_settings import BaseSettings

from ai_architect_demo.core.config import settings
from ai_architect_demo.core.logging import get_logger, log_function_call

logger = get_logger(__name__)


class Environment(Enum):
    """Environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class ConfigFormat(Enum):
    """Configuration file formats."""
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    ENV = "env"


class ConfigSchema(BaseModel):
    """Base configuration schema."""
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = True
    log_level: str = "INFO"
    created_at: datetime = Field(default_factory=datetime.now)
    version: str = "1.0.0"
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DatabaseConfig(BaseModel):
    """Database configuration schema."""
    host: str = "localhost"
    port: int = 5432
    database: str = "ai_architect_demo"
    username: str = "postgres"
    password: str = Field(..., env="DATABASE_PASSWORD")
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False
    
    @validator('port')
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError('Port must be between 1 and 65535')
        return v


class RedisConfig(BaseModel):
    """Redis configuration schema."""
    host: str = "localhost"
    port: int = 6379
    database: int = 0
    password: Optional[str] = Field(None, env="REDIS_PASSWORD")
    max_connections: int = 10
    decode_responses: bool = True


class MLConfig(BaseModel):
    """Machine learning configuration schema."""
    model_registry_uri: str = "sqlite:///mlruns.db"
    experiment_name: str = "default"
    tracking_uri: str = "http://localhost:5000"
    artifact_root: str = "./mlruns"
    default_model_stage: str = "Staging"
    auto_log: bool = True


class APIConfig(BaseModel):
    """API configuration schema."""
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    workers: int = 1
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    cors_origins: List[str] = ["*"]
    api_v1_prefix: str = "/api/v1"


class SecurityConfig(BaseModel):
    """Security configuration schema."""
    secret_key: str = Field(..., env="SECRET_KEY")
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    algorithm: str = "HS256"
    bcrypt_rounds: int = 12
    
    @validator('secret_key')
    def validate_secret_key(cls, v):
        if len(v) < 32:
            raise ValueError('Secret key must be at least 32 characters long')
        return v


class MonitoringConfig(BaseModel):
    """Monitoring and observability configuration."""
    metrics_enabled: bool = True
    metrics_port: int = 8001
    tracing_enabled: bool = False
    tracing_endpoint: Optional[str] = None
    health_check_interval: int = 30
    alert_webhook_url: Optional[str] = None


class ComprehensiveConfig(ConfigSchema):
    """Complete application configuration."""
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    security: SecurityConfig = Field(...)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)


class ConfigManager:
    """Advanced configuration management with validation and updates."""
    
    def __init__(self, config_dir: str = "config", environment: Optional[Environment] = None):
        """Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
            environment: Target environment
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.environment = environment or Environment.DEVELOPMENT
        self.config_cache = {}
        self.config_history = []
        
        # Setup default configuration files
        self._setup_default_configs()
    
    def _setup_default_configs(self) -> None:
        """Setup default configuration files for each environment."""
        for env in Environment:
            config_file = self.config_dir / f"{env.value}.yaml"
            if not config_file.exists():
                self._create_default_config(env, config_file)
    
    def _create_default_config(self, env: Environment, config_file: Path) -> None:
        """Create default configuration file for environment.
        
        Args:
            env: Environment type
            config_file: Path to configuration file
        """
        try:
            # Environment-specific defaults
            if env == Environment.PRODUCTION:
                config_data = {
                    'environment': env.value,
                    'debug': False,
                    'log_level': 'WARNING',
                    'database': {
                        'echo': False,
                        'pool_size': 20,
                        'max_overflow': 40
                    },
                    'api': {
                        'reload': False,
                        'workers': 4,
                        'cors_origins': []
                    },
                    'monitoring': {
                        'metrics_enabled': True,
                        'tracing_enabled': True,
                        'health_check_interval': 15
                    }
                }
            elif env == Environment.TESTING:
                config_data = {
                    'environment': env.value,
                    'debug': True,
                    'log_level': 'DEBUG',
                    'database': {
                        'database': 'test_ai_architect_demo',
                        'echo': True,
                        'pool_size': 5
                    },
                    'redis': {
                        'database': 1
                    }
                }
            else:  # Development/Staging
                config_data = {
                    'environment': env.value,
                    'debug': True,
                    'log_level': 'INFO',
                    'database': {
                        'echo': env == Environment.DEVELOPMENT
                    }
                }
            
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            
            logger.info(f"Created default configuration for {env.value}: {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to create default config for {env.value}: {e}")
    
    def load_config(
        self, 
        config_schema: Type[BaseModel] = ComprehensiveConfig,
        environment: Optional[Environment] = None,
        config_file: Optional[str] = None
    ) -> BaseModel:
        """Load and validate configuration.
        
        Args:
            config_schema: Pydantic model for configuration validation
            environment: Target environment
            config_file: Specific configuration file to load
            
        Returns:
            Validated configuration object
        """
        log_function_call("load_config", schema=config_schema.__name__)
        
        env = environment or self.environment
        
        # Determine config file
        if config_file:
            config_path = Path(config_file)
        else:
            config_path = self.config_dir / f"{env.value}.yaml"
        
        # Check cache
        cache_key = f"{config_schema.__name__}_{config_path}"
        if cache_key in self.config_cache:
            return self.config_cache[cache_key]
        
        try:
            # Load configuration data
            config_data = self._load_config_file(config_path)
            
            # Merge with environment variables
            config_data = self._merge_env_variables(config_data)
            
            # Validate configuration
            config_instance = config_schema(**config_data)
            
            # Cache configuration
            self.config_cache[cache_key] = config_instance
            
            # Record in history
            self.config_history.append({
                'timestamp': datetime.now(),
                'environment': env.value,
                'schema': config_schema.__name__,
                'config_file': str(config_path),
                'success': True
            })
            
            logger.info(f"Loaded configuration for {env.value} from {config_path}")
            return config_instance
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            
            # Record failure in history
            self.config_history.append({
                'timestamp': datetime.now(),
                'environment': env.value,
                'schema': config_schema.__name__,
                'config_file': str(config_path) if config_path else None,
                'success': False,
                'error': str(e)
            })
            
            raise
    
    def save_config(
        self, 
        config_instance: BaseModel,
        environment: Optional[Environment] = None,
        config_file: Optional[str] = None,
        format: ConfigFormat = ConfigFormat.YAML
    ) -> None:
        """Save configuration to file.
        
        Args:
            config_instance: Configuration instance to save
            environment: Target environment
            config_file: Specific file to save to
            format: Configuration file format
        """
        log_function_call("save_config", config_type=type(config_instance).__name__)
        
        env = environment or self.environment
        
        # Determine output file
        if config_file:
            output_path = Path(config_file)
        else:
            extension = format.value if format != ConfigFormat.ENV else "env"
            output_path = self.config_dir / f"{env.value}.{extension}"
        
        try:
            # Convert config to dictionary
            config_data = config_instance.dict()
            
            # Save based on format
            if format == ConfigFormat.JSON:
                with open(output_path, 'w') as f:
                    json.dump(config_data, f, indent=2, default=str)
            elif format == ConfigFormat.YAML:
                with open(output_path, 'w') as f:
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
            elif format == ConfigFormat.ENV:
                self._save_as_env_file(config_data, output_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Saved configuration to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def update_config(
        self,
        updates: Dict[str, Any],
        config_schema: Type[BaseModel] = ComprehensiveConfig,
        environment: Optional[Environment] = None,
        validate: bool = True
    ) -> BaseModel:
        """Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
            config_schema: Configuration schema
            environment: Target environment
            validate: Whether to validate updated configuration
            
        Returns:
            Updated configuration instance
        """
        log_function_call("update_config", updates_count=len(updates))
        
        # Load current configuration
        current_config = self.load_config(config_schema, environment)
        current_data = current_config.dict()
        
        # Apply updates recursively
        updated_data = self._deep_update(current_data, updates)
        
        if validate:
            # Validate updated configuration
            try:
                updated_config = config_schema(**updated_data)
            except ValidationError as e:
                logger.error(f"Configuration validation failed after update: {e}")
                raise
        else:
            updated_config = config_schema.construct(**updated_data)
        
        # Save updated configuration
        self.save_config(updated_config, environment)
        
        # Clear cache for this configuration
        env = environment or self.environment
        cache_key = f"{config_schema.__name__}_{self.config_dir / f'{env.value}.yaml'}"
        if cache_key in self.config_cache:
            del self.config_cache[cache_key]
        
        logger.info(f"Updated configuration for {env.value}")
        return updated_config
    
    def validate_config(
        self,
        config_data: Dict[str, Any],
        config_schema: Type[BaseModel] = ComprehensiveConfig
    ) -> tuple[bool, List[str]]:
        """Validate configuration data against schema.
        
        Args:
            config_data: Configuration data to validate
            config_schema: Schema for validation
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        try:
            config_schema(**config_data)
            return True, []
        except ValidationError as e:
            error_messages = []
            for error in e.errors():
                field = '.'.join(str(x) for x in error['loc'])
                message = error['msg']
                error_messages.append(f"{field}: {message}")
            return False, error_messages
    
    def get_config_diff(
        self,
        config1: BaseModel,
        config2: BaseModel
    ) -> Dict[str, Any]:
        """Get differences between two configurations.
        
        Args:
            config1: First configuration
            config2: Second configuration
            
        Returns:
            Dictionary of differences
        """
        data1 = config1.dict()
        data2 = config2.dict()
        
        return self._get_dict_diff(data1, data2)
    
    def backup_config(
        self,
        environment: Optional[Environment] = None,
        backup_dir: Optional[str] = None
    ) -> str:
        """Create a backup of current configuration.
        
        Args:
            environment: Environment to backup
            backup_dir: Directory to store backup
            
        Returns:
            Path to backup file
        """
        env = environment or self.environment
        
        if backup_dir:
            backup_path = Path(backup_dir)
        else:
            backup_path = self.config_dir / "backups"
        
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped backup filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_path / f"{env.value}_backup_{timestamp}.yaml"
        
        try:
            # Load current config
            current_config = self.load_config(environment=env)
            
            # Save as backup
            with open(backup_file, 'w') as f:
                yaml.dump(current_config.dict(), f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration backup created: {backup_file}")
            return str(backup_file)
            
        except Exception as e:
            logger.error(f"Failed to create configuration backup: {e}")
            raise
    
    def restore_config(
        self,
        backup_file: str,
        environment: Optional[Environment] = None,
        config_schema: Type[BaseModel] = ComprehensiveConfig
    ) -> BaseModel:
        """Restore configuration from backup.
        
        Args:
            backup_file: Path to backup file
            environment: Target environment
            config_schema: Configuration schema
            
        Returns:
            Restored configuration instance
        """
        log_function_call("restore_config", backup_file=backup_file)
        
        env = environment or self.environment
        
        try:
            # Load backup data
            backup_data = self._load_config_file(Path(backup_file))
            
            # Validate backup data
            restored_config = config_schema(**backup_data)
            
            # Save as current configuration
            self.save_config(restored_config, env)
            
            logger.info(f"Configuration restored from {backup_file} to {env.value}")
            return restored_config
            
        except Exception as e:
            logger.error(f"Failed to restore configuration: {e}")
            raise
    
    def _load_config_file(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        file_extension = config_path.suffix.lower()
        
        with open(config_path, 'r') as f:
            if file_extension == '.json':
                return json.load(f)
            elif file_extension in ['.yaml', '.yml']:
                return yaml.safe_load(f) or {}
            elif file_extension == '.env':
                return self._load_env_file(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {file_extension}")
    
    def _load_env_file(self, file_obj) -> Dict[str, Any]:
        """Load configuration from .env file."""
        config = {}
        for line in file_obj:
            line = line.strip()
            if line and not line.startswith('#'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
        return config
    
    def _merge_env_variables(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Merge environment variables into configuration."""
        # This is a simple implementation - in practice, you might want more sophisticated merging
        for key, value in os.environ.items():
            if key.startswith('APP_'):
                # Convert APP_DATABASE_HOST to database.host
                config_key = key[4:].lower().replace('_', '.')
                self._set_nested_dict(config_data, config_key, value)
        
        return config_data
    
    def _save_as_env_file(self, config_data: Dict[str, Any], output_path: Path) -> None:
        """Save configuration as .env file."""
        with open(output_path, 'w') as f:
            self._write_env_vars(f, config_data)
    
    def _write_env_vars(self, file_obj, data: Dict[str, Any], prefix: str = 'APP') -> None:
        """Write configuration as environment variables."""
        for key, value in data.items():
            env_key = f"{prefix}_{key.upper()}"
            
            if isinstance(value, dict):
                self._write_env_vars(file_obj, value, env_key)
            elif isinstance(value, list):
                file_obj.write(f"{env_key}={','.join(map(str, value))}\n")
            else:
                file_obj.write(f"{env_key}={value}\n")
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively update nested dictionary."""
        result = base_dict.copy()
        
        for key, value in update_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_update(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _set_nested_dict(self, dictionary: Dict[str, Any], key_path: str, value: Any) -> None:
        """Set value in nested dictionary using dot notation."""
        keys = key_path.split('.')
        current = dictionary
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _get_dict_diff(self, dict1: Dict[str, Any], dict2: Dict[str, Any], path: str = '') -> Dict[str, Any]:
        """Get differences between two dictionaries."""
        diff = {}
        
        # Check for changes and additions in dict2
        for key, value in dict2.items():
            current_path = f"{path}.{key}" if path else key
            
            if key not in dict1:
                diff[f"+{current_path}"] = value
            elif isinstance(value, dict) and isinstance(dict1[key], dict):
                nested_diff = self._get_dict_diff(dict1[key], value, current_path)
                diff.update(nested_diff)
            elif dict1[key] != value:
                diff[f"~{current_path}"] = {'old': dict1[key], 'new': value}
        
        # Check for removals in dict2
        for key in dict1:
            current_path = f"{path}.{key}" if path else key
            if key not in dict2:
                diff[f"-{current_path}"] = dict1[key]
        
        return diff


# Global configuration manager instance
config_manager = ConfigManager()
"""Platform initialization script for AI Architect Demo.

This script sets up the initial state of the platform including:
- Database connections
- MLflow experiments  
- Ollama model verification
- Initial data seeding
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_architect_demo.core.config import settings
from ai_architect_demo.core.logging import get_logger

logger = get_logger(__name__)


async def init_database():
    """Initialize database connections and verify schema."""
    try:
        # Import here to avoid circular imports
        from ai_architect_demo.core.database import database, get_database
        
        logger.info("Connecting to PostgreSQL database...")
        await database.connect()
        logger.info("Database connection successful")
        
        # Test query
        query = "SELECT COUNT(*) as count FROM app.users"
        result = await database.fetch_one(query)
        logger.info(f"Found {result['count']} users in database")
        
        await database.disconnect()
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


async def init_mlflow():
    """Initialize MLflow experiments and verify connection."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        
        logger.info("Connecting to MLflow tracking server...")
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        
        client = MlflowClient()
        
        # Create default experiment if it doesn't exist
        try:
            experiment = client.get_experiment_by_name(settings.mlflow_experiment_name)
            if experiment is None:
                experiment_id = client.create_experiment(settings.mlflow_experiment_name)
                logger.info(f"Created MLflow experiment: {settings.mlflow_experiment_name} (ID: {experiment_id})")
            else:
                logger.info(f"MLflow experiment exists: {settings.mlflow_experiment_name}")
                
        except Exception as e:
            logger.warning(f"Could not verify MLflow experiment: {e}")
            
    except Exception as e:
        logger.error(f"MLflow initialization failed: {e}")
        raise


async def init_ollama():
    """Initialize and verify Ollama LLM service."""
    try:
        import httpx
        
        logger.info("Verifying Ollama service...")
        ollama_url = f"{settings.ollama_base_url}/api/tags"
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(ollama_url)
            response.raise_for_status()
            
            models = response.json()
            model_names = [model.get('name', '') for model in models.get('models', [])]
            
            if settings.ollama_model in model_names:
                logger.info(f"Ollama model {settings.ollama_model} is available")
            else:
                logger.warning(f"Ollama model {settings.ollama_model} not found. Available models: {model_names}")
                
    except Exception as e:
        logger.error(f"Ollama initialization failed: {e}")
        raise


async def init_kafka():
    """Initialize Kafka topics."""
    try:
        from kafka import KafkaProducer, KafkaAdminClient
        from kafka.admin import ConfigResource, ConfigResourceType, NewTopic
        
        logger.info("Connecting to Kafka...")
        
        # Create admin client
        admin_client = KafkaAdminClient(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            client_id='ai_demo_init'
        )
        
        # Create topics if they don't exist
        topics_to_create = []
        for topic_name in settings.kafka_topics.values():
            topics_to_create.append(
                NewTopic(
                    name=topic_name,
                    num_partitions=3,
                    replication_factor=1
                )
            )
        
        # Create topics
        try:
            result = admin_client.create_topics(topics_to_create, validate_only=False)
            logger.info("Created Kafka topics successfully")
        except Exception as e:
            logger.info(f"Topics may already exist: {e}")
            
        admin_client.close()
        
    except Exception as e:
        logger.error(f"Kafka initialization failed: {e}")
        raise


async def init_minio():
    """Initialize MinIO buckets."""
    try:
        from minio import Minio
        
        logger.info("Connecting to MinIO...")
        
        client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure
        )
        
        # Create bucket if it doesn't exist
        if not client.bucket_exists(settings.minio_bucket_name):
            client.make_bucket(settings.minio_bucket_name)
            logger.info(f"Created MinIO bucket: {settings.minio_bucket_name}")
        else:
            logger.info(f"MinIO bucket exists: {settings.minio_bucket_name}")
            
    except Exception as e:
        logger.error(f"MinIO initialization failed: {e}")
        raise


async def main():
    """Main initialization function."""
    logger.info("Starting AI Architect Demo platform initialization...")
    
    # Create uploads directory
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created uploads directory: {settings.upload_dir}")
    
    # Initialize components
    initialization_tasks = [
        ("Database", init_database),
        ("MLflow", init_mlflow),
        ("Ollama", init_ollama), 
        ("Kafka", init_kafka),
        ("MinIO", init_minio),
    ]
    
    for component_name, init_func in initialization_tasks:
        try:
            logger.info(f"Initializing {component_name}...")
            await init_func()
            logger.info(f"{component_name} initialization completed")
        except Exception as e:
            logger.error(f"{component_name} initialization failed: {e}")
            # Continue with other components
            continue
    
    logger.info("Platform initialization completed!")
    logger.info("You can now start the AI Architect Demo services.")
    logger.info("Access points:")
    logger.info(f"  - API Documentation: http://localhost:8000/docs")
    logger.info(f"  - Streamlit Dashboard: http://localhost:8501") 
    logger.info(f"  - MLflow UI: http://localhost:5000")
    logger.info(f"  - Grafana Monitoring: http://localhost:3000")
    logger.info(f"  - Jupyter Lab: http://localhost:8888")
    logger.info(f"  - Kafka UI: http://localhost:8080")


if __name__ == "__main__":
    asyncio.run(main())
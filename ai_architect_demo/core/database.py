"""Database utilities and connection management for AI Architect Demo.

This module provides database connection management, query utilities,
and data access patterns using SQLAlchemy with async support.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional

import asyncpg
import sqlalchemy as sa
from databases import Database
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from ai_architect_demo.core.config import settings
from ai_architect_demo.core.logging import get_logger

logger = get_logger(__name__)

# SQLAlchemy base class
Base = declarative_base()

# Global database instance
database: Optional[Database] = None
engine: Optional[AsyncEngine] = None


async def init_database() -> None:
    """Initialize database connections and engine."""
    global database, engine
    
    try:
        # Create async database URL
        async_url = settings.database_url.replace("postgresql://", "postgresql+asyncpg://")
        
        # Create engine
        engine = create_async_engine(
            async_url,
            poolclass=NullPool,  # Use NullPool for simplicity in demo
            echo=settings.debug,
        )
        
        # Create database instance
        database = Database(async_url)
        
        logger.info("Database engine and connection initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


async def connect_database() -> None:
    """Connect to the database."""
    global database
    
    if database is None:
        await init_database()
    
    try:
        await database.connect()
        logger.info("Connected to database successfully")
        
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise


async def disconnect_database() -> None:
    """Disconnect from the database."""
    global database
    
    if database:
        try:
            await database.disconnect()
            logger.info("Disconnected from database successfully")
            
        except Exception as e:
            logger.error(f"Failed to disconnect from database: {e}")


@asynccontextmanager
async def get_database() -> AsyncGenerator[Database, None]:
    """Get database connection context manager."""
    global database
    
    if database is None:
        await init_database()
        await connect_database()
    
    try:
        yield database
    finally:
        pass  # Keep connection open for reuse


async def execute_query(
    query: str, 
    values: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Execute a query and return results.
    
    Args:
        query: SQL query string
        values: Query parameters
        
    Returns:
        List of result dictionaries
    """
    async with get_database() as db:
        try:
            result = await db.fetch_all(query, values or {})
            return [dict(row) for row in result]
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Values: {values}")
            raise


async def execute_single(
    query: str,
    values: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """Execute a query and return single result.
    
    Args:
        query: SQL query string  
        values: Query parameters
        
    Returns:
        Single result dictionary or None
    """
    async with get_database() as db:
        try:
            result = await db.fetch_one(query, values or {})
            return dict(result) if result else None
            
        except Exception as e:
            logger.error(f"Single query execution failed: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Values: {values}")
            raise


async def execute_command(
    query: str,
    values: Optional[Dict[str, Any]] = None
) -> int:
    """Execute a command (INSERT, UPDATE, DELETE) and return affected rows.
    
    Args:
        query: SQL command string
        values: Command parameters
        
    Returns:
        Number of affected rows
    """
    async with get_database() as db:
        try:
            result = await db.execute(query, values or {})
            return result
            
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Values: {values}")
            raise


class DatabaseManager:
    """Database manager class for handling connections and transactions."""
    
    def __init__(self):
        self.database = database
        self.engine = engine
        
    async def health_check(self) -> bool:
        """Check database health.
        
        Returns:
            True if database is healthy, False otherwise
        """
        try:
            result = await execute_single("SELECT 1 as health")
            return result is not None and result.get("health") == 1
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    async def get_table_info(self, schema: str = "app") -> List[Dict[str, Any]]:
        """Get information about tables in a schema.
        
        Args:
            schema: Schema name (default: 'app')
            
        Returns:
            List of table information
        """
        query = """
        SELECT 
            table_name,
            table_type,
            table_schema
        FROM information_schema.tables 
        WHERE table_schema = :schema
        ORDER BY table_name
        """
        
        return await execute_query(query, {"schema": schema})
    
    async def get_user_count(self) -> int:
        """Get total number of users.
        
        Returns:
            Number of users in the system
        """
        result = await execute_single("SELECT COUNT(*) as count FROM app.users")
        return result.get("count", 0) if result else 0
    
    async def get_document_stats(self) -> Dict[str, Any]:
        """Get document processing statistics.
        
        Returns:
            Dictionary with document statistics
        """
        query = """
        SELECT 
            processing_status,
            COUNT(*) as count
        FROM app.documents 
        GROUP BY processing_status
        """
        
        results = await execute_query(query)
        stats = {row["processing_status"]: row["count"] for row in results}
        
        # Add total count
        total_query = "SELECT COUNT(*) as total FROM app.documents"
        total_result = await execute_single(total_query)
        stats["total"] = total_result.get("total", 0) if total_result else 0
        
        return stats
    
    async def get_agent_performance(self) -> List[Dict[str, Any]]:
        """Get agent performance metrics.
        
        Returns:
            List of agent performance data
        """
        query = """
        SELECT 
            agent_type,
            agent_name,
            status,
            total_tasks_completed,
            average_processing_time_ms,
            last_activity
        FROM app.agents
        ORDER BY total_tasks_completed DESC
        """
        
        return await execute_query(query)


# Global database manager instance
db_manager = DatabaseManager()


# Database connection event handlers
async def startup_database():
    """Database startup event handler."""
    await init_database()
    await connect_database()
    
    # Verify connection
    is_healthy = await db_manager.health_check()
    if is_healthy:
        logger.info("Database is healthy and ready")
    else:
        logger.warning("Database health check failed")


async def shutdown_database():
    """Database shutdown event handler."""
    await disconnect_database()
    logger.info("Database connections closed")
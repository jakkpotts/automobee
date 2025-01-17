from typing import Dict, List, Optional, Any
import logging
import asyncio
from datetime import datetime
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

logger = logging.getLogger(__name__)

class DatabaseService:
    """Handles database operations for the AutomoBee system"""
    
    def __init__(self, db_url: str):
        """Initialize database service
        
        Args:
            db_url: Database connection URL
        """
        # Convert SQLite URL to async version if needed
        if db_url.startswith('sqlite:///'):
            db_url = db_url.replace('sqlite:///', 'sqlite+aiosqlite:///')
            
        self.engine = create_async_engine(
            db_url,
            echo=False,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10
        )
        self.async_session = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    async def initialize_tables(self) -> None:
        """Initialize database tables"""
        async with self.engine.begin() as conn:
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    vehicle_type TEXT,
                    confidence FLOAT,
                    metadata JSON,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_type TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    value FLOAT,
                    metadata JSON,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_detections_camera_timestamp 
                ON detections(camera_id, timestamp)
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_metrics_type_timestamp 
                ON metrics(metric_type, timestamp)
            """))
    
    async def store_detection(self, camera_id: str, detection_data: Dict) -> None:
        """Store vehicle detection data
        
        Args:
            camera_id: ID of the camera that made the detection
            detection_data: Detection information
        """
        async with self.async_session() as session:
            async with session.begin():
                await session.execute(
                    text("""
                        INSERT INTO detections (
                            camera_id, timestamp, vehicle_type, 
                            confidence, metadata
                        ) VALUES (:camera_id, :timestamp, :vehicle_type, 
                                :confidence, :metadata)
                    """),
                    {
                        "camera_id": camera_id,
                        "timestamp": datetime.now(),
                        "vehicle_type": detection_data.get("type"),
                        "confidence": detection_data.get("confidence"),
                        "metadata": detection_data.get("metadata", {})
                    }
                )
    
    async def store_metric(self, metric_type: str, value: float, metadata: Optional[Dict] = None) -> None:
        """Store system metric
        
        Args:
            metric_type: Type of metric
            value: Metric value
            metadata: Optional metadata
        """
        async with self.async_session() as session:
            async with session.begin():
                await session.execute(
                    text("""
                        INSERT INTO metrics (metric_type, timestamp, value, metadata)
                        VALUES (:metric_type, :timestamp, :value, :metadata)
                    """),
                    {
                        "metric_type": metric_type,
                        "timestamp": datetime.now(),
                        "value": value,
                        "metadata": metadata or {}
                    }
                )
    
    async def get_recent_detections(self, camera_id: str, limit: int = 100) -> List[Dict]:
        """Get recent detections for a camera
        
        Args:
            camera_id: ID of the camera
            limit: Maximum number of records to return
        
        Returns:
            List of detection records
        """
        async with self.async_session() as session:
            result = await session.execute(
                text("""
                    SELECT * FROM detections 
                    WHERE camera_id = :camera_id
                    ORDER BY timestamp DESC 
                    LIMIT :limit
                """),
                {"camera_id": camera_id, "limit": limit}
            )
            return [dict(row) for row in result]
    
    async def get_metrics(self, metric_type: str, 
                         start_time: datetime,
                         end_time: datetime) -> List[Dict]:
        """Get metrics for a specific time range
        
        Args:
            metric_type: Type of metric to retrieve
            start_time: Start of time range
            end_time: End of time range
        
        Returns:
            List of metric records
        """
        async with self.async_session() as session:
            result = await session.execute(
                text("""
                    SELECT * FROM metrics 
                    WHERE metric_type = :metric_type
                    AND timestamp BETWEEN :start_time AND :end_time
                    ORDER BY timestamp DESC
                """),
                {
                    "metric_type": metric_type,
                    "start_time": start_time,
                    "end_time": end_time
                }
            )
            return [dict(row) for row in result]
    
    async def cleanup_old_data(self, days_to_keep: int = 30) -> None:
        """Clean up old data from the database
        
        Args:
            days_to_keep: Number of days of data to retain
        """
        cleanup_date = datetime.now() - timedelta(days=days_to_keep)
        
        async with self.async_session() as session:
            async with session.begin():
                await session.execute(
                    text("""
                        DELETE FROM detections 
                        WHERE timestamp < :cleanup_date
                    """),
                    {"cleanup_date": cleanup_date}
                )
                
                await session.execute(
                    text("""
                        DELETE FROM metrics 
                        WHERE timestamp < :cleanup_date
                    """),
                    {"cleanup_date": cleanup_date}
                )
    
    async def cleanup(self) -> None:
        """Cleanup database resources"""
        await self.engine.dispose()
        logger.info("Database service cleaned up") 
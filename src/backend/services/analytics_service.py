from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta
import json
import asyncio
from collections import defaultdict

logger = logging.getLogger(__name__)

class AnalyticsService:
    """Handles analytics and metrics collection for the AutomoBee system"""
    
    def __init__(self, db_service: 'DatabaseService'):
        """Initialize analytics service
        
        Args:
            db_service: Database service for storing metrics
        """
        self.db_service = db_service
        self._metrics_buffer: Dict[str, List[Dict]] = defaultdict(list)
        self._buffer_size = 100
        self._flush_interval = 60  # seconds
        self._flush_task: Optional[asyncio.Task] = None
        self._is_running = False
    
    async def initialize(self) -> None:
        """Initialize the analytics service"""
        self._is_running = True
        self._flush_task = asyncio.create_task(self._periodic_flush())
        logger.info("Analytics service initialized")
    
    async def store_metrics(self, metrics: Dict[str, Any]) -> None:
        """Store system metrics
        
        Args:
            metrics: Dictionary of metrics to store
        """
        timestamp = datetime.now()
        
        for metric_type, value in metrics.items():
            if isinstance(value, (int, float)):
                self._metrics_buffer[metric_type].append({
                    'timestamp': timestamp,
                    'value': float(value)
                })
            elif isinstance(value, dict):
                # Store complex metrics as separate entries
                for sub_type, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        full_type = f"{metric_type}.{sub_type}"
                        self._metrics_buffer[full_type].append({
                            'timestamp': timestamp,
                            'value': float(sub_value)
                        })
        
        # Flush if buffer is full
        for metric_type, buffer in self._metrics_buffer.items():
            if len(buffer) >= self._buffer_size:
                await self._flush_metric_type(metric_type)
    
    async def _flush_metric_type(self, metric_type: str) -> None:
        """Flush metrics of a specific type to database
        
        Args:
            metric_type: Type of metric to flush
        """
        buffer = self._metrics_buffer[metric_type]
        if not buffer:
            return
        
        try:
            # Calculate average for the period
            total_value = sum(m['value'] for m in buffer)
            avg_value = total_value / len(buffer)
            
            # Store aggregated metric
            await self.db_service.store_metric(
                metric_type=metric_type,
                value=avg_value,
                metadata={
                    'count': len(buffer),
                    'min': min(m['value'] for m in buffer),
                    'max': max(m['value'] for m in buffer),
                    'period_start': buffer[0]['timestamp'].isoformat(),
                    'period_end': buffer[-1]['timestamp'].isoformat()
                }
            )
            
            # Clear buffer
            self._metrics_buffer[metric_type].clear()
            
        except Exception as e:
            logger.error(f"Error flushing metrics for {metric_type}: {e}")
    
    async def _periodic_flush(self) -> None:
        """Periodically flush all metric buffers"""
        while self._is_running:
            try:
                await asyncio.sleep(self._flush_interval)
                
                # Flush all metric types
                for metric_type in list(self._metrics_buffer.keys()):
                    await self._flush_metric_type(metric_type)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic flush: {e}")
                await asyncio.sleep(5)  # Back off on error
    
    async def get_metric_summary(self, 
                               metric_type: str,
                               time_window: timedelta) -> Dict[str, float]:
        """Get summary statistics for a metric
        
        Args:
            metric_type: Type of metric to summarize
            time_window: Time window to analyze
        
        Returns:
            Dictionary of summary statistics
        """
        end_time = datetime.now()
        start_time = end_time - time_window
        
        try:
            # Get metrics from database
            metrics = await self.db_service.get_metrics(
                metric_type=metric_type,
                start_time=start_time,
                end_time=end_time
            )
            
            if not metrics:
                return {
                    'count': 0,
                    'avg': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'current': 0.0
                }
            
            # Calculate statistics
            values = [m['value'] for m in metrics]
            return {
                'count': len(values),
                'avg': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'current': values[-1] if values else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error getting metric summary for {metric_type}: {e}")
            return {
                'count': 0,
                'avg': 0.0,
                'min': 0.0,
                'max': 0.0,
                'current': 0.0,
                'error': str(e)
            }
    
    async def get_detection_trends(self, 
                                 camera_id: Optional[str] = None,
                                 time_window: timedelta = timedelta(hours=24)
                                 ) -> Dict[str, List]:
        """Get vehicle detection trends
        
        Args:
            camera_id: Optional camera ID to filter by
            time_window: Time window to analyze
        
        Returns:
            Dictionary containing trend data
        """
        end_time = datetime.now()
        start_time = end_time - time_window
        
        try:
            # Get recent detections
            detections = await self.db_service.get_recent_detections(
                camera_id=camera_id if camera_id else "all",
                limit=1000
            )
            
            # Filter by time window
            detections = [
                d for d in detections
                if start_time <= d['timestamp'] <= end_time
            ]
            
            # Calculate trends
            vehicle_types = defaultdict(int)
            hourly_counts = defaultdict(int)
            confidence_sum = defaultdict(float)
            confidence_count = defaultdict(int)
            
            for detection in detections:
                vehicle_type = detection['vehicle_type']
                hour = detection['timestamp'].replace(
                    minute=0, second=0, microsecond=0
                )
                
                vehicle_types[vehicle_type] += 1
                hourly_counts[hour] += 1
                confidence_sum[vehicle_type] += detection['confidence']
                confidence_count[vehicle_type] += 1
            
            # Prepare results
            return {
                'total_detections': len(detections),
                'vehicle_distribution': dict(vehicle_types),
                'hourly_counts': dict(hourly_counts),
                'average_confidence': {
                    vtype: confidence_sum[vtype] / confidence_count[vtype]
                    for vtype in vehicle_types
                },
                'time_range': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting detection trends: {e}")
            return {
                'error': str(e),
                'total_detections': 0,
                'vehicle_distribution': {},
                'hourly_counts': {},
                'average_confidence': {}
            }
    
    async def cleanup(self) -> None:
        """Cleanup analytics service resources"""
        self._is_running = False
        
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        # Final flush of all metrics
        for metric_type in list(self._metrics_buffer.keys()):
            await self._flush_metric_type(metric_type)
        
        logger.info("Analytics service cleaned up") 
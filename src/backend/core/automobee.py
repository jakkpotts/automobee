import asyncio
from typing import Dict, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class SystemComponents:
    """Container for all system components."""
    state_manager: 'StateManager'
    websocket_server: 'DashboardWebSocket'
    db_service: 'DatabaseService'
    analytics_service: 'AnalyticsService'
    detector: 'AsyncVehicleDetector'
    stream_manager: 'AsyncStreamManager'
    config_manager: 'CameraConfigManager'
    performance_monitor: 'PerformanceMonitor'

class AutomoBee:
    """Main controller class for the AutomoBee Vehicle Detection System"""
    
    def __init__(self, db_url: str):
        """Initialize AutomoBee system.
        
        Args:
            db_url: Database connection URL
        """
        self.db_url = db_url
        self.components: Optional[SystemComponents] = None
        self._cleanup_tasks: List[asyncio.Task] = []
        self._is_shutting_down = False
        
    async def initialize(self):
        """Initialize all system components in the correct order"""
        try:
            # Initialize database services first
            db_service = DatabaseService(self.db_url)
            analytics_service = AnalyticsService(db_service)
            
            # Initialize state management
            state_manager = StateManager()
            await state_manager.initialize()
            
            # Initialize config management
            config_manager = CameraConfigManager()
            await config_manager.initialize()
            
            # Load camera configuration into state
            camera_config = config_manager.get_config()
            state_manager.load_camera_config(camera_config)
            
            # Initialize WebSocket server
            websocket_server = DashboardWebSocket(state_manager)
            await websocket_server.start()
            
            # Initialize detector with optimized settings
            detector = AsyncVehicleDetector(
                state_manager=state_manager,
                db_service=db_service,
                analytics_service=analytics_service,
                batch_size=4,
                confidence_threshold=0.65
            )
            
            # Initialize stream manager
            stream_manager = AsyncStreamManager(
                state_manager=state_manager,
                detector=detector,
                max_concurrent_streams=8
            )
            
            # Initialize performance monitor
            performance_monitor = PerformanceMonitor(
                state_manager=state_manager,
                detector=detector,
                stream_manager=stream_manager
            )
            
            self.components = SystemComponents(
                state_manager=state_manager,
                websocket_server=websocket_server,
                db_service=db_service,
                analytics_service=analytics_service,
                detector=detector,
                stream_manager=stream_manager,
                config_manager=config_manager,
                performance_monitor=performance_monitor
            )
            
            # Initialize streams from config
            await self._initialize_streams(camera_config)
            
            # Start monitoring
            await self.components.performance_monitor.start_monitoring()
            
            # Start monitoring tasks
            self._start_monitoring_tasks()
            
            logger.info("AutomoBee system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AutomoBee system: {e}")
            await self.cleanup()
            raise
            
    async def _initialize_streams(self, camera_config: Dict):
        """Initialize video streams from configuration"""
        for camera_id, config in camera_config.items():
            try:
                await self.components.stream_manager.add_stream(
                    camera_id=camera_id,
                    url=config['stream_url'],
                    fps=config.get('fps', 30)
                )
            except Exception as e:
                logger.error(f"Failed to initialize stream {camera_id}: {e}")
    
    def _start_monitoring_tasks(self):
        """Start background tasks for system monitoring"""
        tasks = [
            self._monitor_system_health(),
            self._monitor_performance_metrics(),
            self._cleanup_stale_resources()
        ]
        
        for task in tasks:
            task_obj = asyncio.create_task(task)
            self._cleanup_tasks.append(task_obj)
    
    async def _monitor_system_health(self):
        """Monitor overall system health and component status"""
        while not self._is_shutting_down:
            try:
                if self.components:
                    # Get optimization stats
                    optimization_stats = self.components.performance_monitor.get_optimization_stats()
                    
                    # Prepare health metrics
                    health_metrics = {
                        "websocket_connections": len(self.components.websocket_server.clients),
                        "active_detections": len(self.components.stream_manager.active_streams),
                        "memory_usage": self.components.state_manager.get_memory_usage(),
                        "gpu_utilization": await self.components.detector.get_gpu_utilization(),
                        "optimization": optimization_stats
                    }
                    
                    # Update and broadcast metrics
                    self.components.state_manager.update_health_metrics(health_metrics)
                    await self.components.websocket_server.broadcast_health_update(health_metrics)
                    
                await asyncio.sleep(30)
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _monitor_performance_metrics(self):
        """Collect and store system performance metrics"""
        while not self._is_shutting_down:
            try:
                if self.components:
                    # Get optimization stats
                    optimization_stats = self.components.performance_monitor.get_optimization_stats()
                    
                    # Prepare performance metrics
                    metrics = {
                        "detection_rate": self.components.detector.get_detection_rate(),
                        "processing_latency": self.components.detector.get_processing_latency(),
                        "classification_confidence": self.components.detector.get_avg_confidence(),
                        "optimization": optimization_stats
                    }
                    
                    # Store metrics
                    await self.components.analytics_service.store_metrics(metrics)
                    
                await asyncio.sleep(60)
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _cleanup_stale_resources(self):
        """Periodically cleanup stale resources"""
        while not self._is_shutting_down:
            try:
                if self.components:
                    await self.components.state_manager.cleanup_stale_data()
                await asyncio.sleep(300)
            except Exception as e:
                logger.error(f"Resource cleanup error: {e}")
                await asyncio.sleep(5)
    
    async def cleanup(self):
        """Cleanup all system resources"""
        self._is_shutting_down = True
        
        if self._cleanup_tasks:
            for task in self._cleanup_tasks:
                task.cancel()
            await asyncio.gather(*self._cleanup_tasks, return_exceptions=True)
        
        if self.components:
            # Stop performance monitoring
            await self.components.performance_monitor.stop_monitoring()
            
            # Cleanup in reverse order of initialization
            await self.components.stream_manager.cleanup()
            await self.components.detector.cleanup()
            await self.components.websocket_server.shutdown()
            await self.components.state_manager.cleanup()
            
        logger.info("AutomoBee system shutdown complete")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
        
    def run(self):
        """Get async context manager for running the system."""
        return self 
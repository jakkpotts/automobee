import asyncio
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime
import time
from .detection_types import DetectionQueueItem, TargetVehicleConfig, QueueMetrics
from .device_manager import DeviceManager
from ..utils.error_manager import ErrorManager, ErrorCategory, ErrorSeverity

logger = logging.getLogger(__name__)

class DualQueueProcessor:
    """Manages dual-queue processing for standard and priority detections."""
    
    def __init__(self, model_manager: 'ModelManager', device_manager: DeviceManager):
        self.priority_queue = asyncio.PriorityQueue()
        self.standard_queue = asyncio.PriorityQueue()
        self.model_manager = model_manager
        self.device_manager = device_manager
        self.target_configs: Dict[str, TargetVehicleConfig] = {}
        self.metrics = QueueMetrics()
        self.error_manager = ErrorManager()
        self._processing_task: Optional[asyncio.Task] = None
        self._running = False
        
    async def start(self):
        """Start the queue processor."""
        self._running = True
        self._processing_task = asyncio.create_task(self._process_queues())
        
    async def stop(self):
        """Stop the queue processor."""
        self._running = False
        if self._processing_task:
            await self._processing_task
            
    def add_target_config(self, config: TargetVehicleConfig):
        """Add or update a target vehicle configuration."""
        self.target_configs[config.id] = config
        
    def remove_target_config(self, config_id: str):
        """Remove a target vehicle configuration."""
        self.target_configs.pop(config_id, None)
        
    async def process_detection(self, detection: DetectionQueueItem):
        """Process a new detection and route to appropriate queue."""
        try:
            priority = self._calculate_priority(detection)
            detection.priority = priority
            
            if priority == 0:  # High priority
                await self.priority_queue.put((0, detection))
            else:
                await self.standard_queue.put((priority, detection))
                
            self._update_metrics()
            
        except Exception as e:
            await self.error_manager.handle_error(
                e, ErrorCategory.PROCESSING, ErrorSeverity.MEDIUM
            )
            
    def _calculate_priority(self, detection: DetectionQueueItem) -> int:
        """Calculate processing priority based on detection and target configs."""
        if not self.target_configs:
            return 1  # Standard priority if no targets
            
        for config in self.target_configs.values():
            if not config.active:
                continue
                
            if (detection.initial_type == config.vehicle_type and 
                detection.confidence >= config.pre_screening_threshold):
                detection.target_match = True
                return 0  # High priority
                
        return 1  # Standard priority
        
    async def _process_queues(self):
        """Main processing loop for both queues."""
        while self._running:
            try:
                # Process priority queue first
                while not self.priority_queue.empty():
                    start_time = time.time()
                    _, detection = await self.priority_queue.get()
                    await self._process_detection(detection, is_priority=True)
                    self.metrics.priority_processing_time += time.time() - start_time
                    
                # Then process standard queue
                if not self.standard_queue.empty():
                    start_time = time.time()
                    _, detection = await self.standard_queue.get()
                    await self._process_detection(detection, is_priority=False)
                    self.metrics.standard_processing_time += time.time() - start_time
                    
                self._update_metrics()
                await asyncio.sleep(0.01)  # Prevent CPU hogging
                
            except Exception as e:
                await self.error_manager.handle_error(
                    e, ErrorCategory.PROCESSING, ErrorSeverity.HIGH
                )
                await asyncio.sleep(1)  # Back off on error
                
    async def _process_detection(self, detection: DetectionQueueItem, is_priority: bool):
        """Process a single detection."""
        try:
            # Get appropriate model and thresholds
            model_config = self._get_model_config(detection, is_priority)
            
            # Process with model
            async with self.model_manager.processing_locks[model_config.model_name]:
                result = await self.model_manager.process_frame(
                    detection.frame,
                    model_config,
                    detection.bbox
                )
                
            # Update metrics
            if is_priority:
                self.metrics.priority_batch_size += 1
            else:
                self.metrics.standard_batch_size += 1
                
            return result
            
        except Exception as e:
            await self.error_manager.handle_error(
                e, ErrorCategory.PROCESSING, ErrorSeverity.MEDIUM,
                {'detection_id': detection.frame_id}
            )
            return None
            
    def _update_metrics(self):
        """Update queue metrics."""
        self.metrics.priority_queue_size = self.priority_queue.qsize()
        self.metrics.standard_queue_size = self.standard_queue.qsize()
        self.metrics.last_update = datetime.now()
        
        total_processed = (self.metrics.priority_batch_size + 
                         self.metrics.standard_batch_size)
        if total_processed > 0:
            self.metrics.target_match_rate = (
                self.metrics.priority_batch_size / total_processed
            ) 
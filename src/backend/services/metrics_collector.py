import asyncio
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import psutil
import torch
import numpy as np
from ..state_manager import StateManager
from ..detection.detector import AsyncVehicleDetector
from ..streams.manager import AsyncStreamManager
from ...logger_config import logger

@dataclass
class ResourceMetrics:
    """System resource utilization metrics"""
    cpu_percent: float
    memory_percent: float
    gpu_utilization: float
    gpu_memory: float
    disk_io: Dict[str, float]
    network_io: Dict[str, float]

@dataclass
class PerformanceMetrics:
    """Detection and processing performance metrics"""
    detection_rate: float
    processing_latency: float
    batch_efficiency: float
    classification_confidence: float
    frame_drop_rate: float
    stream_health: Dict[str, float]

@dataclass
class OptimizationConfig:
    """Dynamic optimization configuration"""
    batch_size: int = 4
    frame_skip: int = 2
    buffer_size: int = 30
    max_concurrent_streams: int = 8
    confidence_threshold: float = 0.65
    max_batch_latency: float = 0.1

class PerformanceMonitor:
    """Monitors and optimizes system performance"""
    
    def __init__(
        self,
        state_manager: StateManager,
        detector: AsyncVehicleDetector,
        stream_manager: AsyncStreamManager,
        update_interval: int = 30,
        history_window: int = 300  # 5 minutes
    ):
        self.state_manager = state_manager
        self.detector = detector
        self.stream_manager = stream_manager
        self.update_interval = update_interval
        
        # Performance history
        self._resource_history: List[ResourceMetrics] = []
        self._performance_history: List[PerformanceMetrics] = []
        self._history_window = history_window
        self._last_update = time.time()
        
        # Optimization state
        self.current_config = OptimizationConfig()
        self._optimization_lock = asyncio.Lock()
        
    async def start_monitoring(self):
        """Start performance monitoring tasks"""
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop performance monitoring"""
        if hasattr(self, '_monitor_task'):
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Performance monitoring stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while True:
            try:
                # Collect metrics
                resource_metrics = await self._collect_resource_metrics()
                performance_metrics = await self._collect_performance_metrics()
                
                # Update history
                self._update_history(resource_metrics, performance_metrics)
                
                # Analyze and optimize
                await self._analyze_and_optimize()
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(5)
    
    async def _collect_resource_metrics(self) -> ResourceMetrics:
        """Collect system resource metrics"""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.Process().memory_percent()
            
            # GPU metrics
            gpu_utilization = await self.detector.get_gpu_utilization()
            gpu_memory = self._get_gpu_memory_usage() if torch.cuda.is_available() else 0.0
            
            # IO metrics
            disk_io = self._get_disk_io_rates()
            network_io = self._get_network_io_rates()
            
            return ResourceMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                gpu_utilization=gpu_utilization,
                gpu_memory=gpu_memory,
                disk_io=disk_io,
                network_io=network_io
            )
        except Exception as e:
            logger.error(f"Error collecting resource metrics: {e}")
            raise
    
    async def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect detection and processing performance metrics"""
        try:
            # Detection metrics
            detection_rate = self.detector.get_detection_rate()
            processing_latency = self.detector.get_processing_latency()
            classification_confidence = self.detector.get_avg_confidence()
            
            # Stream metrics
            stream_health = self._calculate_stream_health()
            frame_drop_rate = self._calculate_frame_drop_rate()
            
            # Batch efficiency
            batch_efficiency = self._calculate_batch_efficiency()
            
            return PerformanceMetrics(
                detection_rate=detection_rate,
                processing_latency=processing_latency,
                batch_efficiency=batch_efficiency,
                classification_confidence=classification_confidence,
                frame_drop_rate=frame_drop_rate,
                stream_health=stream_health
            )
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
            raise
    
    def _get_gpu_memory_usage(self) -> float:
        """Get GPU memory usage percentage"""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            return 0.0
        except Exception:
            return 0.0
    
    def _get_disk_io_rates(self) -> Dict[str, float]:
        """Get disk IO rates"""
        try:
            disk_io = psutil.disk_io_counters()
            return {
                'read_bytes': disk_io.read_bytes,
                'write_bytes': disk_io.write_bytes
            }
        except Exception:
            return {'read_bytes': 0, 'write_bytes': 0}
    
    def _get_network_io_rates(self) -> Dict[str, float]:
        """Get network IO rates"""
        try:
            net_io = psutil.net_io_counters()
            return {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv
            }
        except Exception:
            return {'bytes_sent': 0, 'bytes_recv': 0}
    
    def _calculate_stream_health(self) -> Dict[str, float]:
        """Calculate health metrics for each stream"""
        health_metrics = {}
        for camera_id, config in self.stream_manager.active_streams.items():
            try:
                buffer = self.stream_manager._frame_buffers[camera_id]
                buffer_usage = len(buffer) / config.buffer_size
                last_processed = self.stream_manager._last_processed[camera_id]
                processing_delay = time.time() - last_processed
                
                health_metrics[camera_id] = min(
                    1.0,
                    (1.0 - buffer_usage) * 0.5 + 
                    (1.0 / (1.0 + processing_delay)) * 0.5
                )
            except Exception:
                health_metrics[camera_id] = 0.0
        
        return health_metrics
    
    def _calculate_frame_drop_rate(self) -> float:
        """Calculate overall frame drop rate"""
        total_frames = 0
        dropped_frames = 0
        
        for camera_id, config in self.stream_manager.active_streams.items():
            expected_frames = (time.time() - self.stream_manager._last_processed[camera_id]) * config.fps
            actual_frames = len(self.stream_manager._frame_buffers[camera_id])
            
            total_frames += expected_frames
            dropped_frames += max(0, expected_frames - actual_frames)
        
        return dropped_frames / total_frames if total_frames > 0 else 0.0
    
    def _calculate_batch_efficiency(self) -> float:
        """Calculate batch processing efficiency"""
        if not self.detector._processing_times:
            return 0.0
            
        avg_batch_size = np.mean(self.detector._detection_counts)
        target_batch_size = self.current_config.batch_size
        
        return min(1.0, avg_batch_size / target_batch_size)
    
    def _update_history(self, resource_metrics: ResourceMetrics, 
                       performance_metrics: PerformanceMetrics):
        """Update metrics history"""
        current_time = time.time()
        
        # Add new metrics
        self._resource_history.append(resource_metrics)
        self._performance_history.append(performance_metrics)
        
        # Remove old metrics
        cutoff_time = current_time - self._history_window
        while self._resource_history and self._last_update < cutoff_time:
            self._resource_history.pop(0)
            self._performance_history.pop(0)
            self._last_update = current_time
    
    async def _analyze_and_optimize(self):
        """Analyze metrics and optimize system configuration"""
        async with self._optimization_lock:
            try:
                # Get latest metrics
                if not self._resource_history or not self._performance_history:
                    return
                    
                resources = self._resource_history[-1]
                performance = self._performance_history[-1]
                
                # Optimize batch processing
                await self._optimize_batch_processing(resources, performance)
                
                # Optimize stream handling
                await self._optimize_stream_handling(resources, performance)
                
                # Optimize detection parameters
                await self._optimize_detection_params(resources, performance)
                
            except Exception as e:
                logger.error(f"Error in performance optimization: {e}")
    
    async def _optimize_batch_processing(self, resources: ResourceMetrics,
                                      performance: PerformanceMetrics):
        """Optimize batch processing parameters"""
        try:
            # Adjust batch size based on GPU utilization and processing latency
            if resources.gpu_utilization < 0.7 and performance.processing_latency < 0.1:
                self.current_config.batch_size = min(8, self.current_config.batch_size + 1)
            elif resources.gpu_utilization > 0.9 or performance.processing_latency > 0.2:
                self.current_config.batch_size = max(2, self.current_config.batch_size - 1)
            
            # Update detector configuration
            self.detector.batch_size = self.current_config.batch_size
            
        except Exception as e:
            logger.error(f"Error optimizing batch processing: {e}")
    
    async def _optimize_stream_handling(self, resources: ResourceMetrics,
                                     performance: PerformanceMetrics):
        """Optimize stream handling parameters"""
        try:
            # Adjust frame skip based on processing capacity
            if performance.frame_drop_rate > 0.2:
                self.current_config.frame_skip = min(4, self.current_config.frame_skip + 1)
            elif performance.frame_drop_rate < 0.1:
                self.current_config.frame_skip = max(1, self.current_config.frame_skip - 1)
            
            # Update stream configurations
            for camera_id, config in self.stream_manager.active_streams.items():
                config.frame_skip = self.current_config.frame_skip
            
        except Exception as e:
            logger.error(f"Error optimizing stream handling: {e}")
    
    async def _optimize_detection_params(self, resources: ResourceMetrics,
                                      performance: PerformanceMetrics):
        """Optimize detection parameters"""
        try:
            # Adjust confidence threshold based on classification confidence
            if performance.classification_confidence > 0.8:
                self.current_config.confidence_threshold = min(
                    0.8,
                    self.current_config.confidence_threshold + 0.05
                )
            elif performance.classification_confidence < 0.6:
                self.current_config.confidence_threshold = max(
                    0.5,
                    self.current_config.confidence_threshold - 0.05
                )
            
            # Update detector configuration
            self.detector.confidence_threshold = self.current_config.confidence_threshold
            
        except Exception as e:
            logger.error(f"Error optimizing detection parameters: {e}")
    
    def get_optimization_stats(self) -> Dict:
        """Get current optimization statistics"""
        return {
            "config": vars(self.current_config),
            "performance": {
                "detection_rate": self.detector.get_detection_rate(),
                "processing_latency": self.detector.get_processing_latency(),
                "classification_confidence": self.detector.get_avg_confidence(),
                "frame_drop_rate": self._calculate_frame_drop_rate(),
                "batch_efficiency": self._calculate_batch_efficiency()
            }
        } 
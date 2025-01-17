import asyncio
from typing import Dict, Optional
import logging
import psutil
import time
from datetime import datetime, timedelta
import torch
import numpy as np

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitors system performance and resource utilization"""
    
    def __init__(
        self,
        state_manager: 'StateManager',
        detector: 'AsyncVehicleDetector',
        stream_manager: 'AsyncStreamManager',
        monitoring_interval: float = 5.0  # seconds
    ):
        """Initialize performance monitor
        
        Args:
            state_manager: System state manager
            detector: Vehicle detector service
            stream_manager: Stream manager service
            monitoring_interval: Interval between monitoring checks
        """
        self.state_manager = state_manager
        self.detector = detector
        self.stream_manager = stream_manager
        self.monitoring_interval = monitoring_interval
        
        # System monitoring
        self._process = psutil.Process()
        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_monitoring = False
        
        # Performance metrics
        self._metrics_window = 300  # 5 minutes
        self._cpu_usage: Dict[datetime, float] = {}
        self._memory_usage: Dict[datetime, Dict] = {}
        self._gpu_metrics: Dict[datetime, Dict] = {}
        self._detection_latency: Dict[datetime, float] = {}
    
    async def start_monitoring(self) -> None:
        """Start performance monitoring"""
        if self._monitoring_task is not None:
            logger.warning("Performance monitoring already started")
            return
        
        self._is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitor_performance())
        logger.info("Performance monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop performance monitoring"""
        self._is_monitoring = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
        
        logger.info("Performance monitoring stopped")
    
    async def _monitor_performance(self) -> None:
        """Main monitoring loop"""
        while self._is_monitoring:
            try:
                # Collect metrics
                timestamp = datetime.now()
                
                # System metrics
                cpu_percent = self._process.cpu_percent()
                memory_info = self._process.memory_info()
                
                self._cpu_usage[timestamp] = cpu_percent
                self._memory_usage[timestamp] = {
                    'rss': memory_info.rss / 1024 / 1024,  # MB
                    'vms': memory_info.vms / 1024 / 1024,  # MB
                    'percent': self._process.memory_percent()
                }
                
                # GPU metrics if available
                gpu_metrics = await self.detector.get_gpu_utilization()
                if gpu_metrics:
                    self._gpu_metrics[timestamp] = gpu_metrics
                
                # Detection performance
                self._detection_latency[timestamp] = self.detector.get_processing_latency()
                
                # Cleanup old metrics
                self._cleanup_old_metrics()
                
                # Update state
                self._update_state()
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(self.monitoring_interval * 2)
    
    def _cleanup_old_metrics(self) -> None:
        """Remove metrics older than the window"""
        cutoff_time = datetime.now() - timedelta(seconds=self._metrics_window)
        
        self._cpu_usage = {
            t: v for t, v in self._cpu_usage.items()
            if t > cutoff_time
        }
        self._memory_usage = {
            t: v for t, v in self._memory_usage.items()
            if t > cutoff_time
        }
        self._gpu_metrics = {
            t: v for t, v in self._gpu_metrics.items()
            if t > cutoff_time
        }
        self._detection_latency = {
            t: v for t, v in self._detection_latency.items()
            if t > cutoff_time
        }
    
    def _update_state(self) -> None:
        """Update system state with current metrics"""
        if not self._cpu_usage:
            return
        
        # Calculate averages
        avg_cpu = sum(self._cpu_usage.values()) / len(self._cpu_usage)
        avg_memory = {
            'rss': np.mean([m['rss'] for m in self._memory_usage.values()]),
            'vms': np.mean([m['vms'] for m in self._memory_usage.values()]),
            'percent': np.mean([m['percent'] for m in self._memory_usage.values()])
        }
        avg_latency = sum(self._detection_latency.values()) / len(self._detection_latency)
        
        # Get latest GPU metrics
        latest_gpu = next(iter(self._gpu_metrics.values())) if self._gpu_metrics else {}
        
        # Update state
        self.state_manager.update_health_metrics({
            'cpu_usage': avg_cpu,
            'memory_usage': avg_memory,
            'gpu_metrics': latest_gpu,
            'detection_latency': avg_latency,
            'active_streams': self.stream_manager.active_stream_count,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_optimization_stats(self) -> Dict:
        """Get system optimization statistics
        
        Returns:
            Dictionary containing optimization metrics
        """
        if not self._cpu_usage:
            return {}
        
        # Calculate statistics
        stats = {
            'cpu': {
                'current': next(iter(self._cpu_usage.values())),
                'avg': sum(self._cpu_usage.values()) / len(self._cpu_usage),
                'max': max(self._cpu_usage.values())
            },
            'memory': {
                'current_mb': next(iter(self._memory_usage.values()))['rss'],
                'avg_mb': np.mean([m['rss'] for m in self._memory_usage.values()]),
                'max_mb': max(m['rss'] for m in self._memory_usage.values())
            },
            'detection': {
                'current_latency': next(iter(self._detection_latency.values())),
                'avg_latency': sum(self._detection_latency.values()) / len(self._detection_latency),
                'max_latency': max(self._detection_latency.values())
            }
        }
        
        # Add GPU stats if available
        if self._gpu_metrics:
            latest_gpu = next(iter(self._gpu_metrics.values()))
            stats['gpu'] = {
                'memory_used_mb': latest_gpu.get('memory_used_mb', 0),
                'memory_cached_mb': latest_gpu.get('memory_cached_mb', 0),
                'device': latest_gpu.get('device_name', 'cpu')
            }
        
        return stats
    
    async def cleanup(self) -> None:
        """Cleanup monitor resources"""
        await self.stop_monitoring()
        
        # Clear metrics
        self._cpu_usage.clear()
        self._memory_usage.clear()
        self._gpu_metrics.clear()
        self._detection_latency.clear()
        
        logger.info("Performance monitor cleaned up") 
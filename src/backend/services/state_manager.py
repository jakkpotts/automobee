import asyncio
from typing import Dict, Any, Optional
import logging
import psutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class SystemState:
    """Container for system-wide state"""
    camera_config: Dict[str, Dict] = field(default_factory=dict)
    health_metrics: Dict[str, Any] = field(default_factory=dict)
    active_streams: Dict[str, Dict] = field(default_factory=dict)
    detection_stats: Dict[str, Dict] = field(default_factory=dict)
    last_update: datetime = field(default_factory=datetime.now)

class StateManager:
    """Manages system-wide state and configuration"""
    
    def __init__(self):
        self._state = SystemState()
        self._state_lock = asyncio.Lock()
        self._process = psutil.Process()
        self._stale_threshold = timedelta(minutes=30)
    
    async def initialize(self) -> None:
        """Initialize the state manager"""
        logger.info("Initializing state manager...")
        async with self._state_lock:
            self._state.last_update = datetime.now()
    
    def load_camera_config(self, config: Dict) -> None:
        """Load camera configuration into state
        
        Args:
            config: Dictionary of camera configurations
        """
        self._state.camera_config = config
        self._state.last_update = datetime.now()
        logger.info(f"Loaded configuration for {len(config)} cameras")
    
    def update_health_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update system health metrics
        
        Args:
            metrics: Dictionary of health metrics
        """
        self._state.health_metrics = metrics
        self._state.last_update = datetime.now()
    
    def update_stream_state(self, stream_id: str, state: Dict) -> None:
        """Update state for a specific stream
        
        Args:
            stream_id: ID of the stream
            state: Stream state information
        """
        self._state.active_streams[stream_id] = {
            **state,
            "last_update": datetime.now()
        }
    
    def update_detection_stats(self, stream_id: str, stats: Dict) -> None:
        """Update detection statistics for a stream
        
        Args:
            stream_id: ID of the stream
            stats: Detection statistics
        """
        self._state.detection_stats[stream_id] = {
            **stats,
            "last_update": datetime.now()
        }
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics
        
        Returns:
            Dictionary containing memory usage metrics
        """
        memory_info = self._process.memory_info()
        return {
            "rss": memory_info.rss / 1024 / 1024,  # MB
            "vms": memory_info.vms / 1024 / 1024,  # MB
            "percent": self._process.memory_percent()
        }
    
    def get_stream_state(self, stream_id: str) -> Optional[Dict]:
        """Get state for a specific stream
        
        Args:
            stream_id: ID of the stream
        
        Returns:
            Stream state if exists, None otherwise
        """
        return self._state.active_streams.get(stream_id)
    
    def get_all_stream_states(self) -> Dict[str, Dict]:
        """Get states for all active streams
        
        Returns:
            Dictionary of stream states
        """
        return self._state.active_streams
    
    def get_detection_stats(self, stream_id: str) -> Optional[Dict]:
        """Get detection statistics for a stream
        
        Args:
            stream_id: ID of the stream
        
        Returns:
            Detection statistics if exists, None otherwise
        """
        return self._state.detection_stats.get(stream_id)
    
    async def cleanup_stale_data(self) -> None:
        """Remove stale data from state"""
        async with self._state_lock:
            current_time = datetime.now()
            
            # Cleanup stale stream states
            stale_streams = [
                stream_id for stream_id, state in self._state.active_streams.items()
                if current_time - state["last_update"] > self._stale_threshold
            ]
            for stream_id in stale_streams:
                del self._state.active_streams[stream_id]
                logger.info(f"Removed stale stream state: {stream_id}")
            
            # Cleanup stale detection stats
            stale_stats = [
                stream_id for stream_id, stats in self._state.detection_stats.items()
                if current_time - stats["last_update"] > self._stale_threshold
            ]
            for stream_id in stale_stats:
                del self._state.detection_stats[stream_id]
                logger.info(f"Removed stale detection stats: {stream_id}")
    
    async def cleanup(self) -> None:
        """Cleanup state manager resources"""
        async with self._state_lock:
            self._state = SystemState()
            logger.info("State manager cleaned up") 
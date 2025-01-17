import asyncio
import time
from typing import Dict, Set, List, Optional, Any
from collections import deque
from dataclasses import dataclass, field
import psutil
from websockets.server import WebSocketServerProtocol
from logger_config import logger

@dataclass
class Zone:
    """Represents a monitored zone"""
    id: str
    name: str
    camera_ids: List[str]
    coordinates: List[Dict[str, float]]
    active: bool = True
    subscribers: Set[WebSocketServerProtocol] = field(default_factory=set)

@dataclass
class Detection:
    """Represents a vehicle detection event"""
    id: str
    timestamp: float
    zone_id: str
    camera_id: str
    vehicle_type: str
    confidence: float
    coordinates: Dict[str, float]
    matched_target: bool = False

class StateManager:
    """Manages application state with memory optimization"""
    
    def __init__(self, max_history: int = 1000):
        self.zones: Dict[str, Zone] = {}
        self.camera_config: Dict[str, Dict] = {}
        self.health_metrics: Dict[str, Any] = {}
        self._detection_history: deque = deque(maxlen=max_history)
        self._last_cleanup: float = time.time()
        self._cleanup_interval: int = 300  # 5 minutes
        
    async def initialize(self):
        """Initialize state manager"""
        try:
            # Initialize empty state
            self.health_metrics = {
                "websocket_connections": 0,
                "active_detections": 0,
                "memory_usage": 0,
                "gpu_utilization": 0
            }
            logger.info("State manager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize state manager: {e}")
            raise
    
    def load_camera_config(self, config: Dict):
        """Load camera configuration"""
        self.camera_config = config
        # Initialize zones from camera config
        for camera_id, camera_data in config.items():
            for zone_data in camera_data.get("zones", []):
                self.add_zone(
                    zone_data["id"],
                    zone_data["name"],
                    [camera_id],
                    zone_data["coordinates"]
                )
    
    def add_zone(self, zone_id: str, name: str, camera_ids: List[str],
                coordinates: List[Dict[str, float]]) -> Zone:
        """Add a new monitoring zone"""
        zone = Zone(
            id=zone_id,
            name=name,
            camera_ids=camera_ids,
            coordinates=coordinates
        )
        self.zones[zone_id] = zone
        return zone
    
    def get_active_zones(self) -> Dict[str, Zone]:
        """Get all active zones"""
        return {
            zone_id: zone
            for zone_id, zone in self.zones.items()
            if zone.active
        }
    
    def subscribe_to_zone(self, client: WebSocketServerProtocol, zone_id: str):
        """Subscribe a client to zone updates"""
        if zone_id in self.zones:
            self.zones[zone_id].subscribers.add(client)
    
    def unsubscribe_from_zone(self, client: WebSocketServerProtocol, zone_id: str):
        """Unsubscribe a client from zone updates"""
        if zone_id in self.zones:
            self.zones[zone_id].subscribers.discard(client)
    
    def get_zone_subscribers(self, zone_id: str) -> Set[WebSocketServerProtocol]:
        """Get all subscribers for a zone"""
        return self.zones.get(zone_id, Zone(zone_id, "", [], [])).subscribers
    
    async def update_camera_config(self, camera_id: str, config: Dict) -> bool:
        """Update camera configuration"""
        try:
            if camera_id in self.camera_config:
                self.camera_config[camera_id].update(config)
                return True
            return False
        except Exception as e:
            logger.error(f"Error updating camera config: {e}")
            return False
    
    def add_detection(self, detection: Detection):
        """Add a new detection event"""
        self._detection_history.append(detection)
        
    def get_recent_detections(self, limit: int = 50, zone_id: Optional[str] = None) -> List[Detection]:
        """Get recent detections, optionally filtered by zone"""
        detections = list(self._detection_history)
        if zone_id:
            detections = [d for d in detections if d.zone_id == zone_id]
        return detections[-limit:]
    
    def update_health_metrics(self, metrics: Dict[str, Any]):
        """Update system health metrics"""
        self.health_metrics.update(metrics)
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get current health metrics"""
        return self.health_metrics
    
    def get_memory_usage(self) -> float:
        """Get current memory usage percentage"""
        process = psutil.Process()
        return process.memory_percent()
    
    def get_camera_config(self) -> Dict[str, Dict]:
        """Get current camera configuration"""
        return self.camera_config
    
    async def cleanup_stale_data(self):
        """Clean up stale data periodically"""
        current_time = time.time()
        if current_time - self._last_cleanup < self._cleanup_interval:
            return
            
        try:
            # Clean up disconnected subscribers
            for zone in self.zones.values():
                # Create a new set with only active connections
                zone.subscribers = {
                    client for client in zone.subscribers
                    if not client.closed
                }
            
            # Update last cleanup time
            self._last_cleanup = current_time
            
        except Exception as e:
            logger.error(f"Error during state cleanup: {e}")
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Clear all subscribers
            for zone in self.zones.values():
                zone.subscribers.clear()
            
            # Clear detection history
            self._detection_history.clear()
            
            logger.info("State manager cleanup complete")
        except Exception as e:
            logger.error(f"Error during state manager cleanup: {e}")
            raise 
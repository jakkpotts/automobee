import logging
from enum import Enum
from typing import Dict, List, Optional
from ..camera.types import CameraStream, StreamStatus  # Import from types instead of stream_manager
from dataclasses import dataclass

class ZoneType(Enum):
    """Enum representing different types of zones."""
    DEFAULT = "default"
    ENTRY = "entry"
    EXIT = "exit"
    MONITORING = "monitoring"

@dataclass
class ZoneDefinition:
    """Defines a monitoring zone."""
    name: ZoneType
    center_lat: float
    center_lng: float
    radius_km: float
    priority: int = 1

class ZoneManager:
    """Manages detection zones and their states."""
    
    ZONE_DEFINITIONS = [
        ZoneDefinition(ZoneType.ENTRY, 36.1699, -115.1398, 2.0),
        ZoneDefinition(ZoneType.EXIT, 36.1700, -115.1399, 2.0),
        ZoneDefinition(ZoneType.MONITORING, 36.1701, -115.1400, 3.0)
    ]
    
    def __init__(self):
        self.zones: Dict[str, List[str]] = {}  # zone_id -> list of stream_ids
        self.stream_assignments: Dict[str, ZoneType] = {}
        self.stream_manager = None
        self.logger = logging.getLogger(__name__)
        
    def get_zone_streams(self, zone_type: ZoneType) -> List[str]:
        """Get list of stream IDs in a zone."""
        return self.zones.get(zone_type.value, []) 
    
    async def initialize(self):
        """Initialize the zone manager."""
        try:
            # Initialize zones
            for zone_type in ZoneType:
                self.zones[zone_type.value] = []
            self.logger.info("Zone manager initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize zone manager: {str(e)}")
            raise
            
    def assign_stream_to_zone(self, stream: CameraStream) -> ZoneType:
        """Assign a stream to a zone based on location."""
        # Simple assignment logic - can be enhanced later
        zone_type = ZoneType.DEFAULT
        self.zones[zone_type].append(stream)
        self.stream_assignments[stream.id] = zone_type
        return zone_type 
# camera_config.py
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path
import json
import time
from datetime import datetime, timezone
from ..logger_config import logger

@dataclass
class Zone:
    """Represents a monitored zone in a camera's view"""
    id: str
    name: str
    coordinates: List[Dict[str, float]]  # List of lat/lng points defining the zone
    active: bool = True

@dataclass
class CameraLocation:
    """Represents a camera location and configuration"""
    id: str
    name: str
    lat: float
    lng: float
    stream_url: str
    fps: int = 30
    direction: int = 0
    roadway: Optional[str] = None
    zones: List[Zone] = None
    last_updated: Optional[str] = None

class CameraConfigManager:
    """Manages camera configuration and zones"""
    
    def __init__(self, config_path: str = 'config/camera_locations.json'):
        self.config_path = Path(config_path)
        self.config_dir = self.config_path.parent
        self.cameras: Dict[str, CameraLocation] = {}
        self.last_refresh = time.time()
        self._config_check_interval = 300  # 5 minutes
        
    async def initialize(self):
        """Initialize the camera configuration"""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            await self.load_config()
            logger.info("Camera configuration initialized")
        except Exception as e:
            logger.error(f"Error initializing camera config: {e}")
            raise
    
    async def load_config(self) -> bool:
        """Load camera configuration from file"""
        try:
            if not self.config_path.exists():
                logger.warning("Config file not found, will create new one")
                return False
                
            with open(self.config_path) as f:
                data = json.load(f)
                
            self.cameras = {
                camera_id: CameraLocation(
                    id=camera_id,
                    name=cam_data.get('name', camera_id),
                    lat=float(cam_data['lat']),
                    lng=float(cam_data['lng']),
                    stream_url=cam_data['stream_url'],
                    fps=cam_data.get('fps', 30),
                    direction=cam_data.get('direction', 0),
                    roadway=cam_data.get('roadway'),
                    zones=[
                        Zone(
                            id=f"{camera_id}_{i}",
                            name=zone.get('name', f"Zone {i}"),
                            coordinates=zone['coordinates'],
                            active=zone.get('active', True)
                        )
                        for i, zone in enumerate(cam_data.get('zones', []))
                    ],
                    last_updated=datetime.now(timezone.utc).isoformat()
                )
                for camera_id, cam_data in data.items()
                if self._validate_camera_data(cam_data)
            }
            
            self.last_refresh = time.time()
            logger.info(f"Loaded configuration for {len(self.cameras)} cameras")
            return True
            
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return False
    
    def get_config(self) -> Dict[str, CameraLocation]:
        """Get current camera configuration"""
        current_time = time.time()
        if current_time - self.last_refresh > self._config_check_interval:
            asyncio.create_task(self.load_config())
        return self.cameras
    
    def _validate_camera_data(self, data: dict) -> bool:
        """Validate individual camera data"""
        required_fields = {'lat', 'lng', 'name', 'stream_url'}
        if not all(key in data and data[key] is not None for key in required_fields):
            return False
            
        # Validate zones if present
        if 'zones' in data:
            for zone in data['zones']:
                if not self._validate_zone_data(zone):
                    return False
        
        return True
    
    def _validate_zone_data(self, zone: dict) -> bool:
        """Validate zone configuration data"""
        if 'coordinates' not in zone:
            return False
            
        coordinates = zone['coordinates']
        if not isinstance(coordinates, list) or len(coordinates) < 3:
            return False
            
        for point in coordinates:
            if not isinstance(point, dict) or 'lat' not in point or 'lng' not in point:
                return False
        
        return True
    
    async def save_config(self) -> bool:
        """Save current configuration to file"""
        try:
            config_data = {}
            for camera_id, camera in self.cameras.items():
                camera_data = {
                    'name': camera.name,
                    'lat': camera.lat,
                    'lng': camera.lng,
                    'stream_url': camera.stream_url,
                    'fps': camera.fps,
                    'direction': camera.direction,
                    'roadway': camera.roadway,
                    'zones': [
                        {
                            'name': zone.name,
                            'coordinates': zone.coordinates,
                            'active': zone.active
                        }
                        for zone in (camera.zones or [])
                    ]
                }
                config_data[camera_id] = camera_data
            
            with open(self.config_path, 'w') as f:
                json.dump(config_data, f, indent=4)
                
            logger.info("Camera configuration saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False
    
    async def add_zone(self, camera_id: str, name: str, coordinates: List[Dict[str, float]]) -> Optional[Zone]:
        """Add a new zone to a camera"""
        try:
            if camera_id not in self.cameras:
                return None
                
            camera = self.cameras[camera_id]
            if camera.zones is None:
                camera.zones = []
                
            zone = Zone(
                id=f"{camera_id}_{len(camera.zones)}",
                name=name,
                coordinates=coordinates
            )
            camera.zones.append(zone)
            
            await self.save_config()
            return zone
            
        except Exception as e:
            logger.error(f"Error adding zone: {e}")
            return None
    
    async def update_zone(self, camera_id: str, zone_id: str, 
                         coordinates: Optional[List[Dict[str, float]]] = None,
                         active: Optional[bool] = None) -> bool:
        """Update an existing zone"""
        try:
            if camera_id not in self.cameras:
                return False
                
            camera = self.cameras[camera_id]
            if not camera.zones:
                return False
                
            for zone in camera.zones:
                if zone.id == zone_id:
                    if coordinates is not None:
                        zone.coordinates = coordinates
                    if active is not None:
                        zone.active = active
                    await self.save_config()
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error updating zone: {e}")
            return False
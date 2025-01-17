import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional
from ..zones.zone_manager import ZoneManager, ZoneType
from ..processing.scheduler import ProcessingScheduler
from .types import CameraStream, StreamStatus  # Import from types.py

logger = logging.getLogger(__name__)

class StreamManager:
    """Manages camera streams from NV Roads API."""
    
    BASE_URL = "https://www.nvroads.com/List"
    REGION = "Las Vegas Area"
    BATCH_SIZE = 30
    
    def __init__(self):
        self.streams = {}
        self.zone_manager = None
        self.scheduler = ProcessingScheduler()
        self.session = None
        self._running = False
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Initialize stream manager and its components."""
        try:
            self.session = aiohttp.ClientSession()  # Initialize session
            await self.scheduler.initialize()
            self.zone_manager = ZoneManager()
            await self.zone_manager.initialize()
            await self.refresh_streams()  # Fetch initial streams
            self.logger.info("Stream manager initialized successfully")
        except Exception as e:
            await self.cleanup()
            raise

    async def cleanup(self):
        """Cleanup resources."""
        if self.session:
            await self.session.close()
            self.session = None
            
    async def get_total_cameras(self) -> int:
        """Get total number of cameras in the region."""
        query = self._build_query(0, 1)
        async with self.session.get(f"{self.BASE_URL}/GetData/Cameras", params=query) as response:
            data = await response.json()
            return data.get("recordsTotal", 0)
            
    async def refresh_streams(self):
        """Refresh the list of camera streams."""
        try:
            total_cameras = await self.get_total_cameras()
            batches = range(0, total_cameras, self.BATCH_SIZE)
            
            for start in batches:
                await self._fetch_batch(start)
                
            logger.info(f"Successfully refreshed {len(self.streams)} camera streams")
            
        except Exception as e:
            logger.error(f"Error refreshing streams: {str(e)}")
            raise
            
    async def _fetch_batch(self, start: int):
        """Fetch a batch of camera streams."""
        query = self._build_query(start, self.BATCH_SIZE)
        
        async with self.session.get(f"{self.BASE_URL}/GetData/Cameras", params=query) as response:
            data = await response.json()
            
            for camera in data.get("data", []):
                stream = self._parse_camera_data(camera)
                if stream:
                    self.streams[stream.id] = stream
                    # Assign stream to a zone
                    zone = self.zone_manager.assign_stream_to_zone(stream)
                    logger.info(f"Assigned stream {stream.id} to zone {zone.value}")
                    
    def _parse_camera_data(self, camera: dict) -> Optional[CameraStream]:
        """Parse camera data into CameraStream object."""
        try:
            images = camera.get("images", [])
            if not images:
                return None
                
            image = images[0]
            lat_lng = camera.get("latLng", {}).get("geography", {})
            
            return CameraStream(
                id=str(camera.get("DT_RowId")),
                name=camera.get("roadway", ""),
                location=camera.get("location", ""),
                lat=float(lat_lng.get("lat", 0)),
                lng=float(lat_lng.get("lng", 0)),
                video_url=image.get("videoUrl", ""),
                image_url=image.get("imageUrl", ""),
                status=StreamStatus.INITIALIZING
            )
        except Exception as e:
            logger.error(f"Error parsing camera data: {str(e)}")
            return None
            
    def _build_query(self, start: int, length: int) -> dict:
        """Build query parameters for API request."""
        return {
            "start": str(start),
            "length": str(length),
            "region": self.REGION,
            "sortOrder": "asc",
            "search": ""
        } 
import json
import os
from typing import Dict, Optional
import logging
from pathlib import Path
import aiofiles
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

class CameraConfigManager:
    """Manages camera configurations and settings"""
    
    def __init__(self, config_path: str = "config/cameras.json"):
        """Initialize camera configuration manager
        
        Args:
            config_path: Path to camera configuration file
        """
        self.config_path = Path(config_path)
        self._config: Dict = {}
        self._last_load: Optional[datetime] = None
        self._config_lock = asyncio.Lock()
        self._watch_task: Optional[asyncio.Task] = None
        
    async def initialize(self) -> None:
        """Initialize the configuration manager"""
        # Ensure config directory exists
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        # Create default config if it doesn't exist
        if not self.config_path.exists():
            await self._create_default_config()
        
        # Load initial configuration
        await self._load_config()
        
        # Start config file watcher
        self._watch_task = asyncio.create_task(self._watch_config())
        
        logger.info("Camera configuration manager initialized")
    
    async def _create_default_config(self) -> None:
        """Create default configuration file"""
        default_config = {
            "camera_1": {
                "name": "Example Camera",
                "stream_url": "rtsp://example.com/stream1",
                "fps": 30,
                "enabled": True,
                "detection_zones": [
                    {
                        "name": "Zone 1",
                        "points": [[0, 0], [1, 0], [1, 1], [0, 1]],
                        "type": "entry"
                    }
                ],
                "settings": {
                    "confidence_threshold": 0.6,
                    "max_detection_size": 0.8,
                    "min_detection_size": 0.1
                }
            }
        }
        
        async with aiofiles.open(self.config_path, 'w') as f:
            await f.write(json.dumps(default_config, indent=4))
        
        logger.info("Created default camera configuration")
    
    async def _load_config(self) -> None:
        """Load configuration from file"""
        try:
            async with aiofiles.open(self.config_path, 'r') as f:
                content = await f.read()
                async with self._config_lock:
                    self._config = json.loads(content)
                    self._last_load = datetime.now()
                    
            logger.info(f"Loaded configuration for {len(self._config)} cameras")
            
        except Exception as e:
            logger.error(f"Failed to load camera configuration: {e}")
            raise
    
    async def _watch_config(self) -> None:
        """Watch for configuration file changes"""
        last_modified = self.config_path.stat().st_mtime
        
        while True:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                current_modified = self.config_path.stat().st_mtime
                if current_modified > last_modified:
                    logger.info("Configuration file changed, reloading...")
                    await self._load_config()
                    last_modified = current_modified
                    
            except Exception as e:
                logger.error(f"Error watching config file: {e}")
                await asyncio.sleep(30)  # Back off on error
    
    def get_config(self) -> Dict:
        """Get current camera configuration
        
        Returns:
            Dictionary containing camera configurations
        """
        return self._config.copy()
    
    def get_camera_config(self, camera_id: str) -> Optional[Dict]:
        """Get configuration for a specific camera
        
        Args:
            camera_id: ID of the camera
        
        Returns:
            Camera configuration if exists, None otherwise
        """
        return self._config.get(camera_id)
    
    async def update_camera_config(self, camera_id: str, config: Dict) -> None:
        """Update configuration for a specific camera
        
        Args:
            camera_id: ID of the camera
            config: New camera configuration
        """
        async with self._config_lock:
            self._config[camera_id] = config
            
            async with aiofiles.open(self.config_path, 'w') as f:
                await f.write(json.dumps(self._config, indent=4))
            
            self._last_load = datetime.now()
            
        logger.info(f"Updated configuration for camera: {camera_id}")
    
    async def delete_camera_config(self, camera_id: str) -> None:
        """Delete configuration for a specific camera
        
        Args:
            camera_id: ID of the camera
        """
        async with self._config_lock:
            if camera_id in self._config:
                del self._config[camera_id]
                
                async with aiofiles.open(self.config_path, 'w') as f:
                    await f.write(json.dumps(self._config, indent=4))
                
                self._last_load = datetime.now()
                logger.info(f"Deleted configuration for camera: {camera_id}")
    
    async def cleanup(self) -> None:
        """Cleanup configuration manager resources"""
        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass
            
        logger.info("Camera configuration manager cleaned up") 
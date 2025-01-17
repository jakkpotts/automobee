import asyncio
import logging
from typing import List, Optional
from pathlib import Path

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try importing folium first
try:
    import folium
except ImportError as e:
    logger.error("Folium package is required. Please install it using: pip install folium")
    raise ImportError("Missing required package: folium") from e

# Import our modules after checking dependencies
from scripts.setup import SetupManager
from src.camera.stream_manager import StreamManager
from src.visualization.zone_dashboard import ZoneDashboard
from src.websocket.dashboard_server import DashboardWebSocket
from src.utils.config_manager import ConfigManager
from src.utils.exceptions import ConfigurationError

class AutomoBee:
    """Main class for the AutomoBee Vehicle Detection System."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the AutomoBee system.
        
        Args:
            config_path (Optional[Path]): Path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.detection_module = None
        self.classification_module = None
        self.storage_module = None
        self.stream_manager = StreamManager()
        
    def _load_config(self, config_path: Optional[Path]) -> dict:
        """Load system configuration from file."""
        try:
            config_manager = ConfigManager(config_path)
            return config_manager.get_all()
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {str(e)}")
            raise ConfigurationError("Failed to load system configuration") from e
        
    async def initialize(self):
        """Initialize all system components."""
        try:
            # Initialize stream manager
            await self.stream_manager.initialize()
            # Initialize dashboard
            self.dashboard = ZoneDashboard(self.stream_manager.zone_manager)
            # Initialize WebSocket server
            self.websocket_server = DashboardWebSocket(self.dashboard)
            await self.websocket_server.start()
            
            self.logger.info("AutomoBee system initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize AutomoBee system: {str(e)}")
            raise
            
    async def start(self):
        """Start the AutomoBee system."""
        try:
            await self.initialize()
            # TODO: Start processing pipeline
            self.logger.info("AutomoBee system started successfully")
        except Exception as e:
            self.logger.error(f"Failed to start AutomoBee system: {str(e)}")
            raise

    async def cleanup(self):
        """Cleanup system resources."""
        try:
            await self.stream_manager.close()
            if hasattr(self, 'websocket_server'):
                await self.websocket_server.stop()
            # Add other cleanup tasks
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            raise  # Propagate error for proper handling

async def main():
    """Main entry point for the AutomoBee system."""
    try:
        # Initialize and check environment
        setup_manager = SetupManager()
        if not setup_manager.check_environment():
            logger.error("Environment check failed")
            return
        
        automobee = AutomoBee()
        await automobee.start()
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise

async def test_stream_data():
    """Test function to verify camera stream data retrieval and dashboard."""
    try:
        # Initialize stream manager
        stream_manager = StreamManager()
        await stream_manager.initialize()
        
        # Initialize dashboard
        dashboard = ZoneDashboard(stream_manager.zone_manager)
        
        # Generate map
        dashboard.generate_map("test_zone_status.html")
        print("\nMap generated: test_zone_status.html")
        
        # Print stream data
        total_cameras = len(stream_manager.streams)
        print(f"\nTotal cameras found: {total_cameras}")
        
        if total_cameras == 0:
            print("Warning: No cameras found!")
            return
            
        # Print first 5 streams
        for i, (stream_id, stream) in enumerate(stream_manager.streams.items()):
            if i >= 5:
                break
            print(f"\nCamera ID: {stream_id}")
            print(f"Name: {stream.name}")
            print(f"Location: {stream.location}")
            print(f"Coordinates: ({stream.lat}, {stream.lng})")
            print(f"Status: {stream.status.value}")
        
        await stream_manager.close()
        
    except Exception as e:
        logger.error(f"Error testing stream data: {str(e)}")
        raise

if __name__ == "__main__":
    # asyncio.run(main())  # Comment this out temporarily
    asyncio.run(test_stream_data())  # Run test function instead

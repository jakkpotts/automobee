from vehicle_detector import VehicleDetector
from feed_selector import CameraFeedSelector
import asyncio

async def main():
    # Target vehicle configuration
    target_vehicle = {
        "type": "truck",        # Options: car, motorcycle, bus, truck
        "color": "black",       # Options: red, blue, white, black
        "make": None,         # Not currently implemented
        "model": None         # Not currently implemented
    }

    # Initialize components
    feed_selector = CameraFeedSelector()
    feeds = feed_selector.fetch_available_feeds()
    
    # Select strategic cameras
    strategic_cameras = feed_selector.select_strategic_cameras()
    
    # Initialize detector with 30-second sampling interval
    detector = VehicleDetector(
        sample_interval=30,
        max_retries=3,
        retry_delay=5,
        alert_email="your.email@domain.com",
        alert_threshold=5
    )
    
    # Start monitoring with target vehicle parameters
    await detector.monitor_feeds(strategic_cameras, target_vehicle)

if __name__ == "__main__":
    asyncio.run(main()) 
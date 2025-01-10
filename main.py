from vehicle_detector import VehicleDetector
from feed_selector import CameraFeedSelector
from dashboard import run_dashboard
import asyncio
import logging
import threading
import time

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def main():
    try:
        # Update target vehicle configuration to include specific F-150 variants
        target_vehicle = {
            "make": "ford",
            "model": "f150",
            "model_ids": ["3177", "3178", "3179", "3180", "3181", "3182", "3183", "3184", "3185", "3186", "3187", "3188", "3189", "3190", "3191", "3192", "3193", "3194", "3195", "3196", "3197", "3198", "3199", "3200", "3201", "3202", "3203", "3204", "3205", "3206", "3207", "3208", "3209", "3210", "3211", "3212", "3213", "3214", "3215", "3216", "3217", "3218", "3219"]
            #"confidence_threshold": 0.7,
            #"variants": [
            #    "Ford F-150 Regular Cab 2007",
            #    "Ford F-150 Regular Cab 2012"
            #]
        }

        # Initialize components
        feed_selector = CameraFeedSelector()
        
        # First fetch all available feeds
        logger.info("Fetching available feeds...")
        feeds = await feed_selector.fetch_available_feeds()
        if not feeds:
            logger.error("Failed to fetch any camera feeds")
            return
            
        logger.info(f"Fetched {len(feeds)} feeds")
        
        # Verify feeds
        feed_selector.verify_feeds()
        
        # Select strategic cameras
        logger.info("Selecting strategic cameras...")
        strategic_cameras = feed_selector.select_strategic_cameras()
        logger.info(f"Selected {len(strategic_cameras)} strategic cameras")
        
        # Initialize detector
        detector = VehicleDetector(
            sample_interval=50,
            max_retries=3,
            retry_delay=5,
            alert_email="your.email@domain.com",
            alert_threshold=5
        )
        
        # Start monitoring with validated cameras
        await detector.monitor_feeds(strategic_cameras, target_vehicle)
        
    except Exception as e:
        logger.error("Error in main:", exc_info=True)
        raise

def run_detection():
    """Run the detection process"""
    asyncio.run(main())

def start_dashboard():
    """Start dashboard with auto-restart capability"""
    while True:
        try:
            from dashboard import run_dashboard
            run_dashboard()
        except Exception as e:
            logger.error(f"Dashboard crashed: {e}. Restarting in 5 seconds...")
            time.sleep(5)

if __name__ == "__main__":
    try:
        # Start dashboard in a new process
        import multiprocessing
        dashboard_process = multiprocessing.Process(
            target=start_dashboard,
            daemon=True  # Make it daemon so it exits with main process
        )
        dashboard_process.start()
        
        # Give the dashboard time to start
        time.sleep(3)
        
        # Run detection in main process
        run_detection()
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    except Exception as e:
        logger.error("Error in main process:", exc_info=True)
    finally:
        # Clean shutdown
        if 'dashboard_process' in locals():
            dashboard_process.terminate()
            dashboard_process.join(timeout=5)
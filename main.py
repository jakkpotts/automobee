from calendar import c
from vehicle_detector import VehicleDetector
from feed_selector import CameraFeedSelector
from dashboard import run_dashboard, dashboard_bp, dashboard_state, serialize_event, event_queue, Alert
import asyncio
import threading
import logging
from logger_config import logger  # Import the shared logger
import time
from datetime import datetime
from datetime import timezone
from app_factory import app
import json

app.register_blueprint(dashboard_bp)

async def main():
    detector = None
    try:
        # Update target vehicle configuration to include specific F-150 variants
        target_vehicle = {
            "make": "ford",
            "model": "f150",
            "confidence_threshold": 0.65,
            "type": "truck"
        }

        # Initialize components with validated feeds
        feed_selector = CameraFeedSelector()
        
        # Load all cameras from config file first
        logger.info("Loading cameras from config...")
        try:
            with open('config/camera_locations.json') as f:
                camera_config = json.load(f)
                logger.info(f"Successfully loaded {len(camera_config)} cameras from config")
        except Exception as e:
            logger.error(f"Error loading camera config: {e}")
            camera_config = {}
        
        if not camera_config:
            logger.error("No cameras found in config file")
            return
        
        # Update dashboard with all available cameras first
        for camera_id, camera_info in camera_config.items():
            try:
                camera_data = {
                    'id': camera_id,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'name': camera_info.get('name', camera_id),
                    'active': True,
                    'location': {
                        'lat': float(camera_info['lat']),
                        'lng': float(camera_info['lng'])
                    },
                    'stream_url': camera_info.get('url', ''),
                    'type': 'camera'
                }
                
                logger.info(f"Adding camera to dashboard: {camera_id}")
                dashboard_state.update_camera(camera_id, camera_data)
                
                # Emit individual map update for debugging
                event_queue.put(serialize_event('map_update', {
                    'points': [{
                        'id': camera_id,
                        'name': camera_data['name'],
                        'latitude': float(camera_info['lat']),
                        'longitude': float(camera_info['lng']),
                        'type': 'camera',
                        'stream_url': camera_info.get('url', '')
                    }]
                }))
                
                logger.info(f"Added camera {camera_id} at {camera_data['location']}")
            except Exception as e:
                logger.error(f"Error processing camera {camera_id}: {e}")
                continue

        # Now continue with feed selection and monitoring
        feeds = feed_selector.validated_feeds
        if not feeds:
            await feed_selector.update_feeds()
            feeds = feed_selector.feeds
        if not feeds:
            logger.error("No valid camera feeds found in configuration")
            return
            
        logger.info(f"Found {len(feeds)} valid feeds")
        
        # Skip verification since feeds are already validated
        # Select strategic cameras
        logger.info("Selecting strategic cameras...")
        strategic_cameras = feed_selector.select_strategic_cameras()
        logger.info(f"Selected {len(strategic_cameras)} strategic cameras")
        
        # Initialize detector and start monitoring
        detector = VehicleDetector(
            sample_interval=50,
            max_retries=3,
            retry_delay=5,
            alert_email="your.email@domain.com",
            alert_threshold=5
        )
        
        # Create monitoring task
        monitor_task = asyncio.create_task(detector._monitor_health())
        
        try:
            # Start monitoring - this will now handle camera initialization
            await detector.monitor_feeds(strategic_cameras, target_vehicle)
        finally:
            # Cancel monitoring task if it's still running
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
        
    except Exception as e:
        logger.error("Error in main:", exc_info=True)
        if detector:
            try:
                await detector.close_session()
            except Exception as cleanup_error:
                logger.error(f"Error during cleanup: {cleanup_error}")
        raise

def run_detection(shared_state):
    """Run the detection process with shared state"""
    try:
        async def run_async():
            detector = VehicleDetector(
                sample_interval=50,
                max_retries=3,
                retry_delay=5,
                alert_email="your.email@domain.com",
                alert_threshold=5,
                shared_state=shared_state
            )
            
            # Create monitoring task
            monitor_task = asyncio.create_task(detector._monitor_health())
            
            try:
                await main()
            finally:
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass
                await detector.close_session()

        asyncio.run(run_async())
    except Exception as e:
        logger.error(f"Detection error: {e}")
        shared_state['error'] = str(e)

def start_dashboard(shared_state):
    """Start dashboard with auto-restart capability"""
    while True:
        try:
            from dashboard import run_dashboard
            # Pass shared state to dashboard
            run_dashboard(shared_state)
        except Exception as e:
            logger.error(f"Dashboard crashed: {e}. Restarting in 5 seconds...")
            time.sleep(5)

if __name__ == "__main__":
    try:
        # Create shared manager for cross-process state
        import multiprocessing
        from multiprocessing import Manager
        
        manager = Manager()
        shared_state = manager.dict()
        
        # Load cameras first
        logger.info("Loading initial camera data...")
        try:
            with open('config/camera_locations.json') as f:
                camera_config = json.load(f)
                
            # Pre-populate shared state with cameras
            shared_state['cameras'] = {
                camera_id: {
                    'id': camera_id,
                    'name': camera_info.get('name', camera_id),
                    'active': True,
                    'location': {
                        'lat': float(camera_info['lat']),
                        'lng': float(camera_info['lng'])
                    },
                    'stream_url': camera_info.get('url', ''),
                    'type': 'camera'
                }
                for camera_id, camera_info in camera_config.items()
            }
            shared_state['detections'] = []
            
            logger.info(f"Loaded {len(shared_state['cameras'])} cameras into shared state")
            
        except Exception as e:
            logger.error(f"Error loading camera config: {e}")
            shared_state['cameras'] = {}
            shared_state['detections'] = []

        # Start dashboard with populated shared state
        dashboard_process = multiprocessing.Process(
            target=start_dashboard,
            args=(shared_state,),
            daemon=True
        )
        dashboard_process.start()
        
        # Give the dashboard time to start
        time.sleep(3)
        
        # Run detection with shared state
        run_detection(shared_state)
        
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    except Exception as e:
        logger.error("Error in main process:", exc_info=True)
    finally:
        if 'dashboard_process' in locals():
            dashboard_process.terminate()
            dashboard_process.join(timeout=5)
import asyncio
import random
import os
from datetime import datetime
from dotenv import load_dotenv
from src.backend.core.automobee import AutomoBee
from logger_config import logger
from src.backend.services.database_service import DatabaseService

# Load environment variables
load_dotenv()

# Database configuration
DB_URL = os.getenv('DATABASE_URL', 'sqlite:///automobee.db')

async def main():
    """Main entry point for the AutomoBee system"""
    try:
        # Initialize AutomoBee system
        system = AutomoBee(DB_URL)
        
        async with system.run() as automobee:
            # System is now running with all components initialized
            logger.info("AutomoBee system running. Press Ctrl+C to exit.")
            
            # Keep the system running
            while True:
                await asyncio.sleep(1)
                
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    except Exception as e:
        logger.error("Error in main process:", exc_info=True)
        raise

async def test_stream_data(enable_database: bool = False):
    """Test function to verify camera stream data retrieval and dashboard.
    
    Args:
        enable_database (bool): Enable actual database operations. Defaults to False.
    """
    try:
        logger.info("Starting test stream data simulation")
        
        # Initialize database if enabled
        db_service = None
        if enable_database:
            logger.info(f"Initializing database connection: {DB_URL}")
            db_service = DatabaseService(DB_URL)
        
        # Simulate camera configurations
        test_cameras = {
            "cam_001": {
                "name": "Downtown Camera 1",
                "lat": 36.1699,
                "lng": -115.1398,
                "stream_url": "rtsp://dummy-1",
                "status": "active"
            },
            "cam_002": {
                "name": "Strip Camera 2",
                "lat": 36.1723,
                "lng": -115.1352,
                "stream_url": "rtsp://dummy-2",
                "status": "connecting"
            },
            "cam_003": {
                "name": "Airport Camera 3",
                "lat": 36.0800,
                "lng": -115.1522,
                "stream_url": "rtsp://dummy-3",
                "status": "error"
            }
        }
        
        # Simulate vehicle detections
        vehicle_types = ["car", "truck", "motorcycle", "bus"]
        makes = ["Ford", "Toyota", "Honda", "Tesla"]
        models = ["F-150", "Camry", "Civic", "Model 3"]
        
        logger.info("Initializing test environment")
        print("\nTest Environment:")
        print("----------------")
        print(f"Database URL: {DB_URL}")
        print(f"Total cameras: {len(test_cameras)}")
        print("Camera Locations:")
        for cam_id, cam in test_cameras.items():
            print(f"- {cam['name']}: ({cam['lat']}, {cam['lng']})")
        
        # Simulate real-time updates
        print("\nSimulating real-time updates (press Ctrl+C to stop)...")
        update_count = 0
        
        while True:
            update_count += 1
            
            # Simulate camera status changes
            for cam_id in test_cameras:
                if random.random() < 0.1:  # 10% chance of status change
                    test_cameras[cam_id]["status"] = random.choice(["active", "connecting", "error"])
            
            # Simulate vehicle detection
            if random.random() < 0.3:  # 30% chance of detection
                detection = {
                    "camera_id": random.choice(list(test_cameras.keys())),
                    "timestamp": datetime.now().isoformat(),
                    "vehicle_type": random.choice(vehicle_types),
                    "make": random.choice(makes),
                    "model": random.choice(models),
                    "confidence": round(random.uniform(0.65, 0.98), 2),
                    "location": {
                        "lat": random.uniform(36.0800, 36.1723),
                        "lng": random.uniform(-115.1522, -115.1352)
                    }
                }
                print(f"\nDetection {update_count}: {detection['make']} {detection['model']} "
                      f"({detection['confidence']:.2f} confidence)")
                
                # Store in database if enabled
                if enable_database and db_service:
                    try:
                        await db_service.store_detection(detection["camera_id"], detection)
                        logger.info("Detection stored in database")
                    except Exception as e:
                        logger.error(f"Failed to store detection in database: {e}")
                else:
                    logger.info("Database disabled, skipping storage")
            
            # Simulate system metrics
            metrics = {
                "cpu_usage": round(random.uniform(20, 80), 1),
                "memory_usage": round(random.uniform(30, 70), 1),
                "gpu_utilization": round(random.uniform(40, 90), 1),
                "detection_rate": round(random.uniform(10, 30), 1),
                "active_streams": len([c for c in test_cameras.values() if c["status"] == "active"]),
                "total_detections": update_count
            }
            
            if update_count % 5 == 0:
                print(f"\nSystem Metrics:")
                print(f"CPU: {metrics['cpu_usage']}% | "
                      f"Memory: {metrics['memory_usage']}% | "
                      f"GPU: {metrics['gpu_utilization']}%")
                # Log metrics to database (simulated)
                logger.info(f"Logging metrics to database: {metrics}")
            
            await asyncio.sleep(2)  # Update every 2 seconds
            
    except KeyboardInterrupt:
        logger.info("Test stream simulation stopped by user")
        print("\nTest simulation stopped")
    except Exception as e:
        logger.error(f"Error in test stream simulation: {str(e)}")
        raise
    finally:
        # Cleanup database connection if it was initialized
        if enable_database and db_service:
            await db_service.close()

if __name__ == "__main__":
    try:
        # Run test function with database disabled by default
        asyncio.run(test_stream_data(enable_database=False))
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        exit(1)
import cv2
import requests
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional
import logging
from ultralytics import YOLO
import torch
import asyncio
import io
from PIL import Image
import aiohttp
import timm  # For efficient model architectures
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Set
import smtplib
from email.message import EmailMessage
from dashboard import Alert, event_queue, dashboard_state, Detection
from rate_limiter import RateLimiter
from vehicle_classifier import VehicleMakeModelClassifier

@dataclass
class CameraStatus:
    name: str
    last_success: Optional[datetime] = None
    consecutive_failures: int = 0
    total_detections: int = 0
    last_error: Optional[str] = None

class VehicleDetector:
    def __init__(self, sample_interval: int = 50, max_retries: int = 3, retry_delay: int = 5, alert_email: Optional[str] = None, alert_threshold: int = 5):
        # Setup logging
        self.logger = self._setup_logging()
        
        # Setup device
        self.device = (
            "mps" if torch.backends.mps.is_available() 
            else "cuda" if torch.cuda.is_available() 
            else "cpu"
        )
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize models
        try:
            self.detector = YOLO('yolov8n.pt')  # Ensure correct initialization
            self.logger.info("Initializing VehicleMakeModelClassifier")
            self.make_model_classifier = VehicleMakeModelClassifier()
            
            # Verify classifier is properly initialized
            if not self.make_model_classifier.is_initialized():
                self.logger.error("Make/model classifier failed to initialize properly")
                raise RuntimeError("Make/model classifier failed to initialize properly")
                
            self.logger.info("Successfully initialized YOLO detector and make/model classifier")
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}", exc_info=True)
            raise
        
        self.sample_interval = sample_interval
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.last_sample_time = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.vehicle_classes = {
            2: 'car', 
            3: 'motorcycle', 
            5: 'bus', 
            7: 'truck',
            # Add any other relevant vehicle classes from YOLO
        }
        
        # Create output directory
        Path('matches').mkdir(exist_ok=True)
        
        # Add rate limiter (max 10 requests per minute per camera)
        self.rate_limiter = RateLimiter(calls=30, period=60)
        
        # Monitoring
        self.camera_status: Dict[str, CameraStatus] = {}
        self.alert_email = alert_email
        self.alert_threshold = alert_threshold
        self.alerted_cameras: Set[str] = set()
        
        # Stats tracking
        self.stats = {
            'total_detections': 0,
            'total_errors': 0,
            'uptime_start': datetime.now(),
            'camera_stats': {}
        }
        
        # Start monitoring task
        if alert_email:
            asyncio.create_task(self._monitor_health())
        
        # Add stream health tracking
        self.stream_health = {}
        self.max_stream_failures = 3  # Maximum consecutive failures before switching servers
        self.stream_retry_delay = 30  # Seconds to wait before retrying failed stream
        self.server_alternatives = ['01', '02', '03', '04', '05']  # Available server numbers

        # Add camera locations tracking
        self.camera_locations = {}

    def _setup_logging(self) -> logging.Logger:
        """Setup logging with custom handler for dashboard"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('vehicle_detection.log'),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)
        
        # Add custom handler for dashboard events
        class DashboardHandler(logging.Handler):
            def emit(self, record):
                try:
                    log_type = 'error' if record.levelno >= logging.ERROR else \
                              'warning' if record.levelno >= logging.WARNING else 'info'
                    event_queue.put({
                        'type': 'log',
                        'data': {
                            'type': log_type,
                            'message': record.getMessage(),
                            'timestamp': datetime.now().isoformat()
                        }
                    })
                except Exception:
                    pass

        logger.addHandler(DashboardHandler())
        return logger

    async def init_session(self):
        """Initialize aiohttp session with retry options"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=10)  # 10 second timeout
            self.session = aiohttp.ClientSession(timeout=timeout)

    async def close_session(self):
        """Properly close the aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None

    async def get_camera_stream(self, url: str) -> Optional[cv2.VideoCapture]:
        """
        Initialize video stream connection with enhanced retry logic and server fallbacks
        """
        camera_id = url.split('/')[-2] if '/' in url else 'unknown'
        
        if camera_id not in self.stream_health:
            self.stream_health[camera_id] = {
                'failures': 0,
                'last_success': None,
                'current_server': None,
                'tried_servers': set()
            }
        
        health = self.stream_health[camera_id]
        
        # If we've had too many failures, wait before retrying
        if health['failures'] >= self.max_stream_failures:
            last_try = health.get('last_try', 0)
            if time.time() - last_try < self.stream_retry_delay:
                self.logger.debug(f"Waiting before retrying camera {camera_id}")
                return None
            
            # Reset failure count after delay
            health['failures'] = 0
            health['tried_servers'] = set()

        try:
            # Try original URL first if we haven't tried any servers yet
            if not health['tried_servers']:
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                if cap is not None and cap.isOpened():
                    health['failures'] = 0
                    health['last_success'] = time.time()
                    health['current_server'] = None
                    self.logger.info(f"Successfully connected to camera {camera_id}")
                    return cap

            # If original URL fails, try alternative servers
            for server in self.server_alternatives:
                if server in health['tried_servers']:
                    continue
                
                try:
                    # Construct alternative URL
                    alt_url = self._get_alternative_url(url, server)
                    self.logger.debug(f"Trying alternative server {server} for camera {camera_id}")
                    
                    cap = cv2.VideoCapture(alt_url, cv2.CAP_FFMPEG)
                    if cap is not None and cap.isOpened():
                        health['failures'] = 0
                        health['last_success'] = time.time()
                        health['current_server'] = server
                        self.logger.info(f"Successfully connected to camera {camera_id} using server {server}")
                        return cap
                    
                    health['tried_servers'].add(server)
                    
                except Exception as e:
                    self.logger.debug(f"Failed to connect using server {server}: {e}")
                    health['tried_servers'].add(server)
                    continue

            # If all attempts failed
            health['failures'] += 1
            health['last_try'] = time.time()
            self.logger.error(
                f"Failed to connect to camera {camera_id} after trying all servers. "
                f"Failures: {health['failures']}"
            )
            return None
            
        except Exception as e:
            health['failures'] += 1
            health['last_try'] = time.time()
            self.logger.error(f"Error connecting to stream {url}: {e}")
            return None

    def _get_alternative_url(self, original_url: str, server: str) -> str:
        """Generate alternative URL using different server number"""
        try:
            # Extract components from original URL
            if 'its.nv.gov' not in original_url:
                return original_url
                
            parts = original_url.split('/')
            camera_id = parts[-2]
            
            # Construct new URL with alternative server
            new_url = f"https://d1wse{server}.its.nv.gov/vegasxcd{server}/{camera_id}_lvflirxcd{server}_public.stream/playlist.m3u8"
            return new_url
            
        except Exception as e:
            self.logger.error(f"Error generating alternative URL: {e}")
            return original_url

    async def monitor_feeds(self, cameras: Dict[str, Dict], target_vehicle: Dict):
        """Monitor camera feeds with error handling and recovery"""
        try:
            # Initialize camera locations first
            for name, camera in cameras.items():
                # Handle different possible location formats
                location = camera.get('location', {})
                if isinstance(location, str):
                    # Parse string location into dict if needed
                    try:
                        location = json.loads(location)
                    except json.JSONDecodeError:
                        location = {'lat': 0, 'lng': 0}  # Default location if parsing fails
                        
                self.camera_locations[name] = {
                    'lat': location.get('lat', 0),
                    'lng': location.get('lng', 0),
                    'status': 'active',
                    'url': camera.get('url', ''),
                    'last_update': datetime.now().isoformat()
                }
            
            # Update dashboard state with initial camera locations
            dashboard_state.camera_locations = self.camera_locations

            await self.init_session()
            
            while True:
                current_time = time.time()
                tasks = []
                
                for name, camera in cameras.items():
                    # Check if camera should be sampled
                    if current_time - self.last_sample_time.get(name, 0) >= self.sample_interval:
                        tasks.append(self.process_camera(name, camera, target_vehicle))
                        self.last_sample_time[name] = current_time
                
                if tasks:
                    # Process cameras concurrently with error handling
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Handle any errors from the tasks
                    for name, result in zip(cameras.keys(), results):
                        if isinstance(result, Exception):
                            self.logger.error(f"Error processing camera {name}: {result}")
                            # Reset last sample time to retry sooner
                            self.last_sample_time[name] = 0
                
                # Brief sleep to prevent CPU overload
                await asyncio.sleep(1)
                
        except Exception as e:
            self.logger.error(f"Monitor loop failed: {e}")
            raise
        finally:
            await self.close_session()

    async def process_camera(self, name: str, camera: Dict, target_vehicle: Dict):
        """Process a single camera feed with enhanced error handling"""
        try:
            # Update camera status in locations
            if name in self.camera_locations:
                self.camera_locations[name].update({
                    'status': 'processing',
                    'last_update': datetime.now().isoformat()
                })
                # Update dashboard state
                dashboard_state.camera_locations = self.camera_locations

            # Get video stream with retries
            cap = await self.get_camera_stream(camera['url'])
            if not cap:
                self.logger.warning(
                    f"Camera {name} is currently unavailable. "
                    f"Will retry in {self.stream_retry_delay} seconds"
                )
                return

            if name not in self.camera_status:
                self.camera_status[name] = CameraStatus(name=name)
                self.logger.info(f"Initialized new camera: {name}")
            
            try:
                self.logger.info(f"Starting processing for camera {name}")
                # Emit camera processing start event
                event_queue.put({
                    'type': 'camera_status',
                    'data': {
                        'name': name,
                        'status': 'processing',
                        'timestamp': datetime.now().isoformat(),
                        'location': camera.get('location', {}),
                        'url': camera.get('url', '')
                    }
                })
                
                # Update camera stats
                self.stats['camera_stats'][name] = {
                    'status': 'processing',
                    'last_update': datetime.now().isoformat(),
                    'total_detections': self.camera_status[name].total_detections,
                    'consecutive_failures': self.camera_status[name].consecutive_failures,
                    'last_error': self.camera_status[name].last_error,
                    'location': camera.get('location', {}),
                    'url': camera.get('url', '')
                }
                
                # Process frames at specified interval
                last_process_time = 0
                frames_processed = 0
                total_detections = 0

                while True:
                    current_time = time.time()
                    
                    if not await self.rate_limiter.acquire(name):
                        await asyncio.sleep(1)
                        continue
                        
                    if current_time - last_process_time >= self.sample_interval:
                        frames_processed += 1
                        self.logger.debug(f"Processing frame {frames_processed} from camera {name}")

                        ret, frame = cap.read()
                        if not ret:
                            self.logger.error(f"Failed to read frame from camera {name}")
                            raise ValueError("Failed to read frame")

                        # Convert frame to PIL Image for processing
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(frame_rgb)
                        
                        # Emit processing status
                        event_queue.put({
                            'type': 'camera_status',
                            'data': {
                                'name': name,
                                'status': 'analyzing',
                                'timestamp': datetime.now().isoformat(),
                                'location': camera.get('location', {}),
                                'url': camera.get('url', '')
                            }
                        })
                        
                        # Run detection on frame
                        results = self.detector(img)
                        detections_this_frame = len(results[0].boxes.data.tolist())
                        total_detections += detections_this_frame

                        if detections_this_frame > 0:
                            print("\n" + "="*70)
                            print(f"📸 Camera: {name}")
                            print(f"🔍 Found {detections_this_frame} potential vehicles")
                            print("="*70)

                        # Initialize matches list ONCE before processing detections
                        matches = []
                        
                        # Process each detection
                        for det in results[0].boxes.data.tolist():
                            x1, y1, x2, y2, conf, cls = det
                            vehicle_type = self.vehicle_classes.get(int(cls), 'unknown')
                            
                            print(f"\n🚗 Vehicle Detection:")
                            print(f"   Type: {vehicle_type}")
                            print(f"   Confidence: {conf*100:.1f}%")
                            
                            if int(cls) in self.vehicle_classes:
                                vehicle_img = img.crop((int(x1), int(y1), int(x2), int(y2)))
                                classification = self.make_model_classifier.classify_vehicle(vehicle_img)
                                
                                if classification:
                                    print(f"\n🔎 Make/Model Analysis:")
                                    print(f"   Make: {classification['make']}")
                                    print(f"   Model: {classification['model']}")
                                    print(f"   Confidence: {classification['confidence']*100:.1f}%")
                                    
                                    # Check for Ford F-150 using the new classifier method
                                    is_f150 = self.make_model_classifier.is_ford_f150(
                                        vehicle_img, 
                                        confidence_threshold=0.7
                                    )
                                    
                                    if is_f150:
                                        matches.append({
                                            'bbox': (x1, y1, x2, y2),
                                            'confidence': classification['confidence'],
                                            'type': 'ford_f150',
                                            'location': name,
                                            'camera_location': camera.get('location', {}),
                                            'camera_url': camera.get('url', ''),
                                            'make': classification['make'],
                                            'model': classification['model']
                                        })

                        if matches:
                            self.logger.info(
                                f"\n=== Match Summary ===\n"
                                f"Camera: {name}\n"
                                f"Total Matches: {len(matches)}\n"
                                f"Running Total: {self.stats['total_detections'] + len(matches)}\n"
                                f"===================="
                            )
                            await self._save_matches(name, img, matches)
                            
                            # Update detection counts
                            self.camera_status[name].total_detections += len(matches)
                            self.stats['total_detections'] += len(matches)
                            self.stats['camera_stats'][name]['total_detections'] = \
                                self.camera_status[name].total_detections

                        # Update processing stats
                        event_queue.put({
                            'type': 'processing_stats',
                            'data': {
                                'camera': name,
                                'frames_processed': frames_processed,
                                'total_detections': total_detections,
                                'matches': len(matches),
                                'timestamp': datetime.now().isoformat()
                            }
                        })

                        last_process_time = current_time
                    
                    await asyncio.sleep(0.1)
                        
            except Exception as e:
                self.logger.error(f"Error processing camera {name}: {str(e)}")
                self.camera_status[name].consecutive_failures += 1
                self.camera_status[name].last_error = str(e)
                self.stats['total_errors'] += 1
                
                # Update camera stats
                self.stats['camera_stats'][name].update({
                    'status': 'error',
                    'last_error': str(e),
                    'consecutive_failures': self.camera_status[name].consecutive_failures,
                    'last_update': datetime.now().isoformat()
                })
                
                # Emit error status
                event_queue.put({
                    'type': 'camera_status',
                    'data': {
                        'name': name,
                        'status': 'error',
                        'error': str(e),
                        'timestamp': datetime.now().isoformat(),
                        'location': camera.get('location', {}),
                        'url': camera.get('url', '')
                    }
                })

                # Update camera status on error
                if name in self.camera_locations:
                    self.camera_locations[name].update({
                        'status': 'error',
                        'last_update': datetime.now().isoformat(),
                        'error': str(e)
                    })
                    # Update dashboard state
                    dashboard_state.camera_locations = self.camera_locations
                
                raise
        except Exception as e:
            self.logger.error(f"Error processing camera {name}: {e}")
            # Update camera status
            if name in self.camera_status:
                self.camera_status[name].consecutive_failures += 1
                self.camera_status[name].last_error = str(e)
            
        finally:
            if cap:
                cap.release()

    def _is_black_vehicle(self, img: Image.Image) -> bool:
        """Check if vehicle is black"""
        try:
            # Convert PIL Image to numpy array for OpenCV
            img_np = np.array(img)
            hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
            
            # Define black color range
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([180, 255, 30])
            
            # Create mask and calculate ratio
            mask = cv2.inRange(hsv, lower_black, upper_black)
            black_ratio = np.count_nonzero(mask) / mask.size
            
            return black_ratio > 0.4
            
        except Exception as e:
            self.logger.error(f"Error in color detection: {e}")
            return False

    async def _save_matches(self, camera_name: str, img: Image.Image, matches: List[Dict]):
        """Save matched vehicles with error handling"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_camera_name = camera_name.replace(' ', '_').replace('&', 'and')
            safe_camera_name = ''.join(c for c in safe_camera_name if c.isalnum() or c in '_-')

            # Convert the PIL image to a format compatible with OpenCV
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            for idx, match in enumerate(matches):
                try:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = match['bbox']
                    
                    # Draw the bounding box on the original image
                    cv2.rectangle(img_cv, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Green box

                    # Save the modified image with bounding box
                    filename = f"matches/{safe_camera_name}_{timestamp}_{idx}.jpg"
                    cv2.imwrite(filename, img_cv)

                    # Save match metadata
                    metadata = {
                        'timestamp': timestamp,
                        'camera': camera_name,  # Keep original camera name in metadata
                        'vehicle_type': match['type'],
                        'confidence': float(match['confidence']),
                        'image': f"{safe_camera_name}_{timestamp}_{idx}.jpg",  # Use safe filename
                        'make': match.get('make'),
                        'model': match.get('model')
                    }

                    meta_filename = f"matches/{safe_camera_name}_{timestamp}_{idx}.json"
                    with open(meta_filename, 'w') as f:
                        json.dump(metadata, f, indent=2)

                    # Add to recent detections for dashboard
                    dashboard_state.add_detection(Detection(
                        timestamp=timestamp,
                        camera=camera_name,
                        image=f"{safe_camera_name}_{timestamp}_{idx}.jpg",
                        type=match['type'],
                        confidence=float(match['confidence']),
                        make=match.get('make'),
                        model=match.get('model')
                    ))

                    event_queue.put({
                        'type': 'detection',
                        'data': metadata
                    })
                    self.logger.info(f"Pushed detection to dashboard: {metadata}")

                except Exception as e:
                    self.logger.error(f"Error saving match {idx} from {camera_name}: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Error saving matches from {camera_name}: {e}")

    async def _monitor_health(self):
        """Monitor system health and send alerts"""
        while True:
            try:
                await self._check_camera_health()
                await self._save_stats()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Error in health monitoring: {e}")

    async def _check_camera_health(self):
        """Check health of all cameras and send alerts if needed"""
        try:
            total_cameras = len(self.camera_status)
            active_cameras = sum(1 for status in self.camera_status.values() 
                               if status.consecutive_failures == 0)
            
            self.logger.info(
                f"Health check: {active_cameras}/{total_cameras} cameras active, "
                f"{self.stats['total_detections']} total detections"
            )

            # System-wide health alert if too many cameras are down
            if active_cameras < total_cameras * 0.7:  # Less than 70% cameras active
                await self._send_alert(
                    "System Health Warning",
                    f"Only {active_cameras}/{total_cameras} cameras are active"
                )

            for name, status in self.camera_status.items():
                # Alert on consecutive failures
                if (status.consecutive_failures >= self.alert_threshold and 
                    name not in self.alerted_cameras):
                    await self._send_alert(
                        f"Camera Failure: {name}",
                        f"Failed {status.consecutive_failures} times consecutively.\n"
                        f"Last error: {status.last_error}\n"
                        f"Last success: {status.last_success}"
                    )
                    self.alerted_cameras.add(name)
                    
                    # Add to dashboard active alerts
                    dashboard_state.add_alert(Alert(
                        type='camera_failure',
                        message=f"Camera {name} has failed {status.consecutive_failures} times",
                        timestamp=datetime.now().isoformat(),
                        camera=name
                    ))
                
                # Recovery alert
                elif status.consecutive_failures == 0 and name in self.alerted_cameras:
                    await self._send_alert(
                        f"Camera Recovery: {name}",
                        "Camera has recovered and is functioning normally"
                    )
                    self.alerted_cameras.remove(name)
                    
                    # Clear alert from dashboard
                    dashboard_state.clear_alert(name)

                # Update dashboard with camera status
                event_queue.put({
                    'type': 'camera_status',
                    'data': {
                        'name': name,
                        'status': 'error' if status.consecutive_failures > 0 else 'active',
                        'failures': status.consecutive_failures,
                        'last_error': status.last_error,
                        'last_success': status.last_success,
                        'total_detections': status.total_detections,
                        'timestamp': datetime.now().isoformat()
                    }
                })

        except Exception as e:
            self.logger.error(f"Error in health monitoring: {e}")

    async def _send_alert(self, subject: str, body: str):
        """Send alert to console and dashboard"""
        try:
            # Format alert message
            timestamp = datetime.now().isoformat()
            alert_msg = f"\n[ALERT] {subject}\n{body}\nTimestamp: {timestamp}\n"
            
            # Print to console
            print(alert_msg)
            self.logger.warning(alert_msg)  # Also log it
            
            # Create alert for dashboard
            alert = {
                'type': 'system_alert',
                'message': f"{subject}: {body}",
                'timestamp': timestamp,
                'severity': 'warning'
            }
            
            # Push to dashboard via event queue
            event_queue.put({
                'type': 'alert',
                'data': alert
            })
            
            # Add to dashboard state
            dashboard_state.add_alert(Alert(
                type='system_alert',
                message=f"{subject}: {body}",
                timestamp=timestamp,
                camera=None  # This is a system-wide alert
            ))
            
        except Exception as e:
            self.logger.error(f"Failed to process alert: {e}")

    async def _save_stats(self):
        """Save monitoring stats to file"""
        try:
            stats_file = Path('monitoring/stats.json')
            stats_file.parent.mkdir(exist_ok=True)
            
            with open(stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Failed to save stats: {e}")

    def _calculate_uptime(self, camera_name: str) -> float:
        """Calculate camera uptime percentage"""
        status = self.camera_status[camera_name]
        total_time = (datetime.now() - self.stats['uptime_start']).total_seconds()
        if total_time == 0:
            return 100.0
            
        # Calculate downtime based on consecutive failures and sample interval
        downtime = status.consecutive_failures * self.sample_interval
        uptime_percentage = ((total_time - downtime) / total_time) * 100
        
        return min(100.0, max(0.0, uptime_percentage))  # Ensure between 0-100%
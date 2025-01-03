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
from dashboard import event_queue, dashboard_state
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
    def __init__(self, sample_interval: int = 30, max_retries: int = 3, retry_delay: int = 5, alert_email: Optional[str] = None, alert_threshold: int = 5):
        # Setup logging
        self.logger = self._setup_logging()
        
        # Setup device
        self.device = (
            "mps" if torch.backends.mps.is_available() 
            else "cuda" if torch.cuda.is_available() 
            else "cpu"
        )
        self.logger.info(f"Using device: {self.device}")
        
        # Load models
        try:
            self.detector = YOLO('yolov8n.pt')
            self.classifier = self._load_classifier()
            if self.classifier is None:
                self.logger.warning("Make/model detection will be disabled")
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            raise
        
        self.sample_interval = sample_interval
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.last_sample_time = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.vehicle_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        
        # Create output directory
        Path('matches').mkdir(exist_ok=True)
        
        # Add rate limiter (max 10 requests per minute per camera)
        self.rate_limiter = RateLimiter(calls=10, period=60)
        
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
        
        # Initialize make/model classifier
        self.make_model_classifier = VehicleMakeModelClassifier()

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
        Initialize video stream connection with retry logic
        """
        try:
            # The URL should already be in the correct format from feed_selector
            # Just try to open it directly first
            cap = cv2.VideoCapture(url)
            
            if not cap.isOpened():
                # Try with explicit FFMPEG backend as fallback
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                
            if cap.isOpened():
                return cap
            else:
                self.logger.error(f"Failed to open stream: {url}")
                return None
            
        except Exception as e:
            self.logger.error(f"Error connecting to stream {url}: {e}")
            return None

    async def monitor_feeds(self, cameras: Dict[str, Dict], target_vehicle: Dict):
        """
        Monitor camera feeds with error handling and recovery
        """
        try:
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
        """Process a single camera feed with rate limiting and monitoring"""
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
            
            # Get video stream
            cap = await self.get_camera_stream(camera['url'])
            if cap is None:
                self.logger.error(f"Failed to connect to stream for camera {name}")
                # Update stats for offline camera
                self.stats['camera_stats'][name].update({
                    'status': 'offline',
                    'last_error': 'Failed to connect to stream'
                })
                self.stats['total_errors'] += 1
                
                # Emit camera offline event
                event_queue.put({
                    'type': 'camera_status',
                    'data': {
                        'name': name,
                        'status': 'offline',
                        'timestamp': datetime.now().isoformat(),
                        'location': camera.get('location', {}),
                        'url': camera.get('url', '')
                    }
                })
                raise ValueError(f"Failed to get stream from camera {name}")
            
            self.logger.info(f"Successfully connected to camera {name}")
            # Update stats for online camera
            self.stats['camera_stats'][name].update({
                'status': 'online',
                'uptime': self._calculate_uptime(name)
            })
            
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
                        self.logger.info(
                            f"Camera {name}: Found {detections_this_frame} potential vehicles "
                            f"(Total: {total_detections})"
                        )

                    matches = []
                    for det in results[0].boxes.data.tolist():
                        x1, y1, x2, y2, conf, cls = det
                        
                        vehicle_type = self.vehicle_classes.get(int(cls), 'unknown')
                        self.logger.debug(
                            f"Detection on camera {name}: {vehicle_type} "
                            f"with {conf*100:.1f}% confidence"
                        )

                        if int(cls) not in self.vehicle_classes:
                            continue

                        # For trucks, check if it's a Ford F-150
                        if (target_vehicle['type'] == 'truck' and int(cls) == 7 and
                            target_vehicle.get('make') == 'ford' and
                            target_vehicle.get('model') == 'f-150'):
                            
                            vehicle_img = img.crop((int(x1), int(y1), int(x2), int(y2)))
                            
                            # Check if it's black and a Ford F-150
                            if (not target_vehicle['color'] or 
                                self._is_black_vehicle(vehicle_img)):
                                
                                if self.make_model_classifier.is_ford_f150(vehicle_img):
                                    self.logger.info(
                                        f"Found target vehicle (Ford F-150) on camera {name} "
                                        f"with {conf*100:.1f}% confidence"
                                    )
                                    matches.append({
                                        'bbox': (x1, y1, x2, y2),
                                        'confidence': conf,
                                        'type': 'truck',
                                        'make': 'ford',
                                        'model': 'f-150',
                                        'location': name,
                                        'camera_location': camera.get('location', {}),
                                        'camera_url': camera.get('url', '')
                                    })

                    if matches:
                        self.logger.info(
                            f"Camera {name}: Confirmed {len(matches)} matching vehicles "
                            f"(Total matches: {self.stats['total_detections'] + len(matches)})"
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
            
            raise
        finally:
            if 'cap' in locals() and cap is not None:
                cap.release()
                self.logger.info(f"Released camera feed: {name}")

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
            
            for idx, match in enumerate(matches):
                try:
                    # Save cropped vehicle image
                    x1, y1, x2, y2 = match['bbox']
                    vehicle_img = img.crop((int(x1), int(y1), int(x2), int(y2)))
                    
                    filename = f"matches/{camera_name}_{timestamp}_{idx}.jpg"
                    vehicle_img.save(filename)
                    
                    # Save match metadata
                    metadata = {
                        'timestamp': timestamp,
                        'camera': camera_name,
                        'vehicle_type': match['type'],
                        'confidence': float(match['confidence']),
                        'image': filename,
                        'make': match.get('make'),
                        'model': match.get('model')
                    }
                    
                    meta_filename = f"matches/{camera_name}_{timestamp}_{idx}.json"
                    with open(meta_filename, 'w') as f:
                        json.dump(metadata, f, indent=2)
                        
                    # Add to recent detections for dashboard
                    dashboard_state.add_detection(Detection(
                        timestamp=timestamp,
                        camera=camera_name,
                        image=filename,
                        type=match['type'],
                        confidence=float(match['confidence']),
                        make=match.get('make'),
                        model=match.get('model')
                    ))
                    
                    # Push update to dashboard
                    event_queue.put({
                        'type': 'detection',
                        'data': metadata
                    })

                    self.logger.info(
                        f"Saved detection {idx+1}/{len(matches)} from camera {camera_name}: "
                        f"{match['type']} ({match.get('make', 'unknown')} {match.get('model', '')})"
                    )
                        
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

            for name, status in self.camera_status.items():
                # Alert on consecutive failures
                if (status.consecutive_failures >= self.alert_threshold and 
                    name not in self.alerted_cameras):
                    self.logger.warning(
                        f"Camera {name} has failed {status.consecutive_failures} times. "
                        f"Last error: {status.last_error}"
                    )
                    await self._send_alert(
                        f"Camera {name} has failed {status.consecutive_failures} times",
                        f"Last error: {status.last_error}\n"
                        f"Last success: {status.last_success}"
                    )
                    self.alerted_cameras.add(name)
                
                # Clear alert if camera recovers
                elif status.consecutive_failures == 0 and name in self.alerted_cameras:
                    self.logger.info(f"Camera {name} has recovered")
                    await self._send_alert(
                        f"Camera {name} has recovered",
                        f"Camera is now functioning normally"
                    )
                    self.alerted_cameras.remove(name)

                if status.consecutive_failures >= self.alert_threshold:
                    alert = {
                        'camera': name,
                        'type': 'error',
                        'message': f"Camera {name} has failed {status.consecutive_failures} times"
                    }
                    dashboard_state.active_alerts.append(alert)
                    event_queue.put({
                        'type': 'alert',
                        'data': alert
                    })

        except Exception as e:
            self.logger.error(f"Error in health monitoring: {e}")

    async def _send_alert(self, subject: str, body: str):
        """Send email alert"""
        if not self.alert_email:
            return
            
        try:
            msg = EmailMessage()
            msg.set_content(body)
            msg['Subject'] = f"Vehicle Detection Alert: {subject}"
            msg['From'] = "vehicle.detector@yourdomain.com"
            msg['To'] = self.alert_email
            
            # Configure your SMTP settings
            with smtplib.SMTP('smtp.yourdomain.com', 587) as server:
                server.starttls()
                server.login('username', 'password')
                server.send_message(msg)
                
        except Exception as e:
            self.logger.error(f"Failed to send alert: {e}")

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

    def _load_classifier(self):
        """
        Load the make/model classifier model
        Using EfficientNet as it's good for fine-grained classification
        """
        try:
            model = timm.create_model('efficientnet_b0', pretrained=True)
            
            # Modify final layer for binary classification (F-150 or not)
            model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
            
            # Load trained weights if available
            model_path = Path('models/vehicle_classifier.pth')
            if model_path.exists():
                self.logger.info("Loading pre-trained make/model classifier")
                checkpoint = torch.load(model_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.logger.warning(
                    "No pre-trained make/model classifier found. "
                    "Run train_classifier.py first for make/model detection."
                )
            
            model = model.to(self.device)
            model.eval()
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load make/model classifier: {e}")
            return None 

class RateLimiter:
    def __init__(self, calls: int, period: int):
        """
        Initialize rate limiter
        Args:
            calls: Number of calls allowed per period
            period: Time period in seconds
        """
        self.calls = calls
        self.period = period
        self.timestamps: Dict[str, List[datetime]] = {}

    async def acquire(self, key: str) -> bool:
        """
        Check if we can make a request for the given key
        Returns True if request is allowed, False otherwise
        """
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.period)
        
        # Initialize timestamps list for new keys
        if key not in self.timestamps:
            self.timestamps[key] = []
        
        # Remove old timestamps
        self.timestamps[key] = [ts for ts in self.timestamps[key] if ts > cutoff]
        
        # Check if we can make another request
        if len(self.timestamps[key]) < self.calls:
            self.timestamps[key].append(now)
            return True
            
        return False 
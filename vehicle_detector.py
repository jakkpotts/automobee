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
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('vehicle_detection.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)

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
            # Convert snapshot URL to stream URL
            stream_url = url.replace('/snapshot', '/stream')
            
            # Try HLS stream first
            hls_url = f"{stream_url}/playlist.m3u8"
            cap = cv2.VideoCapture(hls_url)
            
            if not cap.isOpened():
                # Try RTSP stream as fallback
                rtsp_url = f"rtsp://{stream_url}"
                cap = cv2.VideoCapture(rtsp_url)
                
            if cap.isOpened():
                return cap
            else:
                self.logger.error(f"Failed to open stream: {stream_url}")
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
        
        try:
            # Get video stream
            cap = await self.get_camera_stream(camera['url'])
            if cap is None:
                raise ValueError(f"Failed to get stream from camera {name}")
            
            # Process frames at specified interval
            last_process_time = 0
            while True:
                current_time = time.time()
                
                # Check rate limit
                if not await self.rate_limiter.acquire(name):
                    await asyncio.sleep(1)
                    continue
                    
                # Process frame at sample interval
                if current_time - last_process_time >= self.sample_interval:
                    ret, frame = cap.read()
                    if not ret:
                        raise ValueError("Failed to read frame")
                    
                    # Convert frame to PIL Image for processing
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    
                    # Run detection on frame
                    results = self.detector(img)
                    
                    matches = []
                    for det in results[0].boxes.data.tolist():
                        x1, y1, x2, y2, conf, cls = det
                        
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
                                    matches.append({
                                        'bbox': (x1, y1, x2, y2),
                                        'confidence': conf,
                                        'type': 'truck',
                                        'make': 'ford',
                                        'model': 'f-150',
                                        'location': name
                                    })
                
                    if matches:
                        self.logger.info(f"Camera {name}: Found {len(matches)} matching vehicles")
                        await self._save_matches(name, img, matches)
                    
                    # Update stats
                    self.camera_status[name].last_success = datetime.now()
                    self.camera_status[name].consecutive_failures = 0
                    self.camera_status[name].total_detections += len(matches)
                    
                    last_process_time = current_time
                
                    # Small sleep to prevent CPU overload
                    await asyncio.sleep(0.1)
        except Exception as e:
            self.camera_status[name].consecutive_failures += 1
            self.camera_status[name].last_error = str(e)
            self.stats['total_errors'] += 1
            raise
        finally:
            if 'cap' in locals() and cap is not None:
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
                        'image': filename
                    }
                    
                    meta_filename = f"matches/{camera_name}_{timestamp}_{idx}.json"
                    with open(meta_filename, 'w') as f:
                        json.dump(metadata, f, indent=2)
                        
                    # Add to recent detections for dashboard
                    dashboard_state.recent_detections.insert(0, {
                        'timestamp': timestamp,
                        'camera': camera_name,
                        'image': filename,
                        'type': match['type'],
                        'confidence': float(match['confidence'])
                    })
                    
                    # Keep only last 10 detections
                    if len(dashboard_state.recent_detections) > 10:
                        dashboard_state.recent_detections.pop()
                    
                    # Push update to dashboard
                    event_queue.put({
                        'type': 'detection',
                        'data': metadata
                    })
                        
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
        for name, status in self.camera_status.items():
            # Alert on consecutive failures
            if (status.consecutive_failures >= self.alert_threshold and 
                name not in self.alerted_cameras):
                await self._send_alert(
                    f"Camera {name} has failed {status.consecutive_failures} times",
                    f"Last error: {status.last_error}\n"
                    f"Last success: {status.last_success}"
                )
                self.alerted_cameras.add(name)
            
            # Clear alert if camera recovers
            elif status.consecutive_failures == 0 and name in self.alerted_cameras:
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
            
        failure_time = status.consecutive_failures * self.sample_interval
        return ((total_time - failure_time) / total_time) * 100 

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
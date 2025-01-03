import cv2
import requests
import numpy as np
from datetime import datetime
import time
from typing import Dict, List
import logging
from ultralytics import YOLO
import torch

class TrafficCameraMonitor:
    def __init__(self, model_path: str = "yolov8n.pt"):
        """
        Initialize the traffic camera monitor
        Args:
            model_path: Path to YOLOv8 model weights. If not provided, will download yolov8n.pt
        """
        self.model = self._load_model(model_path)
        self.camera_feeds: Dict[str, Dict] = {}
        self.logger = self._setup_logging()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Class mapping for COCO dataset (used by YOLOv8)
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }
        
        # Color ranges for vehicle detection
        self.color_ranges = {
            'red': ([0, 50, 50], [10, 255, 255]),  # HSV ranges for red
            'blue': ([100, 50, 50], [130, 255, 255]),
            'white': ([0, 0, 200], [180, 30, 255]),
            'black': ([0, 0, 0], [180, 255, 30]),
            # Add more colors as needed
        }

    def _setup_logging(self) -> logging.Logger:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def _load_model(self, model_path: str):
        """Load the YOLO model"""
        try:
            if not model_path.endswith('.pt'):
                self.logger.info("No model specified, downloading YOLOv8n...")
                model_path = 'yolov8n.pt'  # Will auto-download if not present
            
            model = YOLO(model_path)
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def add_camera_feed(self, camera_id: str, url: str, camera_info: Dict = None):
        """
        Add a camera feed URL to monitor
        Args:
            camera_id: Unique identifier for the camera
            url: URL of the camera feed
            camera_info: Additional camera information (name, coordinates, etc.)
        """
        self.camera_feeds[camera_id] = {
            'url': url,
            'name': camera_info.get('name', f"Camera {camera_id}"),
            'coordinates': camera_info.get('coordinates', None),
            'intersection': camera_info.get('intersection', None)
        }
        self.logger.info(f"Added camera feed: {self.camera_feeds[camera_id]['name']} ({camera_id})")

    def _detect_color(self, img, bbox) -> str:
        """
        Detect the dominant color of a vehicle in the given bounding box
        """
        x1, y1, x2, y2 = map(int, bbox)
        vehicle_roi = img[y1:y2, x1:x2]
        
        if vehicle_roi.size == 0:
            return None
            
        # Convert to HSV color space
        hsv = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2HSV)
        
        max_ratio = 0
        detected_color = None
        
        # Check each color range
        for color, (lower, upper) in self.color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            ratio = np.count_nonzero(mask) / mask.size
            
            if ratio > max_ratio:
                max_ratio = ratio
                detected_color = color
                
        return detected_color if max_ratio > 0.15 else None

    def process_frame(self, frame, target_vehicle_desc: Dict):
        """
        Process a single frame to detect vehicles matching the description
        """
        if frame is None:
            return []
            
        # Run YOLO detection
        results = self.model(frame)[0]
        matches = []
        
        # Process each detection
        for det in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = det
            
            # Check if detection is a vehicle
            if int(cls) not in self.vehicle_classes:
                continue
                
            vehicle_type = self.vehicle_classes[int(cls)]
            
            # If vehicle type matches target
            if vehicle_type == target_vehicle_desc.get('type', '').lower():
                # Detect color
                color = self._detect_color(frame, (x1, y1, x2, y2))
                
                # If color matches target
                if color and color == target_vehicle_desc.get('color', '').lower():
                    matches.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf,
                        'type': vehicle_type,
                        'color': color
                    })
        
        return matches

    def monitor_feeds(self, target_vehicle_desc: Dict):
        """
        Main monitoring loop
        """
        while True:
            for camera_id, camera_info in self.camera_feeds.items():
                try:
                    # Get frame from camera feed
                    response = requests.get(camera_info['url'], stream=True)
                    frame = cv2.imdecode(
                        np.frombuffer(response.content, np.uint8),
                        cv2.IMREAD_COLOR
                    )
                    
                    # Process frame
                    matches = self.process_frame(frame, target_vehicle_desc)
                    
                    if matches:
                        # Save match details with intersection info
                        self._save_match(camera_id, frame, matches)
                        
                except Exception as e:
                    self.logger.error(
                        f"Error processing camera {camera_id} "
                        f"({camera_info.get('name', 'Unknown Location')}): {e}"
                    )
                    
                time.sleep(1)  # Rate limiting

    def _save_match(self, camera_id: str, frame, matches: List):
        """
        Save matched frame and detection details with intersection and timing information
        """
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        
        # Create matches directory if it doesn't exist
        import os
        os.makedirs("matches", exist_ok=True)
        
        # Save the full frame
        filename = f"matches/{camera_id}_{timestamp_str}.jpg"
        cv2.imwrite(filename, frame)
        
        # Save individual vehicle crops
        crops = []
        for idx, match in enumerate(matches):
            x1, y1, x2, y2 = map(int, match['bbox'])
            crop = frame[y1:y2, x1:x2]
            crop_filename = f"matches/{camera_id}_{timestamp_str}_vehicle_{idx}.jpg"
            cv2.imwrite(crop_filename, crop)
            crops.append(crop_filename)
        
        # Get intersection details from camera feed info
        intersection_info = {
            'camera_id': camera_id,
            'location': self.camera_feeds.get(camera_id, {}).get('name', 'Unknown Location'),
            'coordinates': self.camera_feeds.get(camera_id, {}).get('coordinates', None)
        }
        
        # Save match metadata with detailed timing and location
        metadata = {
            'camera_id': camera_id,
            'intersection': intersection_info,
            'timestamp': {
                'iso': timestamp.isoformat(),
                'unix': int(timestamp.timestamp()),
                'date': timestamp.strftime("%Y-%m-%d"),
                'time': timestamp.strftime("%H:%M:%S"),
                'timezone': datetime.now().astimezone().tzname()
            },
            'matches': [{
                **match,
                'crop_file': crops[i]
            } for i, match in enumerate(matches)],
            'frame_file': filename
        }
        
        # Save metadata to JSON
        metadata_file = f"matches/{camera_id}_{timestamp_str}_metadata.json"
        with open(metadata_file, 'w') as f:
            import json
            json.dump(metadata, f, indent=2)
        
        # Log the match with location information
        self.logger.info(
            f"Match found at {intersection_info['location']} "
            f"(Camera ID: {camera_id}) "
            f"at {timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        ) 
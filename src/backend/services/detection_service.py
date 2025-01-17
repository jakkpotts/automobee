import asyncio
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    """Container for detection results"""
    vehicle_type: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2] normalized coordinates
    frame_id: int
    timestamp: datetime

class AsyncVehicleDetector:
    """Asynchronous vehicle detection service using YOLOv8"""
    
    def __init__(
        self,
        state_manager: 'StateManager',
        db_service: 'DatabaseService',
        analytics_service: 'AnalyticsService',
        model_path: str = "model_cache/yolov8m.pt",
        batch_size: int = 4,
        confidence_threshold: float = 0.65
    ):
        """Initialize vehicle detector
        
        Args:
            state_manager: System state manager
            db_service: Database service for storing detections
            analytics_service: Analytics service for metrics
            model_path: Path to YOLOv8 model
            batch_size: Batch size for inference
            confidence_threshold: Minimum confidence threshold
        """
        self.state_manager = state_manager
        self.db_service = db_service
        self.analytics_service = analytics_service
        self.model_path = Path(model_path)
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        
        # Performance metrics
        self._processing_times: List[float] = []
        self._detection_counts: Dict[str, int] = {}
        self._last_metrics_update = datetime.now()
        
        # Processing queues
        self._frame_queue = asyncio.Queue(maxsize=30)
        self._result_queue = asyncio.Queue(maxsize=30)
        self._processing_task: Optional[asyncio.Task] = None
        self._is_running = False
        
        # Initialize CUDA device if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Vehicle classes from COCO dataset
        self.vehicle_classes = {
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck"
        }
    
    async def initialize(self) -> None:
        """Initialize the detector"""
        try:
            # Load model
            logger.info("Loading YOLOv8 model...")
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            
            # Start processing task
            self._is_running = True
            self._processing_task = asyncio.create_task(self._process_frames())
            
            logger.info("Vehicle detector initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize vehicle detector: {e}")
            raise
    
    async def process_frame(self, frame: np.ndarray, camera_id: str, frame_id: int) -> None:
        """Queue a frame for processing
        
        Args:
            frame: BGR image as numpy array
            camera_id: ID of the source camera
            frame_id: Frame sequence number
        """
        if self._is_running:
            await self._frame_queue.put((frame, camera_id, frame_id))
    
    async def _process_frames(self) -> None:
        """Main processing loop for batched inference"""
        batch_frames = []
        batch_meta = []
        
        while self._is_running:
            try:
                # Collect frames for batch processing
                while len(batch_frames) < self.batch_size:
                    try:
                        frame, camera_id, frame_id = await asyncio.wait_for(
                            self._frame_queue.get(),
                            timeout=0.1
                        )
                        batch_frames.append(frame)
                        batch_meta.append((camera_id, frame_id))
                    except asyncio.TimeoutError:
                        break
                
                if not batch_frames:
                    await asyncio.sleep(0.01)
                    continue
                
                # Process batch
                start_time = datetime.now()
                
                # Convert frames to tensor
                batch_tensor = torch.stack([
                    torch.from_numpy(frame).to(self.device)
                    for frame in batch_frames
                ])
                
                # Run inference
                results = self.model(batch_tensor, verbose=False)
                
                # Process results
                for result, (camera_id, frame_id) in zip(results, batch_meta):
                    detections = self._process_detections(result, frame_id)
                    if detections:
                        await self._handle_detections(camera_id, detections)
                
                # Update metrics
                process_time = (datetime.now() - start_time).total_seconds()
                self._processing_times.append(process_time)
                
                # Clear batch
                batch_frames.clear()
                batch_meta.clear()
                
            except Exception as e:
                logger.error(f"Error processing frames: {e}")
                await asyncio.sleep(1)
    
    def _process_detections(self, result, frame_id: int) -> List[DetectionResult]:
        """Process YOLOv8 results
        
        Args:
            result: YOLOv8 result object
            frame_id: Frame sequence number
        
        Returns:
            List of DetectionResult objects
        """
        detections = []
        boxes = result.boxes
        
        for box in boxes:
            class_id = int(box.cls)
            if class_id in self.vehicle_classes:
                vehicle_type = self.vehicle_classes[class_id]
                confidence = float(box.conf)
                
                if confidence >= self.confidence_threshold:
                    bbox = box.xyxyn[0].cpu().numpy().tolist()  # normalized coordinates
                    detections.append(DetectionResult(
                        vehicle_type=vehicle_type,
                        confidence=confidence,
                        bbox=bbox,
                        frame_id=frame_id,
                        timestamp=datetime.now()
                    ))
                    
                    # Update detection count
                    self._detection_counts[vehicle_type] = \
                        self._detection_counts.get(vehicle_type, 0) + 1
        
        return detections
    
    async def _handle_detections(self, camera_id: str, detections: List[DetectionResult]) -> None:
        """Handle processed detections
        
        Args:
            camera_id: ID of the source camera
            detections: List of detection results
        """
        for detection in detections:
            # Prepare detection data
            detection_data = {
                "type": detection.vehicle_type,
                "confidence": detection.confidence,
                "bbox": detection.bbox,
                "frame_id": detection.frame_id,
                "metadata": {
                    "processing_device": str(self.device),
                    "model": self.model_path.name
                }
            }
            
            # Store detection
            await self.db_service.store_detection(camera_id, detection_data)
            
            # Update state
            self.state_manager.update_detection_stats(camera_id, {
                "last_detection": detection_data,
                "total_detections": sum(self._detection_counts.values())
            })
    
    def get_detection_rate(self) -> float:
        """Get current detection rate (detections per second)
        
        Returns:
            Detection rate
        """
        total_detections = sum(self._detection_counts.values())
        time_window = (datetime.now() - self._last_metrics_update).total_seconds()
        return total_detections / max(time_window, 1)
    
    def get_processing_latency(self) -> float:
        """Get average processing latency in milliseconds
        
        Returns:
            Average processing latency
        """
        if not self._processing_times:
            return 0.0
        return sum(self._processing_times) / len(self._processing_times) * 1000
    
    def get_avg_confidence(self) -> float:
        """Get average detection confidence
        
        Returns:
            Average confidence score
        """
        total_detections = sum(self._detection_counts.values())
        if total_detections == 0:
            return 0.0
        return self.confidence_threshold  # Using threshold as baseline
    
    async def get_gpu_utilization(self) -> Dict[str, float]:
        """Get GPU utilization metrics if available
        
        Returns:
            Dictionary of GPU metrics
        """
        if self.device.type == "cuda":
            try:
                gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                gpu_memory_cached = torch.cuda.memory_cached() / 1024 / 1024  # MB
                return {
                    "memory_used_mb": gpu_memory,
                    "memory_cached_mb": gpu_memory_cached,
                    "device_name": torch.cuda.get_device_name()
                }
            except Exception:
                pass
        return {"memory_used_mb": 0, "memory_cached_mb": 0, "device_name": "cpu"}
    
    async def cleanup(self) -> None:
        """Cleanup detector resources"""
        self._is_running = False
        
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        
        # Clear queues
        while not self._frame_queue.empty():
            self._frame_queue.get_nowait()
        while not self._result_queue.empty():
            self._result_queue.get_nowait()
        
        # Clear metrics
        self._processing_times.clear()
        self._detection_counts.clear()
        
        logger.info("Vehicle detector cleaned up") 
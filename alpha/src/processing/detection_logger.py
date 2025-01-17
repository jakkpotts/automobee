import logging
from typing import Dict, List
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class Detection:
    frame_id: str
    timestamp: datetime
    zone_id: str
    confidence: float
    metadata: dict = field(default_factory=dict)

class DetectionLogger:
    """Logs and manages detection history."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.detections: Dict[str, List[Detection]] = {}
        
    def log_detection(self, detection: Detection):
        """Log a new detection."""
        if detection.zone_id not in self.detections:
            self.detections[detection.zone_id] = []
            
        self.detections[detection.zone_id].append(detection)
        
        # Trim history if needed
        if len(self.detections[detection.zone_id]) > self.max_history:
            self.detections[detection.zone_id] = self.detections[detection.zone_id][-self.max_history:] 
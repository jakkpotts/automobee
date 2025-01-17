from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime
import numpy as np
from ..zones.zone_manager import ZoneType

@dataclass
class TargetVehicleConfig:
    """Configuration for target vehicle detection and processing."""
    id: str
    vehicle_type: str
    make: Optional[str] = None
    model: Optional[str] = None
    priority_level: int = 1
    pre_screening_threshold: float = 0.6
    classification_threshold: float = 0.85
    active: bool = True
    metadata: Dict[str, Any] = None

@dataclass
class DetectionQueueItem:
    """Represents an item in the detection processing queue."""
    frame: np.ndarray
    frame_id: str
    timestamp: datetime
    stream_id: str
    zone: ZoneType
    bbox: tuple[float, float, float, float]
    initial_type: str
    confidence: float
    priority: int = 0
    target_match: bool = False
    processing_metadata: Dict[str, Any] = None

@dataclass
class QueueMetrics:
    """Metrics for monitoring queue performance."""
    priority_queue_size: int = 0
    standard_queue_size: int = 0
    priority_processing_time: float = 0.0
    standard_processing_time: float = 0.0
    priority_batch_size: int = 0
    standard_batch_size: int = 0
    target_match_rate: float = 0.0
    last_update: datetime = None 
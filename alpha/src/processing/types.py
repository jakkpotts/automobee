from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any
from ..zones.zone_manager import ZoneType

class ProcessingStatus(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"

@dataclass
class ProcessingConfig:
    batch_size: int = 32
    max_queue_size: int = 100
    processing_interval: float = 0.1

@dataclass
class FocusConfig:
    """Configuration for camera focus and processing priority."""
    priority: int = 1
    processing_weight: float = 1.0
    min_confidence: float = 0.5
    max_latency: float = 1.0

@dataclass
class ZoneFocusConfig:
    """Zone-specific focus configuration."""
    zone_type: ZoneType
    focus_config: FocusConfig
    active: bool = True
    metadata: Dict[str, Any] = None
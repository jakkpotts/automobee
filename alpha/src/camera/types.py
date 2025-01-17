from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

class StreamStatus(Enum):
    ACTIVE = "active"
    IDLE = "idle"
    ERROR = "error"
    INITIALIZING = "initializing"
    OFFLINE = "offline"

@dataclass
class CameraStream:
    """Represents a camera stream with its metadata."""
    id: str
    name: str
    location: str
    lat: float
    lng: float
    image_url: str
    video_url: str
    status: StreamStatus = StreamStatus.INITIALIZING
    metadata: Dict[str, Any] = field(default_factory=dict)

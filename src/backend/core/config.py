"""Configuration classes for AutomoBee services."""

from dataclasses import dataclass
from typing import Tuple, List
from pathlib import Path

@dataclass
class ModelConfig:
    """Configuration for vehicle classification model."""
    
    # Model settings
    architecture: str = 'tf_efficientnetv2_s'
    device: str = 'cpu'
    input_size: Tuple[int, int] = (384, 384)
    top_k: int = 5
    scale_factor: float = 100.0
    
    # Paths
    cache_dir: str = 'model_cache'
    weights_file: str = 'vmmr_weights.pth'
    labels_file: str = 'vmmr_labels.json'
    weights_url: str = 'https://mobilewireless.tech/vmmr_weights.pth'
    
    # Preprocessing
    normalization_mean: List[float] = (0.485, 0.456, 0.406)
    normalization_std: List[float] = (0.229, 0.224, 0.225)
    
    def __post_init__(self):
        """Validate and process configuration after initialization."""
        # Convert paths to Path objects
        self.cache_dir = Path(self.cache_dir)
        self.weights_file = Path(self.weights_file)
        self.labels_file = Path(self.labels_file)
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(exist_ok=True)

@dataclass
class DetectionConfig:
    """Configuration for vehicle detection service."""
    
    # Detection settings
    confidence_threshold: float = 0.65
    nms_threshold: float = 0.45
    max_detections: int = 100
    
    # Processing
    batch_size: int = 8
    enable_batching: bool = True
    
    # Performance
    enable_gpu: bool = True
    half_precision: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.confidence_threshold < 0 or self.confidence_threshold > 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if self.nms_threshold < 0 or self.nms_threshold > 1:
            raise ValueError("nms_threshold must be between 0 and 1") 
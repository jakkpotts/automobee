"""
Vehicle classification service for AutomoBee.
Handles vehicle make, model, and color detection using deep learning models.
"""

import json
import torch
from PIL import Image
import logging
import cv2
import numpy as np
from pathlib import Path
import timm
from torchvision import transforms
from typing import Dict, Optional, List

from src.backend.services.detection.color_classifier import VehicleColorClassifier
from src.backend.utils.model_utils import download_model_weights
from src.backend.utils.image_utils import preprocess_image
from src.backend.core.config import ModelConfig

logger = logging.getLogger(__name__)

class VehicleClassificationService:
    """Service for classifying vehicles in images."""
    
    def __init__(self, model_config: ModelConfig):
        """
        Initialize the vehicle classification service.
        
        Args:
            model_config: Configuration for model paths and parameters
        """
        self.logger = logging.getLogger(__name__)
        self.initialized = False
        self.model = None
        self.model_config = model_config
        self.model_cache_dir = Path(model_config.cache_dir)
        self.model_cache_dir.mkdir(exist_ok=True)
        
        # Initialize color classifier
        self.color_classifier = VehicleColorClassifier()
        
        try:
            self._initialize_device()
            self._load_model()
            self._setup_processor()
            self.initialized = True
            
        except Exception as e:
            self.initialized = False
            self.logger.error(f"Failed to initialize vehicle classifier: {e}", exc_info=True)
            raise

    def _initialize_device(self):
        """Initialize the compute device (CPU/GPU/MPS)."""
        self.device = self.model_config.device
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        
        self.logger.info(f"Using device: {self.device}")

    def _load_model(self):
        """Load the vehicle classification model."""
        self.logger.info("Loading VMMR labels")
        self._load_vmmr_labels()
        
        if not hasattr(self, 'id2label') or not self.id2label:
            raise ValueError("id2label is not loaded")
            
        self.logger.info("Loading model")
        self.model = timm.create_model(
            self.model_config.architecture,
            pretrained=False,
            num_classes=len(self.id2label)
        )
        
        # Load weights
        weights_path = self.model_cache_dir / self.model_config.weights_file
        if not weights_path.exists():
            download_model_weights(self.model_config.weights_url, weights_path)
            
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        
        # Setup model for inference
        self.model = self.model.to(self.device)
        self.model.eval()
        
        if self.device.type == 'cuda':
            self.model = self.model.half()

    def _setup_processor(self):
        """Setup image preprocessing pipeline."""
        self.processor = transforms.Compose([
            transforms.Resize(self.model_config.input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.model_config.normalization_mean,
                std=self.model_config.normalization_std
            )
        ])

    def classify_vehicle(self, image: Image.Image) -> Optional[Dict]:
        """
        Classify a vehicle in an image.
        
        Args:
            image: PIL Image containing the vehicle
            
        Returns:
            Dictionary with classification results or None if failed
        """
        if not self.initialized:
            self.logger.error("Classifier is not initialized")
            return None

        try:
            # Get color classification
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            color_result = self.color_classifier.classify_color(opencv_image)
            
            # Prepare image for model
            inputs = self.processor(image).unsqueeze(0).to(self.device)
            if self.device.type == 'cuda':
                inputs = inputs.half()
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(inputs)
                probs = torch.softmax(outputs * self.model_config.scale_factor, dim=-1)[0]
                
                # Get top predictions
                top_probs, top_indices = torch.topk(probs, self.model_config.top_k)
                
                predictions = []
                for prob, idx in zip(top_probs, top_indices):
                    label = self.id2label.get(str(idx.item()), "unknown")
                    make, model, year = self._parse_vmmr_label(label)
                    
                    prediction = {
                        'make': make,
                        'model': model,
                        'year': year,
                        'label': label,
                        'confidence': prob.item(),
                        'color': color_result
                    }
                    predictions.append(prediction)
                
                return predictions[0]
            
        except Exception as e:
            self.logger.error(f"Error in vehicle classification: {e}", exc_info=True)
            return None

    def match_vehicle(self, 
                     image: Image.Image,
                     target_make: Optional[str] = None,
                     target_model: Optional[str] = None,
                     target_color: Optional[str] = None,
                     confidence_threshold: float = 0.65) -> bool:
        """
        Check if a vehicle matches specified criteria.
        
        Args:
            image: PIL Image of the vehicle
            target_make: Optional make to match (e.g., 'ford', 'toyota')
            target_model: Optional model to match (e.g., 'f150', 'camry')
            target_color: Optional color to match (supports variations)
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            bool: True if vehicle matches all specified criteria
        """
        try:
            result = self.classify_vehicle(image)
            if not result:
                return False

            matches = True
            
            if target_make and matches:
                matches = target_make.lower() in result['make'].lower()
                
            if target_model and matches:
                matches = target_model.lower() in result['model'].lower()
                
            if matches:
                matches = result['confidence'] >= confidence_threshold
            
            if matches and target_color is not None:
                matches = self.color_classifier.match_color(
                    target_color,
                    result['color']
                )
            
            if matches:
                self.logger.info(f"\n🎯 Vehicle Match Found!")
                self.logger.info(f"   Make/Model: {result['make']} {result['model']}")
                self.logger.info(f"   Confidence: {result['confidence']*100:.1f}%")
                if target_color:
                    self.logger.info(f"   Color Match: {result['color']}")
                
            return matches

        except Exception as e:
            self.logger.error(f"Error in vehicle matching: {e}")
            return False

    def _parse_vmmr_label(self, label: str) -> tuple:
        """Parse VMMRdb label format."""
        try:
            parts = label.split('_')
            make = parts[0].capitalize()
            model = parts[1].capitalize() if len(parts) > 1 else "Unknown"
            year = parts[2] if len(parts) > 2 else "Unknown"
            return make, model, year
        except Exception:
            return "Unknown", "Unknown", "Unknown" 
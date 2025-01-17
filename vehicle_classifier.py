import json
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image
import logging
from logger_config import logger
from typing import Dict, Optional
from pathlib import Path
import timm
import os
import requests
from tqdm import tqdm
from torchvision import transforms
import cv2
import numpy as np
from src.backend.services.color_classifier import VehicleColorClassifier

logger = logging.getLogger(__name__)

class VehicleMakeModelClassifier:
    def __init__(self, model_path: str = 'model_cache/vmmr_weights.pth', device: str = 'cpu'):
        self.logger = self._setup_logging()
        self.initialized = False
        self.model = None
        self.model_cache_dir = Path('model_cache')
        self.model_cache_dir.mkdir(exist_ok=True)
        
        # Initialize color classifier
        self.color_classifier = VehicleColorClassifier()
    
        try:
            # Initialize device
            self.device = device
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            
            self.logger.info(f"Using device: {self.device}")

            model_path = self.model_cache_dir / 'vmmr_weights.pth'
        
            # Load labels first
            self.logger.info("Loading VMMR labels")
            self._load_vmmr_labels()
        
            if not hasattr(self, 'id2label') or not self.id2label:
                self.logger.error("id2label is not loaded. Ensure vmmr_labels.json is populated.")
                raise ValueError("id2label is not loaded.")
            
            # Load the model with the proper number of classes
            self.logger.info("Loading model")
            self.model = timm.create_model('tf_efficientnetv2_s', pretrained=False, num_classes=len(self.id2label))
        
            # Setup image preprocessing
            self.logger.info("Setting up image preprocessing")
            self.processor = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        
            # Move model to device and set eval mode
            self.logger.info("Moving model to device and setting eval mode")
            self.model = self.model.to(self.device)
            self.model.eval()
        
            if torch.cuda.is_available():
                self.logger.info("Enabling FP16 for faster inference")
                self.model = self.model.half()  # Enable FP16 for faster inference

            self.initialized = True
         
            # Verify model
            self.logger.info("Verifying model")
            if not self._verify_model():
                self.initialized = False
                raise RuntimeError("Model verification failed")
            
            self.logger.info("VehicleMakeModelClassifier initialized successfully")
    
        except Exception as e:
            self.initialized = False
            self.logger.error(f"Failed to initialize VMMRdb classifier: {e}", exc_info=True)
            raise
        
    def _download_vmmr_weights(self, save_path: Path):
        """Download VMMRdb weights"""
        try:
            # You'll need to host these weights somewhere accessible
            # or include them in your project
            weights_url = "https://mobilewireless.tech/vmmr_weights.pth"
            response = requests.get(weights_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            with open(save_path, 'wb') as f, tqdm(
                total=total_size,
                unit='iB',
                unit_scale=True
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)
        except Exception as e:
            self.logger.error(f"Error downloading weights: {e}")
            raise

    def _load_vmmr_labels(self):
        """Load VMMRdb labels"""
        try:
            # Load label mapping from VMMRdb
            labels_path = self.model_cache_dir / 'vmmr_labels.json'
            if not labels_path.exists():
                raise FileNotFoundError("Labels file not found. Run prepare_vmmr.py first.")

            with open(labels_path) as f:
                label_data = json.load(f)
                
            # Extract label mappings from the data structure
            self.id2label = label_data['id2label']
            self.label2id = label_data['label2id']
            
            # Cache F-150 related labels
            self.f150_variants = label_data.get('metadata', {}).get('f150_variants', [])
            self.f150_label_ids = {
                label_id for label_id, label in self.id2label.items()
                if 'ford' in label.lower() and ('f150' in label.lower() or 'f-150' in label.lower())
            }
            
            self.logger.info(f"Loaded {len(self.id2label)} labels")
            self.logger.info(f"Found {len(self.f150_variants)} F-150 variants")
            
        except Exception as e:
            self.logger.error(f"Error loading VMMRdb labels: {e}")
            raise

    def _verify_model(self) -> bool:
        """Verify model functionality with test images"""
        try:
            # Create a simple test image
            self.logger.info("Creating test image for model verification")
            test_image = Image.new('RGB', (224, 224), color='white')
            
            self.logger.info("Classifying test image")
            result = self.classify_vehicle(test_image)
            
            if result is None:
                self.logger.error("Model verification failed: No result returned")
                return False
            #Log the result
            self.logger.info(f"Model verification result: {result}")
            
            required_keys = ['make', 'model', 'confidence']
            missing_keys = [key for key in required_keys if key not in result]
            
            if missing_keys:
                self.logger.error(f"Model verification failed: Missing keys in result: {missing_keys}")
                return False
            
            self.logger.info("Model verification succeeded")
            return True
            
        except Exception as e:
            self.logger.error(f"Model verification failed: {e}", exc_info=True)
            return False

    def _setup_logging(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def is_initialized(self) -> bool:
        """Check if the classifier is properly initialized"""
        return self.initialized and self.model is not None and self.processor is not None

    def classify_vehicle(self, image: Image.Image) -> Optional[Dict]:
        """Classify vehicle using VMMRdb model and detect color"""
        if not self.is_initialized():
            self.logger.error("Classifier is not initialized")
            self.logger.error(f"Initialization state: initialized={self.initialized}, model={'present' if self.model else 'missing'}, processor={'present' if self.processor else 'missing'}")
            return None

        try:
            # Convert PIL Image to OpenCV format for color detection
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Get color classification
            color_result = self.color_classifier.classify_color(opencv_image)
            
            #Add debug logging
            self.logger.debug(f"Processing image of size: {image.size}")
            # Preprocess image
            inputs = self.processor(image).unsqueeze(0)
            self.logger.debug(f"Input device: {inputs.device}")
            self.logger.debug(f"Model device: {next(self.model.parameters()).device}")
            inputs = inputs.to(self.device)
            
            if self.device.type == 'cuda':
                inputs = inputs.half()
            
            with torch.no_grad():
                self.logger.debug(f"Input tensor shape: {inputs.shape}")
                
                outputs = self.model(inputs)
                self.logger.debug(f"Output tensor shape: {outputs.shape}")
                self.logger.debug(f"Raw output values: {outputs[0][:5]}")
                scale_factor = 100.0  # You'll need to tune this value
                outputs = outputs * scale_factor
                probs = torch.softmax(outputs, dim=-1)[0]
                self.logger.debug(f"Softmax probabilities: {probs[:5]}")
                
                # Get top 5 predictions
                top5_probs, top5_indices = torch.topk(probs, 5)
                
                predictions = []
                for prob, idx in zip(top5_probs, top5_indices):
                    label = self.id2label.get(str(idx.item()), "unknown")
                    make, model, year = self._parse_vmmr_label(label)
                    
                    prediction = {
                        'make': make,
                        'model': model,
                        'year': year,
                        'label': label,
                        'confidence': prob.item(),
                        'color': color_result  # Add color classification results
                    }
                    predictions.append(prediction)
                
                return predictions[0]
            
        except Exception as e:
            self.logger.error(f"Error in vehicle classification: {e}", exc_info=True)
            return None

    def _parse_vmmr_label(self, label: str) -> tuple:
        """Parse VMMRdb label format"""
        try:
            parts = label.split('_')
            make = parts[0].capitalize()
            model = parts[1].capitalize() if len(parts) > 1 else "Unknown"
            year = parts[2] if len(parts) > 2 else "Unknown"
            return make, model, year
        except Exception:
            return "Unknown", "Unknown", "Unknown"

    def match_vehicle(self, image: Image.Image, 
                      target_make: Optional[str] = None,
                      target_model: Optional[str] = None,
                      target_color: Optional[str] = None,
                      confidence_threshold: float = 0.65) -> bool:
        """
        Check if vehicle matches specified criteria (make, model, color).
        
        Args:
            image: PIL Image of the vehicle
            target_make: Optional make to match (e.g., 'ford', 'toyota')
            target_model: Optional model to match (e.g., 'f150', 'camry')
            target_color: Optional color to match (supports variations like 'grey'/'gray'/'silver')
            confidence_threshold: Minimum confidence for model match
            
        Returns:
            bool: True if vehicle matches all specified criteria
        """
        try:
            result = self.classify_vehicle(image)
            if not result:
                return False

            # Start with True and apply each filter if specified
            matches = True
            
            # Check make if specified
            if target_make and matches:
                matches = target_make.lower() in result['make'].lower()
                
            # Check model if specified
            if target_model and matches:
                matches = target_model.lower() in result['model'].lower()
                
            # Check confidence threshold
            if matches:
                matches = result['confidence'] >= confidence_threshold
            
            # Check color if specified and other criteria matched
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
            
    def is_ford_f150(self, image: Image.Image, confidence_threshold: float = 0.65, target_color: Optional[str] = None) -> bool:
        """
        Legacy method for F-150 detection. Use match_vehicle() instead.
        
        @deprecated Use match_vehicle() with target_make='ford', target_model='f150' instead
        """
        return self.match_vehicle(
            image=image,
            target_make='ford',
            target_model='f150',
            target_color=target_color,
            confidence_threshold=confidence_threshold
        )

    def _extract_model(self, label: str) -> str:
        """Extract model from label with F-150 specific handling."""
        try:
            parts = label.split()
            if "F-150" in label:
                return "f-150"
            # ...handle other models...
            return parts[1].lower() if len(parts) > 1 else "unknown"
        except Exception:
            return "unknown"

    def _load_model(self, model_path: str):
        """Load the ResNet50 model with proper state dict handling"""
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.logger.info(f"Loaded state dict with {len(state_dict)} layers")
            
            model = timm.create_model('resnet50', pretrained=False, num_classes=len(self.id2label))
            model_state = model.state_dict()
            self.logger.info(f"Model requires {len(model_state)} layers")
            # In _load_model:
            self.logger.info(f"Model output range: {model.fc.weight.min().item()} to {model.fc.weight.max().item()}")
            
            missing_keys = [k for k in model_state.keys() if k not in state_dict]
            if missing_keys:
                self.logger.warning(f"Missing keys in state dict: {missing_keys}")
                
            model.load_state_dict(state_dict)
            return model
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
        
        
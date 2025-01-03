from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image
import logging
from typing import Dict, Optional
import json

class VehicleMakeModelClassifier:
    def __init__(self):
        self.logger = self._setup_logging()
        
        # Load model and processor
        try:
            # Use Microsoft's ResNet-50 model
            model_name = "microsoft/resnet-50"
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageClassification.from_pretrained(model_name)
            
            # Define our own labels for vehicle classification
            self.labels = {
                0: "ford_f150",
                1: "other_truck",
                2: "other_vehicle"
            }
            
            # Move model to available device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Load custom weights if available
            weights_path = "models/ford_f150_classifier.pth"
            try:
                state_dict = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.logger.info("Loaded custom F-150 classifier weights")
            except:
                self.logger.warning("No custom weights found, using base model")
            
            self.logger.info(f"Vehicle classifier loaded successfully on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load vehicle classifier: {e}")
            raise

    def _setup_logging(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def classify_vehicle(self, image: Image.Image) -> Optional[Dict]:
        """
        Classify vehicle make and model
        Returns dict with make, model, and confidence if successful
        """
        try:
            # Prepare image
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            # Get prediction
            pred_idx = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_idx].item()
            
            prediction = {
                'label': self.labels[pred_idx],
                'confidence': confidence
            }
            
            # Parse make/model from label
            if prediction['label'] == 'ford_f150':
                prediction.update({
                    'make': 'ford',
                    'model': 'f-150',
                })
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error classifying vehicle: {e}")
            return None

    def is_ford_f150(self, image: Image.Image, confidence_threshold: float = 0.7) -> bool:
        """
        Specifically check if vehicle is a Ford F-150
        """
        try:
            prediction = self.classify_vehicle(image)
            if not prediction:
                return False
                
            return (prediction['label'] == 'ford_f150' and 
                   prediction['confidence'] >= confidence_threshold)
            
        except Exception as e:
            self.logger.error(f"Error checking for Ford F-150: {e}")
            return False 
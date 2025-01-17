import unittest
from unittest.mock import Mock, patch
from PIL import Image
import numpy as np
from src.backend.services.vehicle_classifier import VehicleMakeModelClassifier

class TestVehicleMakeModelClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = VehicleMakeModelClassifier()
        
    def _create_mock_result(self, make: str, model: str, confidence: float, color: dict) -> dict:
        """Helper to create mock classification results"""
        return {
            'make': make,
            'model': model,
            'confidence': confidence,
            'color': color,
            'label': f"{make}_{model}_2024",
            'year': "2024"
        }
        
    @patch('src.backend.services.vehicle_classifier.VehicleMakeModelClassifier.classify_vehicle')
    def test_match_vehicle_make_only(self, mock_classify):
        """Test matching by make only"""
        mock_result = self._create_mock_result(
            make="Toyota",
            model="Camry",
            confidence=0.8,
            color={'silver': 0.8, 'gray': 0.2}
        )
        mock_classify.return_value = mock_result
        
        # Should match Toyota
        self.assertTrue(self.classifier.match_vehicle(
            image=Mock(),
            target_make="toyota"
        ))
        
        # Should not match Ford
        self.assertFalse(self.classifier.match_vehicle(
            image=Mock(),
            target_make="ford"
        ))
        
    @patch('src.backend.services.vehicle_classifier.VehicleMakeModelClassifier.classify_vehicle')
    def test_match_vehicle_model_only(self, mock_classify):
        """Test matching by model only"""
        mock_result = self._create_mock_result(
            make="Honda",
            model="Civic",
            confidence=0.9,
            color={'blue': 0.9, 'black': 0.1}
        )
        mock_classify.return_value = mock_result
        
        # Should match Civic
        self.assertTrue(self.classifier.match_vehicle(
            image=Mock(),
            target_model="civic"
        ))
        
        # Should not match Accord
        self.assertFalse(self.classifier.match_vehicle(
            image=Mock(),
            target_model="accord"
        ))
        
    @patch('src.backend.services.vehicle_classifier.VehicleMakeModelClassifier.classify_vehicle')
    def test_match_vehicle_color_only(self, mock_classify):
        """Test matching by color only"""
        mock_result = self._create_mock_result(
            make="Ford",
            model="F150",
            confidence=0.85,
            color={'blue': 0.9, 'black': 0.1}
        )
        mock_classify.return_value = mock_result
        
        # Should match blue
        self.assertTrue(self.classifier.match_vehicle(
            image=Mock(),
            target_color="blue"
        ))
        
        # Should match navy (color alias)
        self.assertTrue(self.classifier.match_vehicle(
            image=Mock(),
            target_color="navy"
        ))
        
        # Should not match red
        self.assertFalse(self.classifier.match_vehicle(
            image=Mock(),
            target_color="red"
        ))
        
    @patch('src.backend.services.vehicle_classifier.VehicleMakeModelClassifier.classify_vehicle')
    def test_match_vehicle_all_criteria(self, mock_classify):
        """Test matching with all criteria"""
        mock_result = self._create_mock_result(
            make="Ford",
            model="F150",
            confidence=0.95,
            color={'red': 0.8, 'orange': 0.2}
        )
        mock_classify.return_value = mock_result
        
        # Should match all criteria
        self.assertTrue(self.classifier.match_vehicle(
            image=Mock(),
            target_make="ford",
            target_model="f150",
            target_color="red",
            confidence_threshold=0.9
        ))
        
        # Should fail on make
        self.assertFalse(self.classifier.match_vehicle(
            image=Mock(),
            target_make="toyota",
            target_model="f150",
            target_color="red"
        ))
        
        # Should fail on model
        self.assertFalse(self.classifier.match_vehicle(
            image=Mock(),
            target_make="ford",
            target_model="ranger",
            target_color="red"
        ))
        
        # Should fail on color
        self.assertFalse(self.classifier.match_vehicle(
            image=Mock(),
            target_make="ford",
            target_model="f150",
            target_color="blue"
        ))
        
        # Should fail on confidence
        self.assertFalse(self.classifier.match_vehicle(
            image=Mock(),
            target_make="ford",
            target_model="f150",
            target_color="red",
            confidence_threshold=0.99
        ))
        
    @patch('src.backend.services.vehicle_classifier.VehicleMakeModelClassifier.classify_vehicle')
    def test_legacy_ford_f150_method(self, mock_classify):
        """Test that legacy method still works"""
        mock_result = self._create_mock_result(
            make="Ford",
            model="F150",
            confidence=0.9,
            color={'blue': 0.9, 'black': 0.1}
        )
        mock_classify.return_value = mock_result
        
        # Should work with legacy method
        self.assertTrue(self.classifier.is_ford_f150(
            image=Mock(),
            target_color="blue"
        ))
        
        # Should fail with wrong color
        self.assertFalse(self.classifier.is_ford_f150(
            image=Mock(),
            target_color="red"
        ))
        
if __name__ == '__main__':
    unittest.main() 
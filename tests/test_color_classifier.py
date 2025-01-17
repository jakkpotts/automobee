import unittest
import cv2
import numpy as np
from src.backend.services.color_classifier import VehicleColorClassifier, VehicleColor

class TestVehicleColorClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = VehicleColorClassifier()
        
    def _create_color_image(self, bgr_color: tuple, size: tuple = (100, 100)) -> np.ndarray:
        """Helper to create test images"""
        return np.full((*size, 3), bgr_color, dtype=np.uint8)
        
    def test_basic_colors(self):
        """Test basic color detection"""
        # Test pure colors
        test_colors = {
            "red": ((0, 0, 255), "red"),
            "blue": ((255, 0, 0), "blue"),
            "white": ((255, 255, 255), "white"),
            "black": ((0, 0, 0), "black")
        }
        
        for name, (bgr, expected) in test_colors.items():
            img = self._create_color_image(bgr)
            result = self.classifier.classify_color(img)
            self.assertIn(expected, result)
            self.assertGreater(result[expected], 0.5)
            
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Test empty image
        empty_img = np.array([], dtype=np.uint8)
        result = self.classifier.classify_color(empty_img)
        self.assertIn("error", result)
        
        # Test single pixel image
        single_pixel = np.array([[[255, 0, 0]]], dtype=np.uint8)
        result = self.classifier.classify_color(single_pixel)
        self.assertIsInstance(result, dict)
        
        # Test image with mixed colors
        mixed_img = np.zeros((100, 100, 3), dtype=np.uint8)
        mixed_img[:50, :] = [255, 0, 0]  # Blue top half
        mixed_img[50:, :] = [0, 0, 255]  # Red bottom half
        result = self.classifier.classify_color(mixed_img)
        self.assertGreaterEqual(len(result), 2)
        
    def test_preprocessing(self):
        """Test image preprocessing"""
        # Create test image
        test_img = self._create_color_image((100, 100, 100))
        
        # Test preprocessing doesn't crash
        processed = self.classifier.preprocess_image(test_img)
        self.assertEqual(processed.shape, test_img.shape)
        self.assertEqual(processed.dtype, test_img.dtype)
        
    def test_performance(self):
        """Test performance benchmarks"""
        import time
        
        # Create large test image
        large_img = self._create_color_image((255, 0, 0), (1920, 1080))
        
        # Measure classification time
        start_time = time.time()
        self.classifier.classify_color(large_img)
        elapsed = time.time() - start_time
        
        # Should process HD image in under 100ms
        self.assertLess(elapsed, 0.1)
        
    def test_color_matching(self):
        """Test color matching with variations"""
        # Create a predominantly gray image
        img = self._create_color_image((128, 128, 128))
        colors = self.classifier.classify_color(img)
        
        # Test exact match
        self.assertTrue(self.classifier.match_color("gray", colors))
        
        # Test variations
        self.assertTrue(self.classifier.match_color("grey", colors))
        self.assertTrue(self.classifier.match_color("silver", colors))
        
        # Test aliases
        self.assertTrue(self.classifier.match_color("graphite", colors))
        self.assertTrue(self.classifier.match_color("charcoal", colors))
        
        # Test non-matches
        self.assertFalse(self.classifier.match_color("red", colors))
        
    def test_optional_color_matching(self):
        """Test optional color matching"""
        img = self._create_color_image((255, 0, 0))  # Red image
        colors = self.classifier.classify_color(img)
        
        # None should always match
        self.assertTrue(self.classifier.match_color(None, colors))
        
        # Test color variations
        self.assertTrue(self.classifier.match_color("red", colors))
        self.assertTrue(self.classifier.match_color("maroon", colors))
        self.assertTrue(self.classifier.match_color("burgundy", colors))
        
    def test_neutral_color_matching(self):
        """Test matching of neutral colors (white/gray/silver)"""
        # Test light gray
        light_gray = self._create_color_image((192, 192, 192))
        light_colors = self.classifier.classify_color(light_gray)
        
        self.assertTrue(self.classifier.match_color("silver", light_colors))
        self.assertTrue(self.classifier.match_color("grey", light_colors))
        self.assertTrue(self.classifier.match_color("white", light_colors))
        
        # Test dark gray
        dark_gray = self._create_color_image((64, 64, 64))
        dark_colors = self.classifier.classify_color(dark_gray)
        
        self.assertTrue(self.classifier.match_color("gray", dark_colors))
        self.assertTrue(self.classifier.match_color("charcoal", dark_colors))
        
    def test_color_confidence(self):
        """Test color confidence thresholds"""
        # Create mixed color image
        mixed = np.zeros((100, 100, 3), dtype=np.uint8)
        mixed[:70, :] = [255, 0, 0]  # 70% red
        mixed[70:, :] = [0, 0, 255]  # 30% blue
        
        colors = self.classifier.classify_color(mixed)
        
        # Red should match with default threshold
        self.assertTrue(self.classifier.match_color("red", colors))
        
        # Blue should not match with high threshold
        self.assertFalse(self.classifier.match_color("blue", colors, threshold=0.5))
        
        # But should match with lower threshold
        self.assertTrue(self.classifier.match_color("blue", colors, threshold=0.2))
        
if __name__ == '__main__':
    unittest.main() 
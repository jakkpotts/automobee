import cv2
import numpy as np
from typing import Dict, Tuple, List, Optional, Set
import logging
from dataclasses import dataclass
from enum import Enum
from difflib import get_close_matches

logger = logging.getLogger(__name__)

class VehicleColor(Enum):
    BLACK = "black"
    WHITE = "white"
    SILVER = "silver"
    GRAY = "gray"
    RED = "red"
    BLUE = "blue"
    GREEN = "green"
    YELLOW = "yellow"
    BROWN = "brown"
    ORANGE = "orange"

@dataclass
class ColorRange:
    name: VehicleColor
    lower: np.ndarray
    upper: np.ndarray
    aliases: Set[str]

class VehicleColorClassifier:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._initialize_color_ranges()
        self._initialize_color_aliases()

    def _initialize_color_aliases(self):
        """Initialize color aliases for flexible matching"""
        self.color_aliases = {
            VehicleColor.BLACK: {"black", "blk", "midnight"},
            VehicleColor.WHITE: {"white", "pearl", "ivory", "snow"},
            VehicleColor.SILVER: {"silver", "metallic", "chrome", "platinum"},
            VehicleColor.GRAY: {"gray", "grey", "charcoal", "graphite", "gunmetal"},
            VehicleColor.RED: {"red", "maroon", "burgundy", "crimson", "wine"},
            VehicleColor.BLUE: {"blue", "navy", "azure", "cobalt"},
            VehicleColor.GREEN: {"green", "emerald", "forest", "olive"},
            VehicleColor.YELLOW: {"yellow", "gold", "amber"},
            VehicleColor.BROWN: {"brown", "bronze", "copper", "tan", "beige"},
            VehicleColor.ORANGE: {"orange", "coral", "rust"}
        }
        
        # Create reverse lookup for all aliases
        self.alias_to_color = {}
        for color, aliases in self.color_aliases.items():
            for alias in aliases:
                self.alias_to_color[alias.lower()] = color

    def _initialize_color_ranges(self):
        """Initialize HSV color ranges for vehicle colors"""
        self.color_ranges = [
            ColorRange(VehicleColor.BLACK, np.array([0, 0, 0]), np.array([180, 255, 30]), set()),
            ColorRange(VehicleColor.WHITE, np.array([0, 0, 200]), np.array([180, 30, 255]), set()),
            ColorRange(VehicleColor.SILVER, np.array([0, 0, 120]), np.array([180, 30, 220]), set()),
            ColorRange(VehicleColor.GRAY, np.array([0, 0, 40]), np.array([180, 30, 190]), set()),
            # Expanded red range to catch variations
            ColorRange(VehicleColor.RED, np.array([0, 100, 100]), np.array([10, 255, 255]), set()),
            ColorRange(VehicleColor.BLUE, np.array([100, 100, 100]), np.array([130, 255, 255]), set()),
            ColorRange(VehicleColor.GREEN, np.array([40, 100, 100]), np.array([80, 255, 255]), set()),
            ColorRange(VehicleColor.YELLOW, np.array([20, 100, 100]), np.array([40, 255, 255]), set()),
            ColorRange(VehicleColor.BROWN, np.array([10, 100, 20]), np.array([20, 255, 200]), set()),
            ColorRange(VehicleColor.ORANGE, np.array([10, 100, 100]), np.array([25, 255, 255]), set())
        ]
        
        # Add second range for red (handles wrap-around in HSV)
        self.color_ranges.append(
            ColorRange(VehicleColor.RED, np.array([170, 100, 100]), np.array([180, 255, 255]), set())
        )

    def match_color(self, target_color: Optional[str], detected_colors: Dict[str, float], threshold: float = 0.3) -> bool:
        """
        Match a target color against detected colors, handling variations and aliases
        
        Args:
            target_color: The target color to match (can be None for no color preference)
            detected_colors: Dictionary of detected colors and their confidence scores
            threshold: Minimum confidence threshold for matching
            
        Returns:
            bool: True if colors match or target_color is None
        """
        if target_color is None:
            return True
            
        target_color = target_color.lower()
        
        # Try exact match first
        if target_color in detected_colors and detected_colors[target_color] >= threshold:
            return True
            
        # Try alias matching
        matched_colors = get_close_matches(target_color, self.alias_to_color.keys(), n=3, cutoff=0.8)
        if matched_colors:
            for matched_alias in matched_colors:
                color = self.alias_to_color[matched_alias].value
                if color in detected_colors and detected_colors[color] >= threshold:
                    return True
                    
        # Special handling for gray/silver/white variations
        if target_color in {'gray', 'grey', 'silver', 'white'}:
            neutral_colors = {'gray', 'silver', 'white'}
            for color in neutral_colors:
                if color in detected_colors and detected_colors[color] >= threshold:
                    return True
                    
        return False

    def classify_color(self, image) -> Dict[str, float]:
        """
        Classify the dominant color of a vehicle in the image
        
        Args:
            image: OpenCV/numpy image in BGR format
            
        Returns:
            Dict with color name and confidence score
        """
        try:
            # Preprocess image
            image = self.preprocess_image(image)
            
            # Convert to HSV color space
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Calculate dominant color
            color_scores = {}
            
            for color_range in self.color_ranges:
                mask = cv2.inRange(hsv, color_range.lower, color_range.upper)
                ratio = np.count_nonzero(mask) / mask.size
                
                # Accumulate scores for same colors (e.g. two red ranges)
                if color_range.name.value in color_scores:
                    color_scores[color_range.name.value] = max(color_scores[color_range.name.value], float(ratio))
                else:
                    color_scores[color_range.name.value] = float(ratio)
            
            # Normalize scores
            total = sum(color_scores.values())
            if total > 0:
                color_scores = {k: v/total for k, v in color_scores.items()}
            
            # Return top 3 colors with scores
            sorted_colors = sorted(color_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            return dict(sorted_colors)
            
        except Exception as e:
            self.logger.error(f"Error in color classification: {e}")
            return {"error": str(e)}

    def preprocess_image(self, image) -> np.ndarray:
        """Preprocess image for better color detection"""
        try:
            # Apply slight blur to reduce noise
            blurred = cv2.GaussianBlur(image, (5, 5), 0)
            
            # Apply CLAHE for better color distinction
            lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced = cv2.merge((cl, a, b))
            
            return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
        except Exception as e:
            self.logger.error(f"Error in image preprocessing: {e}")
            return image 
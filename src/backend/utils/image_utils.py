"""Utility functions for image processing."""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def preprocess_image(image: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Preprocess image for model input.
    
    Args:
        image: OpenCV/numpy image in BGR format
        target_size: Optional target size (height, width)
        
    Returns:
        Preprocessed image
    """
    try:
        # Apply slight blur to reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Apply CLAHE for better color distinction
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Resize if target size specified
        if target_size:
            enhanced = cv2.resize(enhanced, target_size)
            
        return enhanced
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return image

def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV format."""
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(image: np.ndarray) -> Image.Image:
    """Convert OpenCV image to PIL format."""
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def draw_detection_box(
    image: np.ndarray,
    box: Tuple[int, int, int, int],
    label: str,
    confidence: float,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw detection box with label on image.
    
    Args:
        image: OpenCV image to draw on
        box: Bounding box coordinates (x1, y1, x2, y2)
        label: Text label to draw
        confidence: Detection confidence
        color: Box color in BGR format
        thickness: Line thickness
        
    Returns:
        Image with detection box drawn
    """
    try:
        x1, y1, x2, y2 = box
        
        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label text
        label_text = f"{label} {confidence:.2f}"
        
        # Get text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, font, font_scale, thickness
        )
        
        # Draw label background
        cv2.rectangle(
            image,
            (x1, y1 - text_height - baseline),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            image,
            label_text,
            (x1, y1 - baseline),
            font,
            font_scale,
            (0, 0, 0),
            thickness
        )
        
        return image
        
    except Exception as e:
        logger.error(f"Error drawing detection box: {e}")
        return image 
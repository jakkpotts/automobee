"""Utility functions for model operations."""

import requests
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def download_model_weights(url: str, save_path: Path, chunk_size: int = 8192) -> bool:
    """
    Download model weights from URL.
    
    Args:
        url: URL to download weights from
        save_path: Path to save weights to
        chunk_size: Size of chunks to download
        
    Returns:
        bool: True if download successful
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(save_path, 'wb') as f, tqdm(
            total=total_size,
            unit='iB',
            unit_scale=True,
            desc=f"Downloading {save_path.name}"
        ) as pbar:
            for data in response.iter_content(chunk_size=chunk_size):
                size = f.write(data)
                pbar.update(size)
                
        return True
        
    except Exception as e:
        logger.error(f"Error downloading weights from {url}: {e}")
        if save_path.exists():
            save_path.unlink()
        return False

def verify_model_weights(weights_path: Path, expected_size: Optional[int] = None) -> bool:
    """
    Verify downloaded model weights.
    
    Args:
        weights_path: Path to weights file
        expected_size: Expected file size in bytes
        
    Returns:
        bool: True if verification successful
    """
    try:
        if not weights_path.exists():
            logger.error(f"Weights file not found: {weights_path}")
            return False
            
        if expected_size and weights_path.stat().st_size != expected_size:
            logger.error(f"Weights file size mismatch: {weights_path}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error verifying weights file {weights_path}: {e}")
        return False 
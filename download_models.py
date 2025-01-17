import torch
from ultralytics import YOLO
import timm
import os
import logging
from pathlib import Path
from datetime import datetime
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def backup_existing_models():
    """Backup existing models before downloading new ones"""
    try:
        model_cache = Path('model_cache')
        if not model_cache.exists():
            return
        
        backup_dir = Path('backups') / datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        for model_file in model_cache.glob('*.pt*'):
            backup_path = backup_dir / model_file.name
            shutil.copy2(model_file, backup_path)
            logger.info(f"Backed up {model_file.name} to {backup_path}")
            
    except Exception as e:
        logger.error(f"Error backing up models: {e}")
        raise

def download_models():
    """Download YOLOv8m and EfficientNetV2-S models"""
    try:
        # Create model cache directory
        model_cache = Path('model_cache')
        model_cache.mkdir(exist_ok=True)
        
        # Backup existing models
        backup_existing_models()
        
        # Download YOLOv8m
        logger.info("Downloading YOLOv8m...")
        model = YOLO('yolov8m.pt')
        model.save(model_cache / 'yolo_weights.pt')
        
        # Download EfficientNetV2-S
        logger.info("Downloading EfficientNetV2-S...")
        model = timm.create_model('tf_efficientnetv2_s', pretrained=True)
        torch.save(model.state_dict(), model_cache / 'vmmr_weights.pth')
        
        logger.info("Model downloads completed successfully!")
        
    except Exception as e:
        logger.error(f"Error downloading models: {e}")
        raise

def verify_models():
    """Verify downloaded models"""
    try:
        model_cache = Path('model_cache')
        
        # Verify YOLOv8m
        yolo_path = model_cache / 'yolo_weights.pt'
        if not yolo_path.exists():
            raise FileNotFoundError("YOLOv8m weights not found")
        
        # Load YOLO model to verify
        logger.info("Verifying YOLOv8m...")
        YOLO(str(yolo_path))
        
        # Verify EfficientNetV2-S
        vmmr_path = model_cache / 'vmmr_weights.pth'
        if not vmmr_path.exists():
            raise FileNotFoundError("VMMR weights not found")
        
        # Load EfficientNetV2-S to verify
        logger.info("Verifying EfficientNetV2-S...")
        model = timm.create_model('efficientnetv2_s', pretrained=False)
        model.load_state_dict(torch.load(vmmr_path))
        
        logger.info("Model verification completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error verifying models: {e}")
        return False

if __name__ == '__main__':
    try:
        download_models()
        if verify_models():
            logger.info("✅ Models downloaded and verified successfully!")
        else:
            logger.error("❌ Model verification failed!")
    except Exception as e:
        logger.error(f"❌ Failed to download or verify models: {e}")
        exit(1) 
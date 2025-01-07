import json
from pathlib import Path
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

def prepare_vmmr_dataset(dataset_path: str):
    """Process existing VMMRdb dataset and prepare files for classifier"""
    
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Could not find dataset at: {dataset_path}")
    
    logger.info(f"Using existing dataset at: {dataset_path}")
    
    cache_dir = Path('model_cache')
    cache_dir.mkdir(exist_ok=True)
    
    # Create label mapping from flat directory structure
    logger.info("Creating label mapping from dataset...")
    id2label = {}
    label2id = {}
    
    # Process directories in flat structure
    all_dirs = sorted(d for d in dataset_path.iterdir() if d.is_dir())
    
    for dir_path in tqdm(all_dirs, desc="Processing vehicle directories"):
        # Directory name format: make_model_year
        dir_name = dir_path.name.lower()
        
        # Split into components
        try:
            parts = dir_name.split('_')
            if len(parts) >= 3:
                make = parts[0]
                year = parts[-1]  # Last part is year
                model = '_'.join(parts[1:-1])  # Everything between make and year is model
                
                # Create standardized label
                label = f"{make}_{model}_{year}"
                id2label[str(len(id2label))] = label
                label2id[label] = str(len(label2id))
                
                if 'ford' in make and 'f150' in model or 'f-150' in model:
                    logger.info(f"Found F-150: {label}")
        except Exception as e:
            logger.warning(f"Could not process directory {dir_name}: {e}")
            continue
    
    # Save enhanced label mapping
    logger.info("Saving label mapping...")
    label_data = {
        'id2label': id2label,
        'label2id': label2id,
        'metadata': {
            'total_classes': len(id2label),
            'f150_variants': [
                label for label in id2label.values() 
                if 'ford' in label and ('f150' in label or 'f-150' in label)
            ]
        }
    }
    
    with open(cache_dir / 'vmmr_labels.json', 'w') as f:
        json.dump(label_data, f, indent=2)
        # Ensure labels are written correctly
        logger.info("vmmr_labels.json has been populated with label mappings.")
    
    logger.info(f"Created mapping for {len(id2label)} vehicle types")
    logger.info(f"Found {len(label_data['metadata']['f150_variants'])} F-150 variants")
    
    logger.info(f"Found {len(id2label)} classes")
    
    # Convert pretrained ResNet50 to VMMR format with correct number of classes
    logger.info(f"Converting model weights for {len(id2label)} classes...")
    model = timm.create_model('resnet50', pretrained=True, num_classes=len(id2label))
    
    # Save model weights
    save_path = cache_dir / 'vmmr_weights.pth'
    torch.save(model.state_dict(), save_path)
    logger.info(f"Saved model weights to {save_path}")
    
    # Verify weights
    state_dict = torch.load(save_path, weights_only=True)
    logger.info(f"Verified weights shape: {state_dict['fc.weight'].shape}")
    
    # Create sample test file
    logger.info("Creating test image...")
    test_img = Image.new('RGB', (224, 224))
    test_img.save(cache_dir / 'test_image.jpg')
    
    logger.info("Setup complete! Files created in model_cache/:")
    logger.info("- vmmr_labels.json")
    logger.info("- vmmr_weights.pth")
    logger.info("- test_image.jpg")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, help="Path to existing VMMRdb dataset directory")
    args = parser.parse_args()

    if args.dataset_path:
        dataset_path = args.dataset_path
    else:
        dataset_path = input("Enter path to existing VMMRdb dataset directory: ")
    
    try:
        prepare_vmmr_dataset(dataset_path)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"\nError: {e}")

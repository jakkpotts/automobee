import gdown
import tarfile
import shutil
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import os

def download_and_extract_vmmr():
    """
    Download and extract only Ford F-150 images from VMMRdb dataset
    """
    data_dir = Path('data/vehicles')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading VMMRdb dataset...")
    
    # VMMRdb dataset parts (hosted on Google Drive)
    file_ids = {
        'part1': '1LjzLh3T8-_JuGEVYZzJqvXrKF1-wGH5e',  # Contains Ford vehicles
    }
    
    for part, file_id in file_ids.items():
        output = f'data/vmmr_{part}.tar.gz'
        
        print(f"Downloading {part}...")
        gdown.download(id=file_id, output=output, quiet=False)
        
        print(f"Extracting {part}...")
        with tarfile.open(output, 'r:gz') as tar:
            # Only extract Ford F-150 images
            members = [m for m in tar.getmembers() 
                      if 'Ford_F150' in m.name or 'Ford_F-150' in m.name]
            tar.extractall(path=data_dir, members=members)
        
        os.remove(output)
    
    organize_f150_dataset(data_dir)

def organize_f150_dataset(data_dir: Path):
    """Organize F-150 images by color"""
    print("Organizing dataset...")
    
    temp_dir = data_dir / 'temp'
    temp_dir.mkdir(exist_ok=True)
    
    # Create color directories
    black_dir = temp_dir / 'black'
    other_dir = temp_dir / 'other'
    black_dir.mkdir(parents=True, exist_ok=True)
    other_dir.mkdir(parents=True, exist_ok=True)
    
    # Move files and classify by color
    for img_path in data_dir.glob('*.jpg'):
        try:
            # Load image and check if it's black
            img = cv2.imread(str(img_path))
            if is_black_vehicle(img):
                shutil.move(str(img_path), str(black_dir / img_path.name))
            else:
                shutil.move(str(img_path), str(other_dir / img_path.name))
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Replace original directory with organized one
    if (data_dir / 'organized').exists():
        shutil.rmtree(data_dir / 'organized')
    shutil.move(str(temp_dir), str(data_dir / 'organized'))
    
    print("Dataset organization complete!")

def is_black_vehicle(img):
    """
    Determine if vehicle is black based on color analysis
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define black color range in HSV
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 30])
    
    # Create mask for black pixels
    mask = cv2.inRange(hsv, lower_black, upper_black)
    
    # Calculate percentage of black pixels
    black_ratio = np.count_nonzero(mask) / mask.size
    
    return black_ratio > 0.4  # Threshold for black vehicle classification

if __name__ == "__main__":
    download_and_extract_vmmr() 
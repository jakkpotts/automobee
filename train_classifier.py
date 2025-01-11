import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from transformers import AutoImageProcessor, AutoModelForImageClassification
from pathlib import Path
from tqdm import tqdm
import numpy as np
import argparse

class VehicleDataset(Dataset):
    def __init__(self, image_dir, processor):
        self.image_dir = image_dir
        self.processor = processor
        self.images = []
        self.labels = []
        
        # Load images and labels
        for label_dir in os.listdir(image_dir):
            label_path = os.path.join(image_dir, label_dir)
            if os.path.isdir(label_path):
                label_idx = 0 if label_dir == "ford_f150" else 1 if label_dir == "other_truck" else 2
                for img_name in os.listdir(label_path):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        self.images.append(os.path.join(label_path, img_name))
                        self.labels.append(label_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        inputs = self.processor(image, return_tensors="pt")
        return {
            'pixel_values': inputs['pixel_values'].squeeze(),
            'labels': torch.tensor(self.labels[idx])
        }

def train_model(resume_from_checkpoint=None):
    print("\nInitializing training...")
    
    # Initialize model and processor
    model_name = "microsoft/resnet-50"
    print(f"Loading base model: {model_name}")
    processor = AutoImageProcessor.from_pretrained(model_name)
    
    # Load model with ignore_mismatched_sizes=True
    label2id = {
        "ford_f150": 0,
        "other_truck": 1,
        "other_vehicle": 2
    }
    id2label = {
        0: "ford_f150",
        1: "other_truck", 
        2: "other_vehicle"
    }

    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    
    # Setup training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = VehicleDataset("data/vehicles", processor)
    print(f"Total images found: {len(dataset)}")
    
    # Print class distribution
    labels = dataset.labels
    class_counts = {
        "ford_f150": labels.count(0),
        "other_truck": labels.count(1),
        "other_vehicle": labels.count(2)
    }
    print("\nClass distribution:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count} images")
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize training parameters and variables
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    num_epochs = 10
    start_epoch = 0  # Initialize start_epoch here
    best_loss = float('inf')
    
    # Initialize variables for interruption handling
    epoch = 0
    epoch_loss = float('inf')
    epoch_accuracy = 0
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    # Create output directory
    Path("models").mkdir(exist_ok=True)
    
    # Then handle checkpoint loading if needed
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        print("\n=== Resuming from checkpoint ===")
        checkpoint = torch.load(resume_from_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Only override if loading checkpoint
        best_loss = checkpoint['loss']
        print(f"Checkpoint details:")
        print(f"- Resuming from epoch: {start_epoch}/{num_epochs}")
        print(f"- Best loss achieved: {best_loss:.4f}")
        print(f"- Previous accuracy: {checkpoint['accuracy']:.2f}%")
        print("===============================\n")

    print("\nStarting training...")
    model.train()
    try:
        for epoch in range(start_epoch, num_epochs):
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0
            
            # Create progress bar for this epoch
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch in progress_bar:
                optimizer.zero_grad()
                
                inputs = {
                    'pixel_values': batch['pixel_values'].to(device),
                    'labels': batch['labels'].to(device)
                }
                
                outputs = model(**inputs)
                loss = outputs.loss
                
                # Calculate accuracy
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct_predictions += (predictions == inputs['labels']).sum().item()
                total_predictions += inputs['labels'].size(0)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Update progress bar
                current_loss = total_loss / (progress_bar.n + 1)
                current_accuracy = (correct_predictions / total_predictions) * 100
                progress_bar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'accuracy': f'{current_accuracy:.2f}%'
                })
            
            # Epoch summary
            epoch_loss = total_loss / len(dataloader)
            epoch_accuracy = (correct_predictions / total_predictions) * 100
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"Average Loss: {epoch_loss:.4f}")
            print(f"Accuracy: {epoch_accuracy:.2f}%")
            
            # Save best model
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                print("New best model! Saving checkpoint...")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                    'accuracy': epoch_accuracy
                }, "models/ford_f150_classifier_best.pth")
        
        # Save final model
        print("\nSaving final model...")
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            'accuracy': epoch_accuracy
        }, "models/ford_f150_classifier.pth")
        
        # Save the processor config
        processor.save_pretrained("models")
        print("\nTraining completed successfully!")
        print(f"Best loss achieved: {best_loss:.4f}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted! Saving checkpoint...")
        try:
            # Save interrupted model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss / max(1, len(dataloader)),  # Prevent division by zero
                'accuracy': (correct_predictions / max(1, total_predictions)) * 100  # Prevent division by zero
            }, "models/ford_f150_classifier_interrupted.pth")
            
            # Save the processor config
            processor.save_pretrained("models")
            print(f"Checkpoint saved at epoch {epoch+1}!")
        except Exception as e:
            print(f"\nError saving checkpoint: {str(e)}")
        finally:
            return

if __name__ == "__main__":
    # Create the data directory structure if it doesn't exist
    data_dirs = [
        "data/vehicles/ford_f150",
        "data/vehicles/other_truck",
        "data/vehicles/other_vehicle"
    ]
    
    for dir_path in data_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        
    if not any(os.listdir(dir_path) for dir_path in data_dirs):
        print("\nPlease add training images to the following directories:")
        print("data/vehicles/ford_f150/    - Add Ford F-150 images here")
        print("data/vehicles/other_truck/   - Add other truck images here")
        print("data/vehicles/other_vehicle/ - Add other vehicle images here")
        print("\nThen run this script again to train the model.")
        exit(1)
        
    train_model() 
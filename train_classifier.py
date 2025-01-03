import torch
import torchvision
import timm
from torch.utils.data import DataLoader
from torchvision import transforms
import json
from pathlib import Path
import wandb
from tqdm import tqdm
import logging
from datetime import datetime

class F150Detector:
    def __init__(self, data_dir: str, batch_size: int = 32):
        # Setup logging
        self.logger = self._setup_logging()
        
        # Setup device - optimize for M2
        self.device = (
            "mps" if torch.backends.mps.is_available() 
            else "cuda" if torch.cuda.is_available() 
            else "cpu"
        )
        self.logger.info(f"Using device: {self.device}")
        
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        
        # Data augmentation and normalization
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        try:
            # Create datasets and loaders
            self.train_dataset, self.val_dataset = self._create_datasets()
            self.train_loader = DataLoader(
                self.train_dataset, 
                batch_size=batch_size, 
                shuffle=True,
                num_workers=2,  # Reduced for MacBook
                pin_memory=True if self.device != "cpu" else False
            )
            self.val_loader = DataLoader(
                self.val_dataset, 
                batch_size=batch_size,
                num_workers=2,
                pin_memory=True if self.device != "cpu" else False
            )
        except Exception as e:
            self.logger.error(f"Failed to create datasets: {e}")
            raise
        
        # Binary classifier (black F-150 or not)
        self.model = self._create_model()
    
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _create_datasets(self):
        """Create train and validation datasets"""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
            
        full_dataset = torchvision.datasets.ImageFolder(
            self.data_dir,
            transform=self.transform
        )
        
        if len(full_dataset) == 0:
            raise ValueError("No images found in data directory")
            
        # Split into train/val
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        self.logger.info(f"Total images: {len(full_dataset)}")
        self.logger.info(f"Training images: {train_size}")
        self.logger.info(f"Validation images: {val_size}")
        
        return torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    def _create_model(self):
        """Create binary classifier based on EfficientNet"""
        try:
            model = timm.create_model('efficientnet_b0', pretrained=True)
            model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
            return model.to(self.device)
        except Exception as e:
            self.logger.error(f"Failed to create model: {e}")
            raise
    
    def train(self, epochs: int = 10):
        """Train the black F-150 detector"""
        try:
            run_name = f"f150-detector-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb.init(project="f150-detector", name=run_name)
            
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.1, patience=3
            )
            
            best_acc = 0.0
            for epoch in range(epochs):
                # Training phase
                self.model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
                for inputs, labels in pbar:
                    try:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        
                        optimizer.zero_grad()
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        
                        train_loss += loss.item()
                        _, predicted = outputs.max(1)
                        train_total += labels.size(0)
                        train_correct += predicted.eq(labels).sum().item()
                        
                        # Update progress bar
                        pbar.set_postfix({
                            'loss': f"{train_loss/train_total:.3f}",
                            'acc': f"{100.*train_correct/train_total:.1f}%"
                        })
                    except Exception as e:
                        self.logger.error(f"Error in training batch: {e}")
                        continue
                
                train_acc = 100. * train_correct / train_total
                
                # Validation phase
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
                    for inputs, labels in pbar:
                        try:
                            inputs = inputs.to(self.device)
                            labels = labels.to(self.device)
                            
                            outputs = self.model(inputs)
                            loss = criterion(outputs, labels)
                            
                            val_loss += loss.item()
                            _, predicted = outputs.max(1)
                            val_total += labels.size(0)
                            val_correct += predicted.eq(labels).sum().item()
                            
                            pbar.set_postfix({
                                'loss': f"{val_loss/val_total:.3f}",
                                'acc': f"{100.*val_correct/val_total:.1f}%"
                            })
                        except Exception as e:
                            self.logger.error(f"Error in validation batch: {e}")
                            continue
                
                val_acc = 100. * val_correct / val_total
                
                # Log metrics
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss / len(self.train_loader),
                    'train_acc': train_acc,
                    'val_loss': val_loss / len(self.val_loader),
                    'val_acc': val_acc
                })
                
                # Save best model
                if val_acc > best_acc:
                    self.logger.info(f'Saving best model with accuracy: {val_acc:.2f}%')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': val_acc,
                    }, f'models/f150_detector_{run_name}_best.pth')
                    best_acc = val_acc
                
                # Adjust learning rate
                scheduler.step(val_acc)
                
                self.logger.info(
                    f'Epoch {epoch+1}/{epochs}:\n'
                    f'Train Loss: {train_loss/len(self.train_loader):.3f}, '
                    f'Train Acc: {train_acc:.2f}%\n'
                    f'Val Loss: {val_loss/len(self.val_loader):.3f}, '
                    f'Val Acc: {val_acc:.2f}%'
                )
                
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
        finally:
            wandb.finish()

if __name__ == "__main__":
    try:
        # Create models directory
        Path('models').mkdir(exist_ok=True)
        
        # Initialize and train detector
        detector = F150Detector('data/vehicles/organized')
        detector.train(epochs=20)
    except Exception as e:
        logging.error(f"Program failed: {e}") 
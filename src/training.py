"""
Training pipeline with transfer learning and hyperparameter tuning
"""
import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import wandb
from torch.utils.tensorboard import SummaryWriter

from src.models import ModelFactory
from src.data_preprocessing import DataPreprocessor

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score: float, model: nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        return False
    
    def save_checkpoint(self, model: nn.Module):
        """Save model checkpoint"""
        self.best_weights = model.state_dict().copy()

class MetricsTracker:
    """Track and compute training metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.predictions = []
        self.targets = []
        self.losses = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, loss: float):
        """Update metrics with batch results"""
        # Convert to numpy
        pred_probs = torch.softmax(predictions, dim=1).detach().cpu().numpy()
        pred_labels = np.argmax(pred_probs, axis=1)
        true_labels = targets.detach().cpu().numpy()
        
        self.predictions.extend(pred_labels)
        self.targets.extend(true_labels)
        self.losses.append(loss)
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute all metrics"""
        if not self.predictions:
            return {}
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        metrics = {
            'loss': np.mean(self.losses),
            'accuracy': accuracy_score(targets, predictions),
            'precision': precision_score(targets, predictions, average='weighted', zero_division=0),
            'recall': recall_score(targets, predictions, average='weighted', zero_division=0),
            'f1': f1_score(targets, predictions, average='weighted', zero_division=0)
        }
        
        # Add AUC if we have probability predictions
        try:
            if len(np.unique(targets)) == 2:  # Binary classification
                metrics['auc'] = roc_auc_score(targets, predictions)
        except:
            pass
        
        return metrics

class DeepfakeTrainer:
    """Main training class for deepfake detection models"""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 config: Dict,
                 save_dir: Path,
                 use_wandb: bool = False,
                 use_tensorboard: bool = True):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Mixed precision training
        self.use_amp = config.get('use_amp', True) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
        # Loss function
        self.criterion = self._get_loss_function()
        
        # Optimizer
        self.optimizer = self._get_optimizer()
        
        # Scheduler
        self.scheduler = self._get_scheduler()
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get('patience', 10),
            min_delta=config.get('min_delta', 0.001)
        )
        
        # Metrics tracking
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()
        
        # History
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': [],
            'lr': []
        }
        
        # Logging
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        
        if self.use_tensorboard:
            self.writer = SummaryWriter(log_dir=self.save_dir / 'tensorboard')
        
        if self.use_wandb:
            wandb.init(
                project="deepfake-detection",
                config=config,
                name=f"{model.__class__.__name__}_{int(time.time())}"
            )
            wandb.watch(self.model)
    
    def _get_loss_function(self) -> nn.Module:
        """Get loss function based on configuration"""
        loss_type = self.config.get('loss_function', 'cross_entropy')
        
        if loss_type == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif loss_type == 'focal_loss':
            return self._focal_loss
        elif loss_type == 'label_smoothing':
            return nn.CrossEntropyLoss(label_smoothing=self.config.get('label_smoothing', 0.1))
        else:
            return nn.CrossEntropyLoss()
    
    def _focal_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Focal loss implementation"""
        alpha = self.config.get('focal_alpha', 1.0)
        gamma = self.config.get('focal_gamma', 2.0)
        
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        
        return focal_loss.mean()
    
    def _get_optimizer(self) -> optim.Optimizer:
        """Get optimizer based on configuration"""
        optimizer_type = self.config.get('optimizer', 'adamw')
        lr = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        if optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            momentum = self.config.get('momentum', 0.9)
            return optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
    
    def _get_scheduler(self) -> Optional[Any]:
        """Get learning rate scheduler"""
        scheduler_type = self.config.get('scheduler', 'reduce_on_plateau')
        
        if scheduler_type == 'reduce_on_plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=self.config.get('lr_factor', 0.5),
                patience=self.config.get('lr_patience', 5)
            )
        elif scheduler_type == 'cosine_annealing':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('epochs', 50),
                eta_min=self.config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'step_lr':
            return StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 10),
                gamma=self.config.get('gamma', 0.1)
            )
        else:
            return None
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        self.train_metrics.reset()
        
        pbar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(data)
                    loss = self.criterion(outputs, targets)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            self.train_metrics.update(outputs, targets, loss.item())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return self.train_metrics.compute_metrics()
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        self.val_metrics.reset()
        
        with torch.no_grad():
            for data, targets in tqdm(self.val_loader, desc="Validation"):
                data, targets = data.to(self.device), targets.to(self.device)
                
                if self.use_amp:
                    with autocast():
                        outputs = self.model(data)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(data)
                    loss = self.criterion(outputs, targets)
                
                self.val_metrics.update(outputs, targets, loss.item())
        
        return self.val_metrics.compute_metrics()
    
    def train(self, epochs: int) -> Dict[str, List[float]]:
        """Main training loop"""
        print(f"Starting training for {epochs} epochs on {self.device}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate_epoch()
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['train_f1'].append(train_metrics['f1'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Scheduler step
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['accuracy'])
                else:
                    self.scheduler.step()
            
            # Logging
            epoch_time = time.time() - start_time
            
            print(f"\nEpoch {epoch+1}/{epochs} ({epoch_time:.2f}s)")
            print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # TensorBoard logging
            if self.use_tensorboard:
                self.writer.add_scalars('Loss', {
                    'train': train_metrics['loss'],
                    'val': val_metrics['loss']
                }, epoch)
                self.writer.add_scalars('Accuracy', {
                    'train': train_metrics['accuracy'],
                    'val': val_metrics['accuracy']
                }, epoch)
                self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Weights & Biases logging
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    'train_acc': train_metrics['accuracy'],
                    'val_acc': val_metrics['accuracy'],
                    'train_f1': train_metrics['f1'],
                    'val_f1': val_metrics['f1'],
                    'lr': self.optimizer.param_groups[0]['lr']
                })
            
            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                self.save_model('best_model.pth', epoch, val_metrics)
            
            # Early stopping
            if self.early_stopping(val_metrics['accuracy'], self.model):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        # Save final model
        self.save_model('final_model.pth', epochs, val_metrics)
        
        # Save training history
        self.save_history()
        
        # Plot training curves
        self.plot_training_curves()
        
        return self.history
    
    def save_model(self, filename: str, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'history': self.history
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, self.save_dir / filename)
        print(f"Model saved: {self.save_dir / filename}")
    
    def load_model(self, filename: str) -> Dict:
        """Load model checkpoint"""
        checkpoint = torch.load(self.save_dir / filename, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Model loaded: {self.save_dir / filename}")
        return checkpoint
    
    def save_history(self):
        """Save training history"""
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(self.save_dir / 'training_history.csv', index=False)
        
        with open(self.save_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def plot_training_curves(self):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(self.history['train_acc'], label='Train Acc')
        axes[0, 1].plot(self.history['val_acc'], label='Val Acc')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 Score
        axes[1, 0].plot(self.history['train_f1'], label='Train F1')
        axes[1, 0].plot(self.history['val_f1'], label='Val F1')
        axes[1, 0].set_title('F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning Rate
        axes[1, 1].plot(self.history['lr'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on given data loader"""
        self.model.eval()
        metrics_tracker = MetricsTracker()
        
        with torch.no_grad():
            for data, targets in tqdm(data_loader, desc="Evaluating"):
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                metrics_tracker.update(outputs, targets, loss.item())
        
        return metrics_tracker.compute_metrics()

def main():
    """Main training function"""
    from config import (DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, 
                       IMAGE_SIZE, AUGMENTATION_CONFIG, BATCH_SIZE, NUM_WORKERS,
                       EPOCHS, LEARNING_RATE, WEIGHT_DECAY, MODEL_CONFIGS)
    
    # Training configuration
    training_config = {
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'batch_size': BATCH_SIZE,
        'optimizer': 'adamw',
        'scheduler': 'reduce_on_plateau',
        'loss_function': 'cross_entropy',
        'use_amp': True,
        'patience': 10,
        'min_delta': 0.001
    }
    
    # Initialize data preprocessor
    preprocessor = DataPreprocessor(
        data_dir=DATA_DIR,
        processed_dir=PROCESSED_DATA_DIR,
        image_size=IMAGE_SIZE,
        augmentation_config=AUGMENTATION_CONFIG
    )
    
    # Load data splits
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    
    # Create data loaders
    train_loader, val_loader, test_loader = preprocessor.create_data_loaders(
        train_df, val_df, test_df, BATCH_SIZE, NUM_WORKERS
    )
    
    # Train different models
    models_to_train = ['efficientnet']
    
    for model_type in models_to_train:
        print(f"\n{'='*50}")
        print(f"Training {model_type.upper()}")
        print(f"{'='*50}")
        
        # Create model
        model_config = MODEL_CONFIGS.get(model_type, {})
        model = ModelFactory.create_model(model_type, model_config)
        
        # Create trainer
        save_dir = MODELS_DIR / model_type
        trainer = DeepfakeTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=training_config,
            save_dir=save_dir,
            use_wandb=False,  # Set to True if you want to use Weights & Biases
            use_tensorboard=True
        )
        
        # Train model
        history = trainer.train(EPOCHS)
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_metrics = trainer.evaluate(test_loader)
        print("Test Metrics:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Save test metrics
        with open(save_dir / 'test_metrics.json', 'w') as f:
            json.dump(test_metrics, f, indent=2)

if __name__ == "__main__":
    main()

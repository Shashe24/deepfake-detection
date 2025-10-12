"""
Data preprocessing pipeline with augmentation and splitting
"""
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class DeepfakeDataset(Dataset):
    """Custom dataset class for deepfake detection"""
    
    def __init__(self, 
                 image_paths: List[str], 
                 labels: List[int], 
                 transform=None, 
                 image_size: Tuple[int, int] = (224, 224)):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.image_size = image_size
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, self.image_size)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        # Convert to tensor if not already
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return image, label

class DataAugmentation:
    """Data augmentation pipeline using Albumentations"""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def get_train_transforms(self, image_size: Tuple[int, int] = (224, 224)):
        """Get training augmentation pipeline"""
        return A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=self.config['rotation_limit'], p=self.config['p']),
            A.RandomBrightnessContrast(
                brightness_limit=self.config['brightness_limit'],
                contrast_limit=self.config['contrast_limit'],
                p=self.config['p']
            ),
            A.HueSaturationValue(
                hue_shift_limit=self.config['hue_shift_limit'],
                sat_shift_limit=self.config['saturation_limit'],
                val_shift_limit=self.config['brightness_limit'],
                p=self.config['p']
            ),
            A.GaussNoise(
                var_limit=self.config['noise_var_limit'],
                p=self.config['p']
            ),
            A.GaussianBlur(
                blur_limit=self.config['blur_limit'],
                p=0.3
            ),
            A.CLAHE(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5
            ),
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                p=0.3
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def get_val_transforms(self, image_size: Tuple[int, int] = (224, 224)):
        """Get validation/test transforms (no augmentation)"""
        return A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

class DataPreprocessor:
    """Main data preprocessing class"""
    
    def __init__(self, 
                 data_dir: Path, 
                 processed_dir: Path,
                 image_size: Tuple[int, int] = (224, 224),
                 augmentation_config: Dict = None):
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(exist_ok=True)
        self.image_size = image_size
        
        # Initialize augmentation
        if augmentation_config:
            self.augmentation = DataAugmentation(augmentation_config)
        else:
            self.augmentation = None
    
    def load_metadata(self) -> pd.DataFrame:
        """Load or create metadata file"""
        metadata_path = self.data_dir / "metadata.csv"
        
        if metadata_path.exists():
            return pd.read_csv(metadata_path)
        else:
            # For Kaggle dataset, check if we have the metadata
            print("No metadata.csv found. Please run create_kaggle_metadata.py first.")
            raise FileNotFoundError("metadata.csv not found. Run create_kaggle_metadata.py to create it.")
    
    def split_data(self, 
                   metadata: pd.DataFrame,
                   train_ratio: float = 0.8,
                   val_ratio: float = 0.1,
                   test_ratio: float = 0.1,
                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets"""
        
        # Check if dataset already has predefined splits (Kaggle dataset)
        if 'split' in metadata.columns:
            print("Using predefined dataset splits...")
            
            # Get train and test data
            train_data = metadata[metadata['split'] == 'train'].copy()
            test_data = metadata[metadata['split'] == 'test'].copy()
            
            # Split train data into train and validation
            X_train_full = train_data['image_path'].values
            y_train_full = train_data['label'].values
            
            # Create validation split from training data
            val_size = val_ratio / (train_ratio + val_ratio)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full, y_train_full, test_size=val_size, stratify=y_train_full, random_state=random_state
            )
            
            # Create DataFrames
            train_df = pd.DataFrame({'image_path': X_train, 'label': y_train})
            val_df = pd.DataFrame({'image_path': X_val, 'label': y_val})
            test_df = test_data[['image_path', 'label']].copy()
            
            # Add class names
            train_df['class_name'] = train_df['label'].map({0: 'real', 1: 'fake'})
            val_df['class_name'] = val_df['label'].map({0: 'real', 1: 'fake'})
            test_df['class_name'] = test_df['label'].map({0: 'real', 1: 'fake'})
            
        else:
            # Original splitting logic for datasets without predefined splits
            assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
            
            # Stratified split to maintain class balance
            X = metadata['image_path'].values
            y = metadata['label'].values
            
            # First split: train + val, test
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_ratio, stratify=y, random_state=random_state
            )
            
            # Second split: train, val
            val_size = val_ratio / (train_ratio + val_ratio)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size, stratify=y_temp, random_state=random_state
            )
            
            # Create DataFrames
            train_df = pd.DataFrame({'image_path': X_train, 'label': y_train})
            val_df = pd.DataFrame({'image_path': X_val, 'label': y_val})
            test_df = pd.DataFrame({'image_path': X_test, 'label': y_test})
            
            # Add class names
            train_df['class_name'] = train_df['label'].map({0: 'real', 1: 'fake'})
            val_df['class_name'] = val_df['label'].map({0: 'real', 1: 'fake'})
            test_df['class_name'] = test_df['label'].map({0: 'real', 1: 'fake'})
        
        # Save splits
        train_df.to_csv(self.processed_dir / "train.csv", index=False)
        val_df.to_csv(self.processed_dir / "val.csv", index=False)
        test_df.to_csv(self.processed_dir / "test.csv", index=False)
        
        print(f"Data split completed:")
        print(f"  Train: {len(train_df)} samples")
        print(f"  Validation: {len(val_df)} samples")
        print(f"  Test: {len(test_df)} samples")
        
        return train_df, val_df, test_df
    
    def create_data_loaders(self,
                           train_df: pd.DataFrame,
                           val_df: pd.DataFrame,
                           test_df: pd.DataFrame,
                           batch_size: int = 32,
                           num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create PyTorch data loaders"""
        
        # Get transforms
        if self.augmentation:
            train_transform = self.augmentation.get_train_transforms(self.image_size)
            val_transform = self.augmentation.get_val_transforms(self.image_size)
        else:
            train_transform = val_transform = None
        
        # Create datasets
        train_dataset = DeepfakeDataset(
            train_df['image_path'].tolist(),
            train_df['label'].tolist(),
            transform=train_transform,
            image_size=self.image_size
        )
        
        val_dataset = DeepfakeDataset(
            val_df['image_path'].tolist(),
            val_df['label'].tolist(),
            transform=val_transform,
            image_size=self.image_size
        )
        
        test_dataset = DeepfakeDataset(
            test_df['image_path'].tolist(),
            test_df['label'].tolist(),
            transform=val_transform,
            image_size=self.image_size
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def analyze_dataset(self, metadata: pd.DataFrame) -> Dict:
        """Analyze dataset statistics"""
        analysis = {
            'total_samples': len(metadata),
            'class_distribution': metadata['class_name'].value_counts().to_dict(),
            'class_balance': metadata['label'].value_counts(normalize=True).to_dict()
        }
        
        # Analyze image properties
        sample_images = metadata.sample(min(100, len(metadata)))
        widths, heights, channels = [], [], []
        
        print("Analyzing sample images...")
        for _, row in tqdm(sample_images.iterrows(), total=len(sample_images)):
            try:
                img = cv2.imread(row['image_path'])
                if img is not None:
                    h, w, c = img.shape
                    heights.append(h)
                    widths.append(w)
                    channels.append(c)
            except Exception as e:
                continue
        
        if heights:
            analysis['image_stats'] = {
                'avg_width': np.mean(widths),
                'avg_height': np.mean(heights),
                'min_width': np.min(widths),
                'max_width': np.max(widths),
                'min_height': np.min(heights),
                'max_height': np.max(heights),
                'channels': np.mean(channels)
            }
        
        return analysis
    
    def visualize_samples(self, 
                         train_loader: DataLoader, 
                         num_samples: int = 8,
                         save_path: Optional[str] = None):
        """Visualize sample images from the dataset"""
        
        # Get a batch of data
        data_iter = iter(train_loader)
        images, labels = next(data_iter)
        
        # Create subplot
        fig, axes = plt.subplots(2, 4, figsize=(15, 8))
        axes = axes.ravel()
        
        for i in range(min(num_samples, len(images))):
            # Denormalize image
            img = images[i].numpy().transpose(1, 2, 0)
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            
            axes[i].imshow(img)
            axes[i].set_title(f"Label: {'Fake' if labels[i] == 1 else 'Real'}")
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Sample visualization saved to {save_path}")
        
        plt.show()
    
    def create_class_distribution_plot(self, 
                                     metadata: pd.DataFrame,
                                     save_path: Optional[str] = None):
        """Create class distribution visualization"""
        
        plt.figure(figsize=(10, 6))
        
        # Count plot
        plt.subplot(1, 2, 1)
        sns.countplot(data=metadata, x='class_name')
        plt.title('Class Distribution (Counts)')
        plt.ylabel('Number of Samples')
        
        # Pie chart
        plt.subplot(1, 2, 2)
        class_counts = metadata['class_name'].value_counts()
        plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%')
        plt.title('Class Distribution (Percentage)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Class distribution plot saved to {save_path}")
        
        plt.show()

def main():
    """Main preprocessing pipeline"""
    from config import (DATA_DIR, PROCESSED_DATA_DIR, IMAGE_SIZE, 
                       AUGMENTATION_CONFIG, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
                       BATCH_SIZE, NUM_WORKERS)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        data_dir=DATA_DIR,
        processed_dir=PROCESSED_DATA_DIR,
        image_size=IMAGE_SIZE,
        augmentation_config=AUGMENTATION_CONFIG
    )
    
    # Load metadata
    print("Loading metadata...")
    metadata = preprocessor.load_metadata()
    
    # Analyze dataset
    print("\nAnalyzing dataset...")
    analysis = preprocessor.analyze_dataset(metadata)
    print("Dataset Analysis:")
    for key, value in analysis.items():
        print(f"  {key}: {value}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    preprocessor.create_class_distribution_plot(
        metadata, 
        save_path=PROCESSED_DATA_DIR / "class_distribution.png"
    )
    
    # Split data
    print("\nSplitting data...")
    train_df, val_df, test_df = preprocessor.split_data(
        metadata, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
    )
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = preprocessor.create_data_loaders(
        train_df, val_df, test_df, BATCH_SIZE, NUM_WORKERS
    )
    
    # Visualize samples
    print("\nVisualizing sample images...")
    preprocessor.visualize_samples(
        train_loader,
        save_path=PROCESSED_DATA_DIR / "sample_images.png"
    )
    
    print("\nData preprocessing completed successfully!")
    print(f"Processed data saved to: {PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    main()

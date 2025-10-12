"""
Data collection and dataset management utilities
"""
import os
import shutil
import requests
import zipfile
from pathlib import Path
from typing import List, Tuple
import pandas as pd
from tqdm import tqdm
import cv2
import numpy as np

class DatasetManager:
    """Manages dataset collection, organization, and validation"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.real_dir = self.data_dir / "real"
        self.fake_dir = self.data_dir / "fake"
        
    def validate_dataset_structure(self) -> bool:
        """Validate that dataset has proper structure"""
        if not self.data_dir.exists():
            print(f"Dataset directory {self.data_dir} does not exist")
            return False
            
        if not self.real_dir.exists() or not self.fake_dir.exists():
            print("Missing 'real' or 'fake' subdirectories")
            return False
            
        real_count = len(list(self.real_dir.glob("*.jpg"))) + len(list(self.real_dir.glob("*.png")))
        fake_count = len(list(self.fake_dir.glob("*.jpg"))) + len(list(self.fake_dir.glob("*.png")))
        
        print(f"Found {real_count} real images and {fake_count} fake images")
        
        if real_count == 0 or fake_count == 0:
            print("Dataset appears to be empty or improperly structured")
            return False
            
        return True
    
    def get_dataset_info(self) -> dict:
        """Get comprehensive dataset information"""
        info = {
            'real_images': 0,
            'fake_images': 0,
            'total_images': 0,
            'image_formats': set(),
            'corrupted_images': []
        }
        
        # Count real images
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            info['real_images'] += len(list(self.real_dir.glob(ext)))
            
        # Count fake images
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            info['fake_images'] += len(list(self.fake_dir.glob(ext)))
            
        info['total_images'] = info['real_images'] + info['fake_images']
        
        # Check for corrupted images
        info['corrupted_images'] = self._find_corrupted_images()
        
        return info
    
    def _find_corrupted_images(self) -> List[str]:
        """Find corrupted or unreadable images"""
        corrupted = []
        
        for img_dir, label in [(self.real_dir, 'real'), (self.fake_dir, 'fake')]:
            for img_path in img_dir.rglob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    try:
                        img = cv2.imread(str(img_path))
                        if img is None:
                            corrupted.append(str(img_path))
                    except Exception as e:
                        corrupted.append(str(img_path))
                        
        return corrupted
    
    def clean_dataset(self) -> None:
        """Remove corrupted images and standardize format"""
        corrupted = self._find_corrupted_images()
        
        print(f"Found {len(corrupted)} corrupted images")
        
        for img_path in corrupted:
            try:
                os.remove(img_path)
                print(f"Removed corrupted image: {img_path}")
            except Exception as e:
                print(f"Failed to remove {img_path}: {e}")
    
    def balance_dataset(self, target_size: int = None) -> None:
        """Balance the dataset by sampling equal numbers from each class"""
        real_images = list(self.real_dir.glob("*.jpg")) + list(self.real_dir.glob("*.png"))
        fake_images = list(self.fake_dir.glob("*.jpg")) + list(self.fake_dir.glob("*.png"))
        
        if target_size is None:
            target_size = min(len(real_images), len(fake_images))
        
        print(f"Balancing dataset to {target_size} images per class")
        
        # Randomly sample images
        np.random.shuffle(real_images)
        np.random.shuffle(fake_images)
        
        # Keep only target_size images
        for i, img_path in enumerate(real_images):
            if i >= target_size:
                os.remove(img_path)
                
        for i, img_path in enumerate(fake_images):
            if i >= target_size:
                os.remove(img_path)
    
    def create_metadata_file(self) -> pd.DataFrame:
        """Create metadata CSV file with image paths and labels"""
        metadata = []
        
        # Process real images
        for img_path in self.real_dir.rglob("*"):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                metadata.append({
                    'image_path': str(img_path),
                    'label': 0,  # 0 for real
                    'class_name': 'real'
                })
        
        # Process fake images
        for img_path in self.fake_dir.rglob("*"):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                metadata.append({
                    'image_path': str(img_path),
                    'label': 1,  # 1 for fake
                    'class_name': 'fake'
                })
        
        df = pd.DataFrame(metadata)
        metadata_path = self.data_dir / "metadata.csv"
        df.to_csv(metadata_path, index=False)
        
        print(f"Created metadata file: {metadata_path}")
        print(f"Total samples: {len(df)}")
        print(f"Class distribution:\n{df['class_name'].value_counts()}")
        
        return df

class DeepDetectDownloader:
    """Download and setup DeepDetect-2025 or other datasets"""
    
    def __init__(self, download_dir: Path):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
    
    def download_file(self, url: str, filename: str) -> bool:
        """Download file from URL with progress bar"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(self.download_dir / filename, 'wb') as file:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file.write(chunk)
                            pbar.update(len(chunk))
            
            return True
            
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
            return False
    
    def extract_dataset(self, archive_path: str, extract_to: str) -> bool:
        """Extract downloaded dataset archive"""
        try:
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            print(f"Extracted dataset to {extract_to}")
            return True
        except Exception as e:
            print(f"Failed to extract dataset: {e}")
            return False

if __name__ == "__main__":
    # Example usage
    from config import DATA_DIR
    
    # Initialize dataset manager
    dataset_manager = DatasetManager(DATA_DIR)
    
    # Validate and get info about existing dataset
    if dataset_manager.validate_dataset_structure():
        info = dataset_manager.get_dataset_info()
        print("Dataset Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Create metadata file
        metadata_df = dataset_manager.create_metadata_file()
        
        # Clean dataset if needed
        if info['corrupted_images']:
            print(f"\nCleaning {len(info['corrupted_images'])} corrupted images...")
            dataset_manager.clean_dataset()
    else:
        print("Dataset validation failed. Please check your dataset structure.")

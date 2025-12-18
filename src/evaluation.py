"""
Comprehensive evaluation metrics and robustness testing for deepfake detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix,
    classification_report, average_precision_score
)
from sklearn.calibration import calibration_curve
import cv2
from tqdm import tqdm
import albumentations as A
from torch.utils.data import DataLoader
import json

class ModelEvaluator:
    """Comprehensive model evaluation with various metrics and robustness tests"""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()
        
    def predict_proba(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Get probability predictions and true labels"""
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in tqdm(data_loader, desc="Predicting"):
                data = data.to(self.device)
                
                outputs = self.model(data)
                probs = F.softmax(outputs, dim=1)
                
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.numpy())
        
        return np.vstack(all_probs), np.concatenate(all_labels)
    
    def compute_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        """Compute basic classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
        }
        
        # Binary classification metrics
        if len(np.unique(y_true)) == 2:
            metrics.update({
                'roc_auc': roc_auc_score(y_true, y_proba[:, 1]),
                'average_precision': average_precision_score(y_true, y_proba[:, 1]),
                'precision_binary': precision_score(y_true, y_pred),
                'recall_binary': recall_score(y_true, y_pred),
                'f1_binary': f1_score(y_true, y_pred),
                'specificity': self._compute_specificity(y_true, y_pred),
                'sensitivity': recall_score(y_true, y_pred)
            })
        
        return metrics
    
    def _compute_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute specificity (true negative rate)"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    def evaluate_comprehensive(self, data_loader: DataLoader, save_dir: Path = None) -> Dict[str, Any]:
        """Comprehensive evaluation with all metrics and visualizations"""
        
        # Get predictions
        y_proba, y_true = self.predict_proba(data_loader)
        y_pred = np.argmax(y_proba, axis=1)
        
        # Basic metrics
        metrics = self.compute_basic_metrics(y_true, y_pred, y_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        class_report = classification_report(y_true, y_pred, output_dict=True)
        metrics['classification_report'] = class_report
        
        # Per-class metrics
        metrics['per_class_metrics'] = self._compute_per_class_metrics(y_true, y_pred, y_proba)
        
        # Confidence analysis
        metrics['confidence_analysis'] = self._analyze_confidence(y_proba, y_true, y_pred)
        
        # Create visualizations
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)
            
            self.plot_confusion_matrix(cm, save_dir / 'confusion_matrix.png')
            self.plot_roc_curve(y_true, y_proba, save_dir / 'roc_curve.png')
            self.plot_precision_recall_curve(y_true, y_proba, save_dir / 'pr_curve.png')
            self.plot_confidence_histogram(y_proba, y_true, save_dir / 'confidence_histogram.png')
            self.plot_calibration_curve(y_true, y_proba, save_dir / 'calibration_curve.png')
            
            # Save metrics to JSON
            with open(save_dir / 'evaluation_metrics.json', 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                metrics_json = self._convert_for_json(metrics)
                json.dump(metrics_json, f, indent=2)
        
        return metrics
    
    def _convert_for_json(self, obj):
        """Convert numpy arrays to lists for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj
    
    def _compute_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Compute per-class metrics"""
        classes = np.unique(y_true)
        per_class = {}
        
        for cls in classes:
            class_name = 'real' if cls == 0 else 'fake'
            
            # Binary classification for this class vs all others
            y_true_binary = (y_true == cls).astype(int)
            y_pred_binary = (y_pred == cls).astype(int)
            y_proba_binary = y_proba[:, cls]
            
            per_class[class_name] = {
                'precision': precision_score(y_true_binary, y_pred_binary),
                'recall': recall_score(y_true_binary, y_pred_binary),
                'f1_score': f1_score(y_true_binary, y_pred_binary),
                'support': np.sum(y_true == cls),
                'auc': roc_auc_score(y_true_binary, y_proba_binary) if len(np.unique(y_true_binary)) == 2 else 0.0
            }
        
        return per_class
    
    def _analyze_confidence(self, y_proba: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Analyze prediction confidence"""
        max_probs = np.max(y_proba, axis=1)
        correct_mask = (y_pred == y_true)
        
        analysis = {
            'mean_confidence': float(np.mean(max_probs)),
            'mean_confidence_correct': float(np.mean(max_probs[correct_mask])),
            'mean_confidence_incorrect': float(np.mean(max_probs[~correct_mask])),
            'confidence_std': float(np.std(max_probs)),
            'low_confidence_threshold': 0.6,
            'high_confidence_threshold': 0.9
        }
        
        # Accuracy at different confidence thresholds
        for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
            high_conf_mask = max_probs >= threshold
            if np.sum(high_conf_mask) > 0:
                acc_at_threshold = accuracy_score(y_true[high_conf_mask], y_pred[high_conf_mask])
                coverage = np.mean(high_conf_mask)
                analysis[f'accuracy_at_conf_{threshold}'] = float(acc_at_threshold)
                analysis[f'coverage_at_conf_{threshold}'] = float(coverage)
        
        return analysis
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: Path):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        
        plt.title('Confusion Matrix (Normalized)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Add raw counts as text
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j + 0.5, i + 0.7, f'({cm[i, j]})', 
                        ha='center', va='center', fontsize=10, color='red')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray, save_path: Path):
        """Plot ROC curve"""
        if len(np.unique(y_true)) != 2:
            return
        
        fpr, tpr, thresholds = roc_curve(y_true, y_proba[:, 1])
        auc = roc_auc_score(y_true, y_proba[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_proba: np.ndarray, save_path: Path):
        """Plot Precision-Recall curve"""
        if len(np.unique(y_true)) != 2:
            return
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba[:, 1])
        avg_precision = average_precision_score(y_true, y_proba[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'PR curve (AP = {avg_precision:.3f})')
        
        # Baseline (random classifier)
        baseline = np.mean(y_true)
        plt.axhline(y=baseline, color='red', linestyle='--', 
                   label=f'Baseline (AP = {baseline:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confidence_histogram(self, y_proba: np.ndarray, y_true: np.ndarray, save_path: Path):
        """Plot confidence histogram"""
        max_probs = np.max(y_proba, axis=1)
        y_pred = np.argmax(y_proba, axis=1)
        
        correct_mask = (y_pred == y_true)
        
        plt.figure(figsize=(10, 6))
        
        # Histogram for correct predictions
        plt.hist(max_probs[correct_mask], bins=20, alpha=0.7, label='Correct', 
                color='green', density=True)
        
        # Histogram for incorrect predictions
        plt.hist(max_probs[~correct_mask], bins=20, alpha=0.7, label='Incorrect', 
                color='red', density=True)
        
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Density')
        plt.title('Prediction Confidence Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add vertical lines for thresholds
        for threshold in [0.6, 0.8, 0.9]:
            plt.axvline(x=threshold, color='black', linestyle='--', alpha=0.5)
            plt.text(threshold, plt.ylim()[1] * 0.9, f'{threshold}', 
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_calibration_curve(self, y_true: np.ndarray, y_proba: np.ndarray, save_path: Path):
        """Plot calibration curve"""
        if len(np.unique(y_true)) != 2:
            return
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_proba[:, 1], n_bins=10
        )
        
        plt.figure(figsize=(8, 6))
        
        # Calibration curve
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", 
                label="Model", color='blue')
        
        # Perfect calibration line
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

class RobustnessEvaluator:
    """Evaluate model robustness against various attacks and perturbations"""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def evaluate_noise_robustness(self, data_loader: DataLoader, 
                                noise_levels: List[float] = [0.01, 0.05, 0.1, 0.2]) -> Dict[str, float]:
        """Evaluate robustness to Gaussian noise"""
        results = {}
        
        for noise_level in noise_levels:
            print(f"Evaluating noise robustness at level {noise_level}")
            
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, labels in tqdm(data_loader, desc=f"Noise {noise_level}"):
                    data, labels = data.to(self.device), labels.to(self.device)
                    
                    # Add Gaussian noise
                    noise = torch.randn_like(data) * noise_level
                    noisy_data = data + noise
                    noisy_data = torch.clamp(noisy_data, 0, 1)
                    
                    outputs = self.model(noisy_data)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            accuracy = correct / total
            results[f'noise_{noise_level}'] = accuracy
            print(f"Accuracy with noise {noise_level}: {accuracy:.4f}")
        
        return results
    
    def evaluate_brightness_robustness(self, data_loader: DataLoader,
                                     brightness_factors: List[float] = [0.5, 0.7, 1.3, 1.5]) -> Dict[str, float]:
        """Evaluate robustness to brightness changes"""
        results = {}
        
        for factor in brightness_factors:
            print(f"Evaluating brightness robustness at factor {factor}")
            
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, labels in tqdm(data_loader, desc=f"Brightness {factor}"):
                    data, labels = data.to(self.device), labels.to(self.device)
                    
                    # Adjust brightness
                    bright_data = data * factor
                    bright_data = torch.clamp(bright_data, 0, 1)
                    
                    outputs = self.model(bright_data)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            accuracy = correct / total
            results[f'brightness_{factor}'] = accuracy
            print(f"Accuracy with brightness {factor}: {accuracy:.4f}")
        
        return results
    
    def evaluate_compression_robustness(self, data_loader: DataLoader,
                                      quality_levels: List[int] = [95, 85, 75, 50, 25]) -> Dict[str, float]:
        """Evaluate robustness to JPEG compression"""
        results = {}
        
        for quality in quality_levels:
            print(f"Evaluating compression robustness at quality {quality}")
            
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, labels in tqdm(data_loader, desc=f"Quality {quality}"):
                    labels = labels.to(self.device)
                    
                    # Apply JPEG compression
                    compressed_data = []
                    for img in data:
                        # Convert to numpy and apply compression
                        img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        
                        # Encode and decode JPEG
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                        _, encoded_img = cv2.imencode('.jpg', img_np, encode_param)
                        decoded_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
                        
                        # Convert back to tensor
                        decoded_img = torch.from_numpy(decoded_img.transpose(2, 0, 1)).float() / 255.0
                        compressed_data.append(decoded_img)
                    
                    compressed_data = torch.stack(compressed_data).to(self.device)
                    
                    outputs = self.model(compressed_data)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            accuracy = correct / total
            results[f'compression_{quality}'] = accuracy
            print(f"Accuracy with compression quality {quality}: {accuracy:.4f}")
        
        return results
    
    def evaluate_adversarial_robustness(self, data_loader: DataLoader,
                                      epsilon_values: List[float] = [0.01, 0.03, 0.05]) -> Dict[str, float]:
        """Evaluate robustness to FGSM adversarial attacks"""
        results = {}
        
        for epsilon in epsilon_values:
            print(f"Evaluating adversarial robustness with epsilon {epsilon}")
            
            correct = 0
            total = 0
            
            for data, labels in tqdm(data_loader, desc=f"FGSM ε={epsilon}"):
                data, labels = data.to(self.device), labels.to(self.device)
                data.requires_grad = True
                
                # Forward pass
                outputs = self.model(data)
                loss = F.cross_entropy(outputs, labels)
                
                # Backward pass
                self.model.zero_grad()
                loss.backward()
                
                # Generate adversarial examples using FGSM
                data_grad = data.grad.data
                perturbed_data = data + epsilon * data_grad.sign()
                perturbed_data = torch.clamp(perturbed_data, 0, 1)
                
                # Evaluate on perturbed data
                with torch.no_grad():
                    outputs = self.model(perturbed_data)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            accuracy = correct / total
            results[f'fgsm_{epsilon}'] = accuracy
            print(f"Accuracy against FGSM with ε={epsilon}: {accuracy:.4f}")
        
        return results
    
    def comprehensive_robustness_evaluation(self, data_loader: DataLoader, 
                                          save_dir: Path = None) -> Dict[str, Any]:
        """Run comprehensive robustness evaluation"""
        
        print("Starting comprehensive robustness evaluation...")
        
        results = {}
        
        # Noise robustness
        results['noise_robustness'] = self.evaluate_noise_robustness(data_loader)
        
        # Brightness robustness
        results['brightness_robustness'] = self.evaluate_brightness_robustness(data_loader)
        
        # Compression robustness
        results['compression_robustness'] = self.evaluate_compression_robustness(data_loader)
        
        # Adversarial robustness
        results['adversarial_robustness'] = self.evaluate_adversarial_robustness(data_loader)
        
        # Save results
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)
            
            with open(save_dir / 'robustness_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            # Create robustness plots
            self.plot_robustness_results(results, save_dir)
        
        return results
    
    def plot_robustness_results(self, results: Dict[str, Any], save_dir: Path):
        """Plot robustness evaluation results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Noise robustness
        noise_data = results['noise_robustness']
        noise_levels = [float(k.split('_')[1]) for k in noise_data.keys()]
        noise_accuracies = list(noise_data.values())
        
        axes[0, 0].plot(noise_levels, noise_accuracies, 'o-', color='blue')
        axes[0, 0].set_xlabel('Noise Level')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Noise Robustness')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Brightness robustness
        brightness_data = results['brightness_robustness']
        brightness_factors = [float(k.split('_')[1]) for k in brightness_data.keys()]
        brightness_accuracies = list(brightness_data.values())
        
        axes[0, 1].plot(brightness_factors, brightness_accuracies, 'o-', color='orange')
        axes[0, 1].set_xlabel('Brightness Factor')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Brightness Robustness')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Compression robustness
        compression_data = results['compression_robustness']
        quality_levels = [int(k.split('_')[1]) for k in compression_data.keys()]
        compression_accuracies = list(compression_data.values())
        
        axes[1, 0].plot(quality_levels, compression_accuracies, 'o-', color='green')
        axes[1, 0].set_xlabel('JPEG Quality')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Compression Robustness')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Adversarial robustness
        adversarial_data = results['adversarial_robustness']
        epsilon_values = [float(k.split('_')[1]) for k in adversarial_data.keys()]
        adversarial_accuracies = list(adversarial_data.values())
        
        axes[1, 1].plot(epsilon_values, adversarial_accuracies, 'o-', color='red')
        axes[1, 1].set_xlabel('Epsilon (FGSM)')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Adversarial Robustness')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'robustness_plots.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main evaluation function"""
    from config import MODELS_DIR, PROCESSED_DATA_DIR, RESULTS_DIR
    from src.models import ModelFactory
    from src.data_preprocessing import DataPreprocessor
    
    # Load test data
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    
    # Create data loader (without augmentation)
    preprocessor = DataPreprocessor(
        data_dir=Path("dataset"),
        processed_dir=PROCESSED_DATA_DIR
    )
    
    # Create test loader
    from torch.utils.data import DataLoader
    from src.data_preprocessing import DeepfakeDataset
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    test_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    test_dataset = DeepfakeDataset(
        test_df['image_path'].tolist(),
        test_df['label'].tolist(),
        transform=test_transform
    )
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Evaluate different models
    model_types = ['efficientnet', 'xception', 'hybrid_ensemble']
    
    for model_type in model_types:
        print(f"\n{'='*50}")
        print(f"Evaluating {model_type.upper()}")
        print(f"{'='*50}")
        
        model_dir = MODELS_DIR / model_type
        if not (model_dir / 'best_model.pth').exists():
            print(f"Model not found: {model_dir / 'best_model.pth'}")
            continue
        
        # Load model
        checkpoint = torch.load(model_dir / 'best_model.pth', map_location=device)
        
        # Create model
        model_config = checkpoint.get('config', {})
        model = ModelFactory.create_model(model_type, model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        # Create evaluators
        evaluator = ModelEvaluator(model, device)
        robustness_evaluator = RobustnessEvaluator(model, device)
        
        # Results directory
        results_dir = RESULTS_DIR / model_type
        results_dir.mkdir(exist_ok=True)
        
        # Comprehensive evaluation
        print("Running comprehensive evaluation...")
        metrics = evaluator.evaluate_comprehensive(test_loader, results_dir)
        
        print(f"\nEvaluation Results for {model_type}:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        if 'roc_auc' in metrics:
            print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        
        # Robustness evaluation
        print("\nRunning robustness evaluation...")
        robustness_results = robustness_evaluator.comprehensive_robustness_evaluation(
            test_loader, results_dir
        )
        
        print(f"\nRobustness Results for {model_type}:")
        for category, results in robustness_results.items():
            print(f"{category}:")
            for test, accuracy in results.items():
                print(f"  {test}: {accuracy:.4f}")

if __name__ == "__main__":
    main()

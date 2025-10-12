# 🔍 Deepfake Detection System

A comprehensive deepfake detection system using state-of-the-art deep learning models including EfficientNet, XceptionNet, and a hybrid ensemble approach combining CNN + Transformer architectures.

## 🌟 Features

- **Multiple Model Architectures**: EfficientNet, XceptionNet, and Hybrid Ensemble (CNN + Transformer + XceptionNet)
- **Comprehensive Data Pipeline**: Automated data collection, preprocessing, and augmentation
- **Advanced Training**: Transfer learning, hyperparameter tuning, and early stopping
- **Robust Evaluation**: ROC-AUC, confusion matrix, precision, recall, F1-score, and adversarial robustness testing
- **Multiple Deployment Options**: FastAPI REST API, Streamlit web interface, and command-line demo
- **Production Ready**: Model export, API integration, and comprehensive logging

## 📁 Project Structure

```
deepfake-detection/
├── dataset/                    # Your dataset (real/ and fake/ folders)
├── src/
│   ├── data_collection.py     # Dataset management and validation
│   ├── data_preprocessing.py  # Data preprocessing and augmentation
│   ├── models.py             # Model architectures
│   ├── training.py           # Training pipeline
│   ├── evaluation.py         # Evaluation metrics and robustness testing
│   └── deployment.py         # API and web interface
├── models/                   # Saved trained models
├── processed_data/           # Preprocessed data splits
├── results/                  # Evaluation results and plots
├── logs/                     # Training logs
├── config.py                 # Configuration settings
├── main.py                   # Main entry point
├── requirements.txt          # Dependencies
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd deepfake-detection

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup

Place your dataset in the following structure:
```
dataset/
├── real/          # Real images
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── fake/          # Fake/deepfake images
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### 3. Data Preprocessing

```bash
python main.py preprocess
```

This will:
- Validate dataset structure
- Create metadata files
- Split data into train/validation/test sets (80/10/10)
- Generate data visualizations

### 4. Model Training

```bash
# Train all models
python main.py train

# Train specific model
python main.py train --model efficientnet
python main.py train --model xception
python main.py train --model hybrid_ensemble
```

### 5. Model Evaluation

```bash
python main.py evaluate
```

This will generate:
- Comprehensive metrics (accuracy, precision, recall, F1, AUC)
- Confusion matrices
- ROC and Precision-Recall curves
- Robustness testing results
- Calibration curves

### 6. Deployment

#### FastAPI REST API
```bash
python main.py api --port 8000
```

#### Streamlit Web Interface
```bash
python main.py streamlit --port 8501
```

#### Interactive Demo
```bash
python main.py demo
```

## 🏗️ Model Architectures

### 1. EfficientNet
- **Base**: EfficientNet-B4 with ImageNet pretraining
- **Features**: Compound scaling, mobile-optimized architecture
- **Performance**: High accuracy with efficient inference

### 2. XceptionNet
- **Base**: Xception with depthwise separable convolutions
- **Features**: Excellent feature extraction for manipulation detection
- **Performance**: Strong performance on deepfake detection tasks

### 3. Hybrid Ensemble
- **Components**: 
  - CNN backbone (EfficientNet-B4)
  - Vision Transformer (ViT-base-patch16-224)
  - XceptionNet
- **Fusion Methods**: Weighted average, learned fusion, attention fusion
- **Performance**: Best overall performance combining multiple approaches

## 📊 Data Augmentation

The system includes comprehensive data augmentation:
- **Geometric**: Rotation, flips, scaling, translation
- **Photometric**: Brightness, contrast, saturation, hue adjustments
- **Noise**: Gaussian noise, blur, dropout
- **Advanced**: CLAHE, coarse dropout

## 🎯 Training Features

- **Transfer Learning**: Pretrained weights from ImageNet
- **Mixed Precision**: Automatic mixed precision for faster training
- **Early Stopping**: Prevent overfitting with patience-based stopping
- **Learning Rate Scheduling**: ReduceLROnPlateau, CosineAnnealing, StepLR
- **Multiple Optimizers**: Adam, AdamW, SGD with configurable parameters
- **Comprehensive Logging**: TensorBoard and Weights & Biases integration

## 📈 Evaluation Metrics

### Standard Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC and Average Precision
- Confusion Matrix and Classification Report
- Per-class metrics and confidence analysis

### Robustness Testing
- **Noise Robustness**: Gaussian noise at various levels
- **Brightness Robustness**: Brightness variation testing
- **Compression Robustness**: JPEG compression at different quality levels
- **Adversarial Robustness**: FGSM adversarial attack testing

## 🌐 API Usage

### REST API Endpoints

#### Upload Image File
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@image.jpg"
```

#### Base64 Image
```bash
curl -X POST "http://localhost:8000/predict/base64" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{"image": "base64_encoded_image_data"}'
```

#### Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict/batch" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "files=@image1.jpg" \
     -F "files=@image2.jpg"
```

### Response Format
```json
{
  "prediction": "Real",
  "confidence": 0.95,
  "probabilities": {
    "real": 0.95,
    "fake": 0.05
  },
  "predicted_class": 0,
  "inference_time": 0.123,
  "timestamp": "2025-10-09T10:30:00"
}
```

## ⚙️ Configuration

Key configuration options in `config.py`:

```python
# Data configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Training configuration
EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
PATIENCE = 10

# Model configurations
MODEL_CONFIGS = {
    'efficientnet': {
        'model_name': 'efficientnet-b4',
        'pretrained': True,
        'num_classes': 2
    },
    'hybrid_ensemble': {
        'cnn_backbone': 'efficientnet-b4',
        'transformer_model': 'google/vit-base-patch16-224',
        'ensemble_method': 'weighted_average'
    }
}
```

## 🔧 Advanced Usage

### Custom Model Training
```python
from src.models import ModelFactory
from src.training import DeepfakeTrainer

# Create custom model
model = ModelFactory.create_model('hybrid_ensemble', config)

# Custom training configuration
training_config = {
    'epochs': 100,
    'learning_rate': 0.0001,
    'optimizer': 'adamw',
    'scheduler': 'cosine_annealing',
    'use_amp': True
}

# Train model
trainer = DeepfakeTrainer(model, train_loader, val_loader, test_loader, training_config, save_dir)
history = trainer.train(epochs=100)
```

### Custom Evaluation
```python
from src.evaluation import ModelEvaluator, RobustnessEvaluator

# Comprehensive evaluation
evaluator = ModelEvaluator(model, device)
metrics = evaluator.evaluate_comprehensive(test_loader, save_dir)

# Robustness testing
robustness_evaluator = RobustnessEvaluator(model, device)
robustness_results = robustness_evaluator.comprehensive_robustness_evaluation(test_loader)
```

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- See `requirements.txt` for complete list

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- EfficientNet: [Tan & Le, 2019](https://arxiv.org/abs/1905.11946)
- Xception: [Chollet, 2017](https://arxiv.org/abs/1610.02357)
- Vision Transformer: [Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929)
- DeepDetect-2025 and other open-source datasets

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation
- Review the code examples

---

**Note**: This system is designed for educational and research purposes. Always validate results and consider ethical implications when working with deepfake detection technology.

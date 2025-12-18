"""
Model architectures for deepfake detection:
- EfficientNet
- XceptionNet  
- Hybrid Ensemble (CNN + Transformer + XceptionNet)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
from transformers import ViTModel, ViTConfig
from efficientnet_pytorch import EfficientNet
import numpy as np
from typing import Dict, List, Optional, Tuple

class EfficientNetModel(nn.Module):
    """EfficientNet-based deepfake detector"""
    
    def __init__(self, 
                 model_name: str = 'efficientnet-b4',
                 num_classes: int = 2,
                 pretrained: bool = True,
                 dropout_rate: float = 0.3):
        super(EfficientNetModel, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pretrained EfficientNet
        if pretrained:
            self.backbone = EfficientNet.from_pretrained(model_name)
        else:
            self.backbone = EfficientNet.from_name(model_name)
        
        # Get the number of features from the backbone
        num_features = self.backbone._fc.in_features
        
        # Replace the classifier
        self.backbone._fc = nn.Identity()
        
        # Custom classifier with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Classify
        output = self.classifier(features)
        
        return output
    
    def get_features(self, x):
        """Extract features without classification"""
        return self.backbone(x)

class XceptionNet(nn.Module):
    """Xception-based deepfake detector"""
    
    def __init__(self, 
                 num_classes: int = 2,
                 pretrained: bool = True,
                 dropout_rate: float = 0.3):
        super(XceptionNet, self).__init__()
        
        self.num_classes = num_classes
        
        # Load Xception from timm
        self.backbone = timm.create_model('xception', pretrained=pretrained)
        
        # Get number of features
        num_features = self.backbone.num_features
        
        # Replace classifier
        self.backbone.reset_classifier(0)  # Remove original classifier
        
        # Custom classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Classify
        output = self.classifier(features)
        
        return output
    
    def get_features(self, x):
        """Extract features without classification"""
        features = self.backbone(x)
        return F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)

class VisionTransformerHead(nn.Module):
    """Vision Transformer component for ensemble"""
    
    def __init__(self, 
                 model_name: str = 'google/vit-base-patch16-224',
                 num_classes: int = 2,
                 dropout_rate: float = 0.1):
        super(VisionTransformerHead, self).__init__()
        
        # Load pretrained ViT
        self.vit = ViTModel.from_pretrained(model_name)
        
        # Freeze some layers for transfer learning
        for param in list(self.vit.parameters())[:-4]:  # Freeze all but last 4 layers
            param.requires_grad = False
        
        # Get hidden size
        hidden_size = self.vit.config.hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # ViT expects input in range [0, 1]
        if x.max() > 1.0:
            x = x / 255.0
        
        # Get ViT outputs
        outputs = self.vit(pixel_values=x)
        
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0]
        
        # Classify
        logits = self.classifier(cls_output)
        
        return logits
    
    def get_features(self, x):
        """Extract features without classification"""
        if x.max() > 1.0:
            x = x / 255.0
        
        outputs = self.vit(pixel_values=x)
        return outputs.last_hidden_state[:, 0]

class HybridEnsembleModel(nn.Module):
    """Hybrid ensemble combining CNN + Transformer + XceptionNet"""
    
    def __init__(self,
                 cnn_backbone: str = 'efficientnet-b4',
                 transformer_model: str = 'google/vit-base-patch16-224',
                 num_classes: int = 2,
                 ensemble_method: str = 'weighted_average',
                 dropout_rate: float = 0.3):
        super(HybridEnsembleModel, self).__init__()
        
        self.ensemble_method = ensemble_method
        self.num_classes = num_classes
        
        # Initialize individual models
        self.cnn_model = EfficientNetModel(
            model_name=cnn_backbone,
            num_classes=num_classes,
            pretrained=True,
            dropout_rate=dropout_rate
        )
        
        self.transformer_model = VisionTransformerHead(
            model_name=transformer_model,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
        
        self.xception_model = XceptionNet(
            num_classes=num_classes,
            pretrained=True,
            dropout_rate=dropout_rate
        )
        
        # Ensemble fusion layers
        if ensemble_method == 'learned_fusion':
            # Feature dimensions
            cnn_features = 1792  # EfficientNet-B4 features
            transformer_features = 768  # ViT-base features
            xception_features = 2048  # Xception features
            
            total_features = cnn_features + transformer_features + xception_features
            
            self.fusion_layer = nn.Sequential(
                nn.Linear(total_features, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(1024, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(256, num_classes)
            )
        
        elif ensemble_method == 'attention_fusion':
            self.attention_weights = nn.Parameter(torch.ones(3) / 3)
            self.attention_layer = nn.Sequential(
                nn.Linear(num_classes * 3, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 3),
                nn.Softmax(dim=1)
            )
        
        elif ensemble_method == 'weighted_average':
            # Learnable weights for weighted average
            self.ensemble_weights = nn.Parameter(torch.ones(3) / 3)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize fusion layer weights"""
        if hasattr(self, 'fusion_layer'):
            for m in self.fusion_layer.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Get predictions from individual models
        cnn_output = self.cnn_model(x)
        transformer_output = self.transformer_model(x)
        xception_output = self.xception_model(x)
        
        if self.ensemble_method == 'weighted_average':
            # Weighted average of predictions
            weights = F.softmax(self.ensemble_weights, dim=0)
            ensemble_output = (weights[0] * cnn_output + 
                             weights[1] * transformer_output + 
                             weights[2] * xception_output)
        
        elif self.ensemble_method == 'learned_fusion':
            # Extract features and concatenate
            cnn_features = self.cnn_model.get_features(x)
            transformer_features = self.transformer_model.get_features(x)
            xception_features = self.xception_model.get_features(x)
            
            # Concatenate features
            combined_features = torch.cat([
                cnn_features, transformer_features, xception_features
            ], dim=1)
            
            # Pass through fusion layer
            ensemble_output = self.fusion_layer(combined_features)
        
        elif self.ensemble_method == 'attention_fusion':
            # Concatenate predictions
            all_predictions = torch.cat([
                cnn_output, transformer_output, xception_output
            ], dim=1)
            
            # Compute attention weights
            attention_weights = self.attention_layer(all_predictions)
            
            # Apply attention to individual predictions
            ensemble_output = (attention_weights[:, 0:1] * cnn_output + 
                             attention_weights[:, 1:2] * transformer_output + 
                             attention_weights[:, 2:3] * xception_output)
        
        else:
            # Simple average
            ensemble_output = (cnn_output + transformer_output + xception_output) / 3
        
        return ensemble_output
    
    def get_individual_predictions(self, x):
        """Get predictions from individual models"""
        with torch.no_grad():
            cnn_output = self.cnn_model(x)
            transformer_output = self.transformer_model(x)
            xception_output = self.xception_model(x)
        
        return {
            'cnn': F.softmax(cnn_output, dim=1),
            'transformer': F.softmax(transformer_output, dim=1),
            'xception': F.softmax(xception_output, dim=1)
        }

class ModelFactory:
    """Factory class for creating models"""
    
    @staticmethod
    def create_model(model_type: str, config: Dict) -> nn.Module:
        """Create model based on type and configuration"""
        
        if model_type == 'efficientnet':
            return EfficientNetModel(
                model_name=config.get('model_name', 'efficientnet-b4'),
                num_classes=config.get('num_classes', 2),
                pretrained=config.get('pretrained', True),
                dropout_rate=config.get('dropout_rate', 0.3)
            )
        
        elif model_type == 'xception':
            return XceptionNet(
                num_classes=config.get('num_classes', 2),
                pretrained=config.get('pretrained', True),
                dropout_rate=config.get('dropout_rate', 0.3)
            )
        
        elif model_type == 'vit':
            return VisionTransformerHead(
                model_name=config.get('model_name', 'google/vit-base-patch16-224'),
                num_classes=config.get('num_classes', 2),
                dropout_rate=config.get('dropout_rate', 0.1)
            )
        
        elif model_type == 'hybrid_ensemble':
            return HybridEnsembleModel(
                cnn_backbone=config.get('cnn_backbone', 'efficientnet-b4'),
                transformer_model=config.get('transformer_model', 'google/vit-base-patch16-224'),
                num_classes=config.get('num_classes', 2),
                ensemble_method=config.get('ensemble_method', 'weighted_average'),
                dropout_rate=config.get('dropout_rate', 0.3)
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_model_info(model: nn.Module) -> Dict:
        """Get model information"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'model_type': model.__class__.__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }

def test_models():
    """Test model creation and forward pass"""
    
    # Test input
    batch_size = 2
    channels = 3
    height = width = 224
    x = torch.randn(batch_size, channels, height, width)
    
    print("Testing model architectures...")
    
    # Test EfficientNet
    print("\n1. Testing EfficientNet...")
    efficientnet = EfficientNetModel()
    output = efficientnet(x)
    print(f"EfficientNet output shape: {output.shape}")
    print(f"Model info: {ModelFactory.get_model_info(efficientnet)}")
    
    # Test XceptionNet
    print("\n2. Testing XceptionNet...")
    xception = XceptionNet()
    output = xception(x)
    print(f"XceptionNet output shape: {output.shape}")
    print(f"Model info: {ModelFactory.get_model_info(xception)}")
    
    # Test Vision Transformer
    print("\n3. Testing Vision Transformer...")
    vit = VisionTransformerHead()
    output = vit(x)
    print(f"ViT output shape: {output.shape}")
    print(f"Model info: {ModelFactory.get_model_info(vit)}")
    
    # Test Hybrid Ensemble
    print("\n4. Testing Hybrid Ensemble...")
    ensemble = HybridEnsembleModel()
    output = ensemble(x)
    print(f"Ensemble output shape: {output.shape}")
    print(f"Model info: {ModelFactory.get_model_info(ensemble)}")
    
    # Test individual predictions
    individual_preds = ensemble.get_individual_predictions(x)
    print("Individual predictions:")
    for model_name, pred in individual_preds.items():
        print(f"  {model_name}: {pred.shape}")

if __name__ == "__main__":
    test_models()

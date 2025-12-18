"""
Model architecture for deepfake detection:
- EfficientNet-B4
"""
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from typing import Dict

class EfficientNetModel(nn.Module):
    """EfficientNet-B4 based deepfake detector"""
    
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
        else:
            raise ValueError(f"Unknown model type: {model_type}. Only 'efficientnet' is supported.")
    
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
    
    print("Testing EfficientNet-B4...")
    efficientnet = EfficientNetModel()
    output = efficientnet(x)
    print(f"EfficientNet-B4 output shape: {output.shape}")
    print(f"Model info: {ModelFactory.get_model_info(efficientnet)}")


if __name__ == "__main__":
    test_models()

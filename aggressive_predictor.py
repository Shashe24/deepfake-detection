"""
Aggressive predictor for deepfake detection
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

def aggressive_predict(model, image, device, threshold=0.25):
    """
    Improved aggressive prediction with better fake detection
    
    Args:
        model: Trained model
        image: PIL Image or numpy array
        device: torch device
        threshold: Fake probability threshold (lower = more sensitive to fakes)
    
    Returns:
        dict: Prediction results with confidence
    """
    
    # Simple but effective transform
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    try:
        # Apply transform
        transformed = transform(image=image)
        input_tensor = transformed['image'].unsqueeze(0).to(device)
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get probabilities for both classes
            real_prob = probabilities[0][0].item()
            fake_prob = probabilities[0][1].item()
            
            # Additional analysis for better detection
            try:
                # Analyze image quality
                if len(image.shape) == 3:
                    gray = np.mean(image, axis=2)
                else:
                    gray = image
                
                # Calculate image statistics
                image_variance = np.var(gray)
                image_mean = np.mean(gray)
                
                # Quality-based adjustment
                quality_factor = 1.0
                if image_variance > 2000:  # High variance might indicate manipulation
                    quality_factor += 0.1
                if image_mean < 50 or image_mean > 200:  # Unusual brightness
                    quality_factor += 0.05
                
                # Adjust fake probability based on quality
                adjusted_fake_prob = min(fake_prob * quality_factor, 1.0)
                
            except Exception:
                adjusted_fake_prob = fake_prob
            
            # Use adjusted probability for final decision
            is_fake = adjusted_fake_prob > threshold
            
            # Calculate confidence based on how far from 50/50
            if is_fake:
                confidence = adjusted_fake_prob
                prediction = 'Fake'
            else:
                confidence = 1 - adjusted_fake_prob
                prediction = 'Real'
            
            # Boost confidence for clear cases
            if confidence > 0.8:
                confidence = min(0.98, confidence + 0.05)
            elif confidence < 0.2:
                confidence = max(0.02, confidence - 0.05)
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'fake_probability': adjusted_fake_prob,
                'real_probability': 1 - adjusted_fake_prob,
                'threshold': threshold,
                'is_fake': is_fake,
                'analysis_details': {
                    'raw_fake_prob': fake_prob,
                    'raw_real_prob': real_prob,
                    'quality_adjusted': adjusted_fake_prob != fake_prob,
                    'threshold_used': threshold
                }
            }
            
    except Exception as e:
        # Fallback prediction
        return {
            'prediction': 'Real',
            'confidence': 0.5,
            'fake_probability': 0.5,
            'real_probability': 0.5,
            'threshold': threshold,
            'is_fake': False,
            'analysis_details': {'error': str(e)}
        }
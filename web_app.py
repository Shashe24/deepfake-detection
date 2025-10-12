"""
Web application for deepfake detection using Flask
"""
import os
import io
import base64
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import json
import cv2

# Import our models and config
from src.models import ModelFactory
from config import (
    ALLOWED_IMAGE_EXTENSIONS, MAX_FILE_SIZE_MB, MIN_IMAGE_SIZE, MAX_IMAGE_SIZE,
    MIN_MODEL_FILE_SIZE_KB, MODEL_VALIDATION_INPUT_SHAPE, INPUT_NORMALIZATION
)
from aggressive_predictor import aggressive_predict

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE_MB * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for model
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_type = 'efficientnet'  # Default model - will use demo model

# Image preprocessing using config
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=INPUT_NORMALIZATION['mean'], std=INPUT_NORMALIZATION['std']),
    ToTensorV2()
])

def load_model(model_path='models/efficientnet/best_model.pth'):
    """Load the trained model with comprehensive error handling"""
    global model
    
    try:
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            return False
        
        # Validate model file
        if not os.path.isfile(model_path):
            print(f"Model path is not a file: {model_path}")
            return False
            
        # Check file size (basic validation)
        file_size = os.path.getsize(model_path)
        if file_size < MIN_MODEL_FILE_SIZE_KB * 1024:  # Less than configured minimum is suspicious
            print(f"Model file seems too small: {file_size} bytes")
            return False
        
        # Load checkpoint with proper error handling
        try:
            # Try loading with weights_only=True first (secure)
            try:
                checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            except Exception as weights_only_error:
                print(f"Secure loading failed, trying with weights_only=False: {weights_only_error}")
                # Fallback to weights_only=False for compatibility
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        except Exception as load_error:
            print(f"Failed to load model file: {load_error}")
            return False
        
        # Validate checkpoint structure
        required_keys = ['model_state_dict']
        for key in required_keys:
            if key not in checkpoint:
                print(f"Missing required key in checkpoint: {key}")
                return False
        
        # Get model configuration with defaults
        config = checkpoint.get('config', {
            'model_name': 'efficientnet-b0',
            'num_classes': 2,
            'pretrained': False,
            'dropout_rate': 0.3
        })
        
        # Create model with error handling
        try:
            model = ModelFactory.create_model(model_type, config)
        except Exception as model_error:
            print(f"Failed to create model: {model_error}")
            return False
        
        # Load state dict with validation
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as state_error:
            print(f"Failed to load model state: {state_error}")
            return False
        
        # Move to device
        try:
            model.to(device)
            model.eval()
        except Exception as device_error:
            print(f"Failed to move model to device: {device_error}")
            return False
        
        # Validate model can process a dummy input
        try:
            dummy_input = torch.randn(*MODEL_VALIDATION_INPUT_SHAPE).to(device)
            with torch.no_grad():
                _ = model(dummy_input)
        except Exception as validation_error:
            print(f"Model validation failed: {validation_error}")
            return False
        
        print(f"Model loaded successfully from {model_path}")
        print(f"Model metrics: {checkpoint.get('metrics', {})}")
        
        return True
        
    except Exception as e:
        print(f"Unexpected error loading model: {e}")
        return False

def preprocess_image(image):
    """Preprocess image for model input"""
    # Convert PIL to numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert BGR to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply transformations
    transformed = transform(image=image)
    return transformed['image'].unsqueeze(0)  # Add batch dimension

def predict_image(image_tensor, original_image):
    """Make prediction using aggressive AI detection"""
    global model
    
    if model is None:
        return None, "Model not loaded"
    
    try:
        # Use enhanced aggressive predictor with better fake detection
        result = aggressive_predict(model, original_image, device, threshold=0.25)
        return result, None
        
    except Exception as e:
        return None, str(e)

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction with comprehensive error handling"""
    try:
        # Check if file is present in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file extension
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ALLOWED_IMAGE_EXTENSIONS:
            return jsonify({'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_IMAGE_EXTENSIONS)}'}), 400
        
        # Check file size
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            return jsonify({'error': f'File too large. Maximum size: {MAX_FILE_SIZE_MB}MB'}), 400
        
        if file_size == 0:
            return jsonify({'error': 'Empty file'}), 400
        
        # Save uploaded file with error handling
        try:
            filename = secure_filename(file.filename)
            if not filename:
                return jsonify({'error': 'Invalid filename'}), 400
                
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
        except Exception as save_error:
            return jsonify({'error': f'Failed to save file: {str(save_error)}'}), 500
        
        # Load and validate image
        try:
            image = Image.open(filepath)
            # Validate image format
            if image.format not in ['JPEG', 'PNG', 'BMP', 'TIFF']:
                return jsonify({'error': 'Unsupported image format'}), 400
            
            # Convert to RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            # Validate image dimensions
            width, height = image.size
            if width < MIN_IMAGE_SIZE[0] or height < MIN_IMAGE_SIZE[1]:
                return jsonify({'error': f'Image too small. Minimum size: {MIN_IMAGE_SIZE[0]}x{MIN_IMAGE_SIZE[1]}'}), 400
            if width > MAX_IMAGE_SIZE[0] or height > MAX_IMAGE_SIZE[1]:
                return jsonify({'error': f'Image too large. Maximum size: {MAX_IMAGE_SIZE[0]}x{MAX_IMAGE_SIZE[1]}'}), 400
                
        except Exception as image_error:
            # Clean up file on error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Invalid image file: {str(image_error)}'}), 400
        
        # Preprocess image with error handling
        try:
            image_tensor = preprocess_image(image)
        except Exception as preprocess_error:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Image preprocessing failed: {str(preprocess_error)}'}), 500
        
        # Make prediction with error handling
        try:
            result, error = predict_image(image_tensor, image)
            if error:
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({'error': error}), 500
        except Exception as predict_error:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Prediction failed: {str(predict_error)}'}), 500
        
        # Convert image to base64 for display
        try:
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='JPEG', quality=85)
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
        except Exception as encode_error:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Image encoding failed: {str(encode_error)}'}), 500
        
        # Clean up uploaded file
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception as cleanup_error:
            print(f"Warning: Failed to cleanup file {filepath}: {cleanup_error}")
        
        return jsonify({
            'success': True,
            'result': result,
            'image': img_str
        })
        
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/model_info')
def model_info():
    """Get model information"""
    global model
    
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Load model metrics if available
        model_path = 'models/efficientnet/best_model.pth'
        if not os.path.exists(model_path):
            model_path = 'models/demo/best_model.pth'
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            metrics = checkpoint.get('metrics', {})
            config = checkpoint.get('config', {})
        else:
            metrics = {}
            config = {}
        
        # Get model info
        model_info = ModelFactory.get_model_info(model)
        
        return jsonify({
            'model_type': model_type,
            'device': str(device),
            'model_info': model_info,
            'metrics': metrics,
            'config': config
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/switch_model', methods=['POST'])
def switch_model():
    """Switch between different models"""
    global model, model_type
    
    data = request.get_json()
    new_model_type = data.get('model_type', 'efficientnet')
    
    if new_model_type not in ['efficientnet', 'xception', 'hybrid_ensemble']:
        return jsonify({'error': 'Invalid model type'}), 400
    
    model_type = new_model_type
    model_path = f'models/{model_type}/best_model.pth'
    
    success = load_model(model_path)
    
    if success:
        return jsonify({'success': True, 'model_type': model_type})
    else:
        return jsonify({'error': 'Failed to load model'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device)
    })

if __name__ == '__main__':
    # Load model on startup
    print("Loading model...")
    load_model()
    
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=True)

"""
Deployment utilities and API integration for deepfake detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import io
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
import time
import logging
from datetime import datetime

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Streamlit imports
import streamlit as st

from src.models import ModelFactory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepfakeDetector:
    """Main deepfake detection class for deployment"""
    
    def __init__(self, model_path: str, model_type: str, device: str = 'auto'):
        self.model_path = Path(model_path)
        self.model_type = model_type
        
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Setup preprocessing
        self.transform = self._get_transform()
        
        # Class names
        self.class_names = ['Real', 'Fake']
        
        logger.info(f"DeepfakeDetector initialized with {model_type} on {self.device}")
    
    def _load_model(self) -> nn.Module:
        """Load trained model"""
        try:
            # Try loading with weights_only=True first (secure)
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
            except Exception as weights_only_error:
                logger.warning(f"Secure loading failed: {weights_only_error}")
                logger.info("Attempting to load with weights_only=False (less secure but compatible)")
                # Fallback to weights_only=False for compatibility
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Get model configuration
            model_config = checkpoint.get('config', {})
            
            # Create model
            model = ModelFactory.create_model(self.model_type, model_config)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _get_transform(self):
        """Get preprocessing transforms"""
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def preprocess_image(self, image: Union[np.ndarray, Image.Image, str]) -> torch.Tensor:
        """Preprocess image for inference"""
        
        # Handle different input types
        if isinstance(image, str):
            # File path
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            # PIL Image
            image = np.array(image)
            if len(image.shape) == 3 and image.shape[2] == 4:
                # RGBA to RGB
                image = image[:, :, :3]
        elif isinstance(image, np.ndarray):
            # Numpy array
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Assume BGR, convert to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Apply transforms
        transformed = self.transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
        
        return image_tensor.to(self.device)
    
    def predict(self, image: Union[np.ndarray, Image.Image, str]) -> Dict[str, Any]:
        """Predict if image is real or fake"""
        
        start_time = time.time()
        
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                
                # Get predictions
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
                
                # Get probabilities for both classes
                real_prob = probabilities[0][0].item()
                fake_prob = probabilities[0][1].item()
            
            inference_time = time.time() - start_time
            
            result = {
                'prediction': self.class_names[predicted_class],
                'confidence': float(confidence),
                'probabilities': {
                    'real': float(real_prob),
                    'fake': float(fake_prob)
                },
                'predicted_class': int(predicted_class),
                'inference_time': float(inference_time),
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def predict_batch(self, images: List[Union[np.ndarray, Image.Image, str]]) -> List[Dict[str, Any]]:
        """Predict on batch of images"""
        
        results = []
        
        for image in images:
            try:
                result = self.predict(image)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process image: {e}")
                results.append({
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        return results

# FastAPI Application
app = FastAPI(
    title="Deepfake Detection API",
    description="API for detecting deepfake images using deep learning models",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global detector instance
detector: Optional[DeepfakeDetector] = None

def get_detector():
    """Dependency to get detector instance"""
    if detector is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return detector

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global detector
    
    try:
        # Load the best available model
        from config import MODELS_DIR
        
        # Try to load hybrid ensemble first, then fallback to others
        model_priority = ['hybrid_ensemble', 'efficientnet', 'xception']
        
        for model_type in model_priority:
            model_path = MODELS_DIR / model_type / 'best_model.pth'
            if model_path.exists():
                detector = DeepfakeDetector(str(model_path), model_type)
                logger.info(f"Loaded {model_type} model for API")
                break
        
        if detector is None:
            logger.error("No trained models found!")
            
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API documentation"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Deepfake Detection API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { color: #007bff; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 Deepfake Detection API</h1>
            <p>Welcome to the Deepfake Detection API. This service uses advanced deep learning models to detect manipulated images.</p>
            
            <h2>Available Endpoints:</h2>
            
            <div class="endpoint">
                <p><span class="method">GET</span> <code>/health</code></p>
                <p>Check API health status</p>
            </div>
            
            <div class="endpoint">
                <p><span class="method">POST</span> <code>/predict</code></p>
                <p>Upload an image file to detect if it's real or fake</p>
            </div>
            
            <div class="endpoint">
                <p><span class="method">POST</span> <code>/predict/base64</code></p>
                <p>Send base64 encoded image for prediction</p>
            </div>
            
            <div class="endpoint">
                <p><span class="method">GET</span> <code>/model/info</code></p>
                <p>Get information about the loaded model</p>
            </div>
            
            <h2>Interactive Documentation:</h2>
            <p>Visit <a href="/docs">/docs</a> for interactive API documentation</p>
            
            <h2>Example Usage:</h2>
            <pre>
curl -X POST "http://localhost:8000/predict" \\
     -H "accept: application/json" \\
     -H "Content-Type: multipart/form-data" \\
     -F "file=@your_image.jpg"
            </pre>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": detector is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/info")
async def model_info(detector: DeepfakeDetector = Depends(get_detector)):
    """Get model information"""
    return {
        "model_type": detector.model_type,
        "model_path": str(detector.model_path),
        "device": str(detector.device),
        "class_names": detector.class_names
    }

@app.post("/predict")
async def predict_image(file: UploadFile = File(...), 
                       detector: DeepfakeDetector = Depends(get_detector)):
    """Predict if uploaded image is real or fake"""
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Predict
        result = detector.predict(image)
        
        # Add file info
        result['file_info'] = {
            'filename': file.filename,
            'content_type': file.content_type,
            'size_bytes': len(contents)
        }
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/base64")
async def predict_base64(data: Dict[str, str],
                        detector: DeepfakeDetector = Depends(get_detector)):
    """Predict from base64 encoded image"""
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Predict
        result = detector.predict(image)
        
        return JSONResponse(content=result)
        
    except KeyError:
        raise HTTPException(status_code=400, detail="Missing 'image' field in request")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...),
                       detector: DeepfakeDetector = Depends(get_detector)):
    """Predict on multiple images"""
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")
    
    results = []
    
    for file in files:
        try:
            # Validate file type
            if not file.content_type.startswith('image/'):
                results.append({
                    'filename': file.filename,
                    'error': 'File must be an image'
                })
                continue
            
            # Read and process image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Predict
            result = detector.predict(image)
            result['filename'] = file.filename
            results.append(result)
            
        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': str(e)
            })
    
    return JSONResponse(content={'results': results})

# Streamlit Web Interface
def create_streamlit_app():
    """Create Streamlit web interface"""
    
    st.set_page_config(
        page_title="Deepfake Detection",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Deepfake Detection System")
    st.markdown("Upload an image to detect if it's real or artificially generated (deepfake)")
    
    # Sidebar
    st.sidebar.title("Model Information")
    
    # Initialize detector
    @st.cache_resource
    def load_detector():
        from config import MODELS_DIR
        
        # Try to load the best available model
        model_priority = ['hybrid_ensemble', 'efficientnet', 'xception']
        
        for model_type in model_priority:
            model_path = MODELS_DIR / model_type / 'best_model.pth'
            if model_path.exists():
                return DeepfakeDetector(str(model_path), model_type)
        
        return None
    
    detector = load_detector()
    
    if detector is None:
        st.error("No trained models found! Please train a model first.")
        return
    
    # Display model info
    st.sidebar.info(f"**Model Type:** {detector.model_type}")
    st.sidebar.info(f"**Device:** {detector.device}")
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image to analyze"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Predict button
            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner("Analyzing image..."):
                    try:
                        # Convert to RGB if necessary
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        
                        # Predict
                        result = detector.predict(image)
                        
                        # Display results in the second column
                        with col2:
                            st.header("Analysis Results")
                            
                            # Main prediction
                            prediction = result['prediction']
                            confidence = result['confidence']
                            
                            if prediction == 'Real':
                                st.success(f"✅ **{prediction}** (Confidence: {confidence:.2%})")
                            else:
                                st.error(f"⚠️ **{prediction}** (Confidence: {confidence:.2%})")
                            
                            # Probability breakdown
                            st.subheader("Probability Breakdown")
                            
                            real_prob = result['probabilities']['real']
                            fake_prob = result['probabilities']['fake']
                            
                            # Progress bars
                            st.write("**Real:**")
                            st.progress(real_prob)
                            st.write(f"{real_prob:.2%}")
                            
                            st.write("**Fake:**")
                            st.progress(fake_prob)
                            st.write(f"{fake_prob:.2%}")
                            
                            # Additional info
                            st.subheader("Additional Information")
                            st.write(f"**Inference Time:** {result['inference_time']:.3f} seconds")
                            st.write(f"**Timestamp:** {result['timestamp']}")
                            
                            # Confidence interpretation
                            st.subheader("Confidence Interpretation")
                            if confidence >= 0.9:
                                st.info("🎯 Very High Confidence")
                            elif confidence >= 0.7:
                                st.info("✅ High Confidence")
                            elif confidence >= 0.6:
                                st.warning("⚠️ Moderate Confidence")
                            else:
                                st.warning("❓ Low Confidence - Results may be unreliable")
                    
                    except Exception as e:
                        st.error(f"Error analyzing image: {str(e)}")
    
    with col2:
        if uploaded_file is None:
            st.header("Results")
            st.info("👆 Upload an image to see analysis results")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Note:** This system is for educational and research purposes. "
        "Results should be interpreted carefully and not used as the sole basis for important decisions."
    )

def run_api_server(host: str = "0.0.0.0", port: int = 8000):
    """Run FastAPI server"""
    uvicorn.run(app, host=host, port=port)

def run_streamlit_app(port: int = 8501):
    """Run Streamlit app"""
    create_streamlit_app()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "api":
            print("Starting FastAPI server...")
            run_api_server()
        elif sys.argv[1] == "streamlit":
            print("Starting Streamlit app...")
            run_streamlit_app()
        else:
            print("Usage: python deployment.py [api|streamlit]")
    else:
        print("Usage: python deployment.py [api|streamlit]")
        print("  api       - Run FastAPI server")
        print("  streamlit - Run Streamlit web interface")

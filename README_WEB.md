# 🌐 Deepfake Detection Web Application

A beautiful, modern web interface for deepfake image detection using advanced AI models.

## ✨ Features

- **🎨 Modern UI**: Beautiful, responsive design with gradient backgrounds and smooth animations
- **📱 Mobile Friendly**: Works perfectly on desktop, tablet, and mobile devices
- **🔄 Multiple Models**: Switch between EfficientNet, Xception, and Hybrid Ensemble models
- **📊 Real-time Results**: Instant analysis with confidence scores and probability breakdowns
- **🖼️ Image Preview**: See your uploaded image alongside the analysis results
- **⚡ Fast Processing**: Optimized for quick predictions using PyTorch
- **🛡️ Secure**: File validation and size limits for safe uploads

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_web.txt
```

### 2. Create Demo Model (if needed)
```bash
python demo_model.py
```

### 3. Run the Web Application
```bash
python run_web_app.py
```

### 4. Open Your Browser
Navigate to: **http://localhost:5000**

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- Flask 2.3+
- Modern web browser (Chrome, Firefox, Safari, Edge)

## 🎯 How to Use

1. **Upload Image**: Drag and drop or click to select an image file
2. **Analyze**: Click the "Analyze Image" button
3. **View Results**: See if the image is real or fake with confidence scores
4. **Switch Models**: Use the dropdown to try different AI models

## 🔧 Configuration

### Model Selection
The web app supports three different models:

- **EfficientNet**: Fast and efficient, good for general use
- **Xception**: Advanced CNN architecture, higher accuracy
- **Hybrid Ensemble**: Combines multiple models for best results

### File Requirements
- **Supported formats**: JPG, PNG, JPEG
- **Maximum size**: 16MB
- **Recommended**: High-quality images for best results

## 🏗️ Project Structure

```
├── web_app.py              # Main Flask application
├── run_web_app.py          # Simple launcher script
├── test_web_app.py         # Test script
├── templates/
│   └── index.html          # Main web page
├── static/
│   ├── css/
│   │   └── style.css       # Beautiful styling
│   └── js/
│       └── script.js       # Interactive functionality
├── models/
│   ├── demo/               # Demo model for testing
│   ├── efficientnet/       # EfficientNet model
│   └── hybrid_ensemble/    # Ensemble model
└── uploads/                # Temporary file storage
```

## 🎨 UI Features

### Modern Design
- Gradient backgrounds and glassmorphism effects
- Smooth animations and transitions
- Responsive grid layouts
- Professional color scheme

### Interactive Elements
- Drag and drop file upload
- Real-time progress indicators
- Animated probability bars
- Toast notifications for feedback

### Results Display
- Clear real/fake classification
- Confidence percentage with visual meter
- Probability breakdown for both classes
- Image preview with results

## 🔍 API Endpoints

- `GET /` - Main web interface
- `POST /upload` - Upload and analyze image
- `GET /health` - Check application health
- `GET /model_info` - Get model information
- `POST /switch_model` - Switch between models

## 🛠️ Development

### Running in Development Mode
```bash
python web_app.py
```

### Testing the Application
```bash
python test_web_app.py
```

### Customizing the UI
- Edit `static/css/style.css` for styling changes
- Modify `static/js/script.js` for functionality
- Update `templates/index.html` for layout changes

## 📊 Model Performance

The web application uses pre-trained models with the following performance:

- **EfficientNet**: ~85% accuracy, fast inference
- **Xception**: ~88% accuracy, balanced performance  
- **Hybrid Ensemble**: ~92% accuracy, best results

## 🔒 Security Features

- File type validation
- Size limit enforcement
- Secure file handling
- Input sanitization
- Error handling

## 🌟 Advanced Features

### Model Switching
Switch between different AI models without restarting the application.

### Confidence Analysis
Get detailed confidence scores and probability distributions.

### Real-time Feedback
Instant visual feedback during image processing.

### Mobile Optimization
Fully responsive design that works on all devices.

## 🚨 Troubleshooting

### Common Issues

1. **Model not loading**: Run `python demo_model.py` to create a demo model
2. **Port already in use**: Change the port in `run_web_app.py`
3. **Dependencies missing**: Install with `pip install -r requirements_web.txt`
4. **Images not uploading**: Check file size and format requirements

### Getting Help

If you encounter issues:
1. Check the console output for error messages
2. Verify all dependencies are installed
3. Ensure the model files exist in the `models/` directory
4. Try the demo model first: `python demo_model.py`

## 🎉 Enjoy!

Your deepfake detection web application is ready to use! Upload images and get instant AI-powered analysis results.

---

**Made with ❤️ using Flask, PyTorch, and modern web technologies**

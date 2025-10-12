// Global variables
let selectedFile = null;
let isAnalyzing = false;

// DOM elements
const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');
const analyzeBtn = document.getElementById('analyze-btn');
const clearBtn = document.getElementById('clear-btn');
const resultsSection = document.getElementById('results-section');
const loadingOverlay = document.getElementById('loading-overlay');
const currentModelSpan = document.getElementById('current-model');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    loadModelInfo();
});

function initializeApp() {
    console.log('Deepfake Detection App initialized');
    
    // Check if model is loaded
    fetch('/health')
        .then(response => response.json())
        .then(data => {
            if (data.model_loaded) {
                console.log('Model loaded successfully');
                updateModelStatus('healthy');
            } else {
                console.log('Model not loaded');
                updateModelStatus('error');
            }
        })
        .catch(error => {
            console.error('Error checking model status:', error);
            updateModelStatus('error');
        });
}

function setupEventListeners() {
    // File input change
    fileInput.addEventListener('change', handleFileSelect);
    
    // Upload area events
    uploadArea.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    
    // Button events
    analyzeBtn.addEventListener('click', analyzeImage);
    clearBtn.addEventListener('click', clearResults);
    
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        processSelectedFile(file);
    }
}

function handleDragOver(event) {
    event.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(event) {
    event.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(event) {
    event.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        processSelectedFile(files[0]);
    }
}

function processSelectedFile(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showError('Please select a valid image file (JPG, PNG, JPEG)');
        return;
    }
    
    // Validate file size (16MB)
    if (file.size > 16 * 1024 * 1024) {
        showError('File size must be less than 16MB');
        return;
    }
    
    selectedFile = file;
    
    // Update UI
    updateUploadArea(file);
    analyzeBtn.disabled = false;
    clearBtn.disabled = false;
    
    // Hide previous results
    resultsSection.style.display = 'none';
}

function updateUploadArea(file) {
    const uploadContent = uploadArea.querySelector('.upload-content');
    uploadContent.innerHTML = `
        <i class="fas fa-check-circle" style="color: #48bb78;"></i>
        <p><strong>${file.name}</strong></p>
        <p class="file-info">Size: ${formatFileSize(file.size)}</p>
    `;
    uploadArea.style.borderColor = '#48bb78';
    uploadArea.style.background = '#f0fff4';
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

async function analyzeImage() {
    if (!selectedFile || isAnalyzing) return;
    
    isAnalyzing = true;
    showLoading(true);
    
    try {
        const formData = new FormData();
        formData.append('file', selectedFile);
        
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            displayResults(result.result, result.image);
        } else {
            showError(result.error || 'Analysis failed');
        }
        
    } catch (error) {
        console.error('Error analyzing image:', error);
        showError('Network error. Please try again.');
    } finally {
        isAnalyzing = false;
        showLoading(false);
    }
}

function displayResults(result, imageData) {
    // Show results section
    resultsSection.style.display = 'block';
    
    // Update image preview
    const previewImage = document.getElementById('preview-image');
    previewImage.src = `data:image/jpeg;base64,${imageData}`;
    
    // Update prediction
    const predictionCard = document.getElementById('prediction-card');
    const predictionText = document.getElementById('prediction-text');
    const predictionDetails = document.getElementById('prediction-details');
    const predictionIcon = predictionCard.querySelector('.prediction-icon i');
    
    // Set prediction class and icon
    if (result.is_fake) {
        predictionCard.className = 'prediction-card fake';
        predictionIcon.className = 'fas fa-exclamation-triangle';
        predictionText.textContent = 'FAKE IMAGE DETECTED';
        predictionDetails.textContent = 'This image appears to be artificially generated or manipulated.';
    } else {
        predictionCard.className = 'prediction-card real';
        predictionIcon.className = 'fas fa-check-circle';
        predictionText.textContent = 'REAL IMAGE';
        predictionDetails.textContent = 'This image appears to be authentic and unmanipulated.';
    }
    
    // Update confidence meter
    const confidence = Math.round(result.confidence * 100);
    document.getElementById('confidence-fill').style.width = `${confidence}%`;
    document.getElementById('confidence-value').textContent = `${confidence}%`;
    
    // Update probability bars
    const realProb = Math.round(result.real_probability * 100);
    const fakeProb = Math.round(result.fake_probability * 100);
    
    document.getElementById('real-prob').textContent = `${realProb}%`;
    document.getElementById('fake-prob').textContent = `${fakeProb}%`;
    
    // Animate probability bars
    setTimeout(() => {
        document.getElementById('real-bar').style.width = `${realProb}%`;
        document.getElementById('fake-bar').style.width = `${fakeProb}%`;
    }, 100);
    
    // Show forensic analysis if available
    if (result.forensic_score !== undefined) {
        const forensicSection = document.getElementById('forensic-analysis');
        forensicSection.style.display = 'block';
        
        // Update forensic metrics
        document.getElementById('confidence-level').textContent = result.confidence_level || 'Medium';
        document.getElementById('forensic-score').textContent = (result.forensic_score * 100).toFixed(1) + '%';
        document.getElementById('model-prediction').textContent = result.model_prediction || '-';
        
        // Style confidence level
        const confidenceElement = document.getElementById('confidence-level');
        confidenceElement.className = 'metric-value confidence-' + (result.confidence_level || 'medium');
    }
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function clearResults() {
    selectedFile = null;
    
    // Reset upload area
    const uploadContent = uploadArea.querySelector('.upload-content');
    uploadContent.innerHTML = `
        <i class="fas fa-image"></i>
        <p>Choose an image file (JPG, PNG, JPEG)</p>
        <p class="file-info">Max size: 16MB</p>
    `;
    uploadArea.style.borderColor = '#cbd5e0';
    uploadArea.style.background = '#f7fafc';
    
    // Reset buttons
    analyzeBtn.disabled = true;
    clearBtn.disabled = true;
    
    // Hide results
    resultsSection.style.display = 'none';
    
    // Clear file input
    fileInput.value = '';
}


function formatModelName(modelType) {
    const names = {
        'efficientnet': 'EfficientNet',
        'xception': 'Xception',
        'hybrid_ensemble': 'Hybrid Ensemble'
    };
    return names[modelType] || modelType;
}

async function loadModelInfo() {
    try {
        const response = await fetch('/model_info');
        const data = await response.json();
        
        if (response.ok) {
            // Model loaded successfully
            console.log('Model loaded successfully');
        } else {
            console.error('Failed to load model');
        }
        
    } catch (error) {
        console.error('Error loading model info:', error);
    }
}


function showLoading(show, message = 'Analyzing image...') {
    if (show) {
        loadingOverlay.style.display = 'flex';
        const loadingText = loadingOverlay.querySelector('p');
        if (loadingText) {
            loadingText.textContent = message;
        }
    } else {
        loadingOverlay.style.display = 'none';
    }
}

function showError(message) {
    // Create a simple toast notification
    const toast = document.createElement('div');
    toast.className = 'toast toast-error';
    toast.innerHTML = `
        <i class="fas fa-exclamation-circle"></i>
        <span>${message}</span>
    `;
    
    document.body.appendChild(toast);
    
    // Show toast
    setTimeout(() => toast.classList.add('show'), 100);
    
    // Remove toast after 5 seconds
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => document.body.removeChild(toast), 300);
    }, 5000);
}

function showSuccess(message) {
    // Create a simple toast notification
    const toast = document.createElement('div');
    toast.className = 'toast toast-success';
    toast.innerHTML = `
        <i class="fas fa-check-circle"></i>
        <span>${message}</span>
    `;
    
    document.body.appendChild(toast);
    
    // Show toast
    setTimeout(() => toast.classList.add('show'), 100);
    
    // Remove toast after 3 seconds
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => document.body.removeChild(toast), 300);
    }, 3000);
}

// Add toast styles dynamically
const toastStyles = `
    .toast {
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        border-radius: 10px;
        color: white;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 10px;
        z-index: 1001;
        transform: translateX(100%);
        transition: transform 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .toast.show {
        transform: translateX(0);
    }
    
    .toast-error {
        background: linear-gradient(135deg, #f56565, #e53e3e);
    }
    
    .toast-success {
        background: linear-gradient(135deg, #48bb78, #38a169);
    }
    
    .toast i {
        font-size: 1.2rem;
    }
`;

// Inject toast styles
const styleSheet = document.createElement('style');
styleSheet.textContent = toastStyles;
document.head.appendChild(styleSheet);

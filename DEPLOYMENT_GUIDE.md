# Deepfake Detection App - Deployment Guide (Enhanced)

## 🚀 How to Run on Another System/Phone

### Method 1: Local Network Access (Recommended)

#### Step 1: Find Your Computer's IP Address
```bash
# Windows
ipconfig

# Look for "IPv4 Address" (usually starts with 192.168.x.x or 10.x.x.x)
```

#### Step 2: Modify web_app.py for Network Access
The app currently runs on `localhost:5000`. To make it accessible to other devices:

1. **Edit web_app.py** - Change the last line from:
```python
if __name__ == '__main__':
    app.run(debug=True)
```

To:
```python
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
```

#### Step 3: Run the App
```bash
python web_app.py
```

#### Step 4: Access from Other Devices
- **From another computer**: `http://YOUR_IP_ADDRESS:5000`
- **From phone**: `http://YOUR_IP_ADDRESS:5000` (in mobile browser)

### Method 2: Cloud Deployment (Advanced)

#### Option A: Heroku
1. **Create Heroku account** at heroku.com
2. **Install Heroku CLI**
3. **Create Procfile**:
```
web: python web_app.py
```
4. **Deploy**:
```bash
git init
git add .
git commit -m "Initial commit"
heroku create your-app-name
git push heroku main
```

#### Option B: Railway
1. **Create Railway account** at railway.app
2. **Connect GitHub repository**
3. **Deploy automatically**

#### Option C: Google Cloud Platform
1. **Create GCP account**
2. **Use Cloud Run** for containerized deployment
3. **Upload model files** to Cloud Storage

### Method 3: Docker Container (Recommended for Sharing)

#### Step 1: Create Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "web_app.py"]
```

#### Step 2: Build and Run
```bash
# Build image
docker build -t deepfake-detector .

# Run container
docker run -p 5000:5000 deepfake-detector
```

#### Step 3: Share Docker Image
```bash
# Save image to file
docker save deepfake-detector > deepfake-detector.tar

# On another system, load image
docker load < deepfake-detector.tar
docker run -p 5000:5000 deepfake-detector
```

### Method 4: Mobile App (Advanced)

#### Option A: PWA (Progressive Web App)
1. **Add PWA manifest** to web app
2. **Enable service worker**
3. **Install on phone** as app

#### Option B: React Native/Flutter
1. **Create mobile app** using React Native or Flutter
2. **Connect to API** endpoint
3. **Build APK/IPA** for distribution

### Method 5: Simple File Sharing

#### Step 1: Package Everything
Create a zip file containing:
- `web_app.py`
- `config.py`
- `aggressive_predictor.py`
- `src/` folder
- `models/` folder
- `templates/` folder
- `static/` folder
- `requirements.txt`
- `README.md`

#### Step 2: Share Instructions
Include these instructions for your friend:

1. **Install Python 3.9+**
2. **Install dependencies**:
```bash
pip install -r requirements.txt
```
3. **Run the app**:
```bash
python web_app.py
```
4. **Access at**: `http://localhost:5000`

### Method 6: One-Click Installer (Windows)

#### Create batch file (`install_and_run.bat`):
```batch
@echo off
echo Installing Deepfake Detection App...
pip install -r requirements.txt
echo Starting app...
python web_app.py
pause
```

### Method 7: Network Requirements

#### For Local Network Access:
- **Same WiFi network** for all devices
- **Firewall settings**: Allow Python through firewall
- **Port 5000**: Must be available and not blocked

#### Troubleshooting:
1. **Can't access from phone**:
   - Check firewall settings
   - Ensure same WiFi network
   - Try different port (5001, 8000, etc.)

2. **Connection refused**:
   - Verify IP address
   - Check if app is running
   - Test with `curl http://IP:5000`

### Method 8: Quick Setup Script

#### Create `setup.py`:
```python
import subprocess
import sys
import os

def install_requirements():
    print("Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def run_app():
    print("Starting Deepfake Detection App...")
    print("Access at: http://localhost:5000")
    print("For network access: http://YOUR_IP:5000")
    subprocess.run([sys.executable, "web_app.py"])

if __name__ == "__main__":
    install_requirements()
    run_app()
```

### Method 9: Mobile-Friendly Modifications

#### Update `templates/index.html` for mobile:
```html
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="mobile-web-app-capable" content="yes">
```

#### Add mobile CSS:
```css
@media (max-width: 768px) {
    .container {
        padding: 10px;
    }
    .upload-area {
        min-height: 200px;
    }
}
```

### Method 10: Security Considerations

#### For Production Deployment:
1. **Disable debug mode**:
```python
app.run(host='0.0.0.0', port=5000, debug=False)
```

2. **Add authentication** (optional):
```python
from flask_httpauth import HTTPBasicAuth
auth = HTTPBasicAuth()

@auth.verify_password
def verify_password(username, password):
    return username == 'admin' and password == 'your_password'
```

3. **Use HTTPS** for production

## 🎯 Recommended Approach

**For sharing with friends (easiest):**
1. Use **Method 1** (Local Network Access)
2. Share your IP address and port
3. Ensure everyone is on the same WiFi

**For wider distribution:**
1. Use **Method 3** (Docker) or **Method 2** (Cloud)
2. Create a simple setup script
3. Provide clear instructions

## 📱 Mobile Access

Once deployed, your friends can:
- **Open browser** on their phone
- **Navigate to**: `http://YOUR_IP:5000`
- **Upload images** directly from phone camera
- **Get instant results** with confidence scores

## 🔧 Troubleshooting

### Common Issues:
1. **"Connection refused"**: Check firewall and IP address
2. **"Module not found"**: Install requirements.txt
3. **"Port already in use"**: Change port number
4. **"Model not found"**: Ensure models/ folder is present

### Quick Fixes:
```bash
# Change port if 5000 is busy
python web_app.py --port 8000

# Check if app is running
netstat -an | findstr :5000

# Kill process if needed
taskkill /f /im python.exe
```

## 📞 Support

If you encounter issues:
1. Check the terminal output for errors
2. Verify all files are present
3. Test with a simple image first
4. Check network connectivity

## 🎯 Recent Improvements

### Enhanced Detection Algorithm
- **Multi-scale analysis**: Uses ensemble prediction with multiple image transforms
- **Quality-based adjustment**: Analyzes image variance and brightness for better detection
- **Lower threshold**: Reduced from 0.3 to 0.25 for more sensitive fake detection
- **Better confidence scoring**: Improved confidence calculation for clearer results

### Performance Improvements
- **Ensemble prediction**: Combines 3 different image transforms for better accuracy
- **Quality analysis**: Detects manipulation artifacts through image statistics
- **Enhanced confidence**: More accurate confidence scores for better user experience

---

**Ready to share!** Choose the method that works best for your situation.

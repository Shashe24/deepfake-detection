# 📱 How to Share Your Deepfake Detection App

## 🚀 Quick Method (Recommended)

### Step 1: Find Your IP Address
```bash
# Windows Command Prompt
ipconfig

# Look for "IPv4 Address" (usually 192.168.x.x or 10.x.x.x)
```

### Step 2: Run the App
```bash
python web_app.py
```

### Step 3: Share Access Information
Tell your friends:
- **URL**: `http://YOUR_IP_ADDRESS:5000`
- **Requirements**: Same WiFi network
- **Instructions**: Open in any web browser

## 📱 Mobile Access

Your friends can:
1. **Connect to the same WiFi** as your computer
2. **Open their phone browser**
3. **Navigate to**: `http://YOUR_IP:5000`
4. **Upload images** directly from their phone
5. **Get instant results** with confidence scores

## 🖥️ Computer Access

Other computers can:
1. **Connect to the same WiFi** network
2. **Open any web browser**
3. **Navigate to**: `http://YOUR_IP:5000`
4. **Upload images** and get results

## 🔧 Troubleshooting

### "Connection Refused"
- Check if app is running
- Verify IP address
- Check firewall settings

### "Can't Access from Phone"
- Ensure same WiFi network
- Check firewall allows Python
- Try different port (5001, 8000)

### "Module Not Found"
- Run: `pip install -r requirements.txt`
- Check Python version (3.9+)

## 🌐 Alternative Methods

### Method 1: Cloud Deployment
- Deploy to Heroku, Railway, or Google Cloud
- Get a public URL
- Share the URL with anyone

### Method 2: Docker Container
- Package app in Docker
- Share Docker image
- Run on any system with Docker

### Method 3: File Sharing
- Zip the entire project folder
- Share with friends
- They install Python and run locally

## 📋 What Your Friends Need

### Minimum Requirements:
- **Web browser** (Chrome, Safari, Firefox, Edge)
- **Same WiFi network** (for local access)
- **Internet connection** (for cloud access)

### For Local Installation:
- **Python 3.9+**
- **pip** (Python package manager)
- **All project files**

## 🎯 Quick Start Commands

### For You (Host):
```bash
# Install requirements
pip install -r requirements.txt

# Run app
python web_app.py

# Find IP address
ipconfig
```

### For Your Friends:
```bash
# Access via browser
http://YOUR_IP:5000

# Save as bookmark for easy access
```

## 📞 Support

If someone can't access:
1. **Check network**: Same WiFi?
2. **Check firewall**: Python allowed?
3. **Check IP**: Correct address?
4. **Check app**: Is it running?

## 🚀 Pro Tips

1. **Bookmark the URL** for easy access
2. **Test with a simple image** first
3. **Share the IP address** via text/email
4. **Keep the app running** while friends use it
5. **Use a simple port** like 5000 or 8000

---

**Ready to share!** Your friends can now access your deepfake detection app from any device on the same network.

# 🧠 Deepfake Image Detection System

An AI-powered web application that detects whether an image is **Real** or **Fake (AI-generated/manipulated)** using **EfficientNet-B4** and **Deep Learning**.

---

## 📌 Overview

This project uses **transfer learning** with the EfficientNet-B4 architecture to classify images as:

- ✅ **Real Image**
- ❌ **Fake / Deepfake Image**

The system is deployed via a **Flask web application** allowing real-time image analysis with confidence scores.

---

## 🚀 Features

✔ Deep learning-based classification  
✔ EfficientNet-B4 backbone  
✔ Transfer learning for better accuracy  
✔ Real-time inference (< 1 sec)  
✔ Confidence score display  
✔ Web-based UI (Flask + HTML/CSS/JS)  
✔ Robust image validation  

---

## 🏗️ System Architecture

```

User → Web Interface → Flask Server → Preprocessing → EfficientNet-B4 → Softmax → Prediction

````

---

## 🧮 Model Details

- **Architecture:** EfficientNet-B4  
- **Framework:** PyTorch  
- **Classes:** Real / Fake  
- **Loss Function:** Cross-Entropy Loss  
- **Optimizer:** AdamW  
- **Input Size:** 224 × 224  

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Accuracy | **83.33%** |
| Fake Precision | **97.06%** |
| ROC-AUC | **87.33%** |

---

## 📂 Dataset

- **Source:** Kaggle Deepfake Image Dataset  
- **Total Images:** 978  
- **Classes:** Balanced (Real & Fake)

---

## ⚙️ Requirements

- Python 3.8+
- PyTorch
- TorchVision
- Flask
- OpenCV
- Pillow
- Albumentations
- NumPy
- Scikit-learn

Install dependencies:

```bash
pip install -r requirements.txt
````

---

## ▶️ How to Run

1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/deepfake-detection.git
cd deepfake-detection
```

2️⃣ Run Flask app

```bash
python app.py
```

3️⃣ Open browser

```
http://localhost:5000
```

---

## 🖼️ Usage

1. Upload an image
2. System preprocesses input
3. Model predicts Real/Fake
4. Confidence score displayed

---

## ⚠️ Limitations

* Small dataset (978 images)
* Only image-based detection
* Fake recall lower than precision

---

## 🔮 Future Improvements

* Larger & diverse dataset
* Ensemble models
* Video deepfake detection
* Explainable AI (XAI)

---

## 🛡️ Ethical Note

This project is developed for **educational and security purposes only**.
It aims to combat misinformation and synthetic media misuse.

---

## 👨‍💻 Author

**H P Shashank**
AI & ML Department
Navkis College of Engineering, Hassan

---

## 📜 License

This project is for academic/educational use.

```

---

---

## ✅ What You Should Edit Before Uploading

Replace:

```

[https://github.com/your-username/deepfake-detection.git](https://github.com/Shashae24/
deepfake-detection)

```

with your actual repo link.

(Optional)
Add:
- Demo screenshots  
- Model accuracy graphs  
- Confusion matrix  

---

If you want, I can also generate:

✅ `requirements.txt`  
✅ GitHub badges (accuracy, Python, PyTorch)  
✅ Fancy README with icons & layout  
✅ Demo GIF preview  

Just tell me 😄
```
# deepfake-detection
"Deepfake Detection System using EfficientNet, XceptionNet, and Hybrid Ensemble models"

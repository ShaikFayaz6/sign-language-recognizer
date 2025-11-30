# Sign Language Alphabet Recognizer 🤟

A complete ASL (American Sign Language) alphabet recognition system using deep learning with TensorFlow, featuring both local Flask web application and cloud-deployed Gradio interface on Hugging Face Spaces.

**Group 4:** Fayaz Shaik, Harsha Koritala, Mallikarjun Kotha, Sai Grishyanth Magunta, Sai Kiran Dasari

---

## 🌐 Live Demo

**Hugging Face Spaces:** [https://huggingface.co/spaces/ShaikFayaz6/sign-language-recognizer](https://huggingface.co/spaces/ShaikFayaz6/sign-language-recognizer)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Image Upload** | Drag-and-drop or file picker for image classification |
| **Webcam Capture** | Real-time webcam-based sign language recognition |
| **Prediction History** | View all past predictions with timestamps and confidence scores |
| **Download Results** | Export history as CSV or download result images |
| **Cloud Deployment** | Publicly accessible via Hugging Face Spaces |
| **29 ASL Classes** | Supports A-Z letters + Space, Delete, Nothing |

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| ML Framework | TensorFlow 2.x (InceptionV3 Transfer Learning) |
| Web Framework | Flask 3.0 (Local), Gradio 4.0 (Cloud) |
| Database | SQLite (Prediction History) |
| Image Processing | OpenCV, Pillow |
| Cloud Hosting | Hugging Face Spaces |
| Language | Python 3.10+ |

---

## 📁 Project Structure

```
sign-language-alphabet-recognizer/
├── app_gradio.py           # Gradio cloud application (Hugging Face)
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── Phase3_Report.md        # Architecture extension documentation
├── Phase4_Report.md        # UI development and integration report
├── Final_Report.md         # Complete project summary
│
├── logs/
│   ├── output_graph_improved.pb      # Trained TensorFlow model (83.6 MB)
│   └── output_labels_improved.txt    # 29 class labels
│
├── webapp/                 # Flask Web Application
│   ├── app.py              # Main Flask server
│   ├── ml_inference.py     # ML model integration
│   ├── database.py         # SQLite database operations
│   ├── templates/          # HTML templates (base, index, upload, webcam, history)
│   └── static/css/         # Custom styling
│
├── dataset/                # Training data (A-Z folders, ~1GB)
│
├── train.py                # Model training script
├── classify.py             # Command-line image classification
└── classify_webcam.py      # Command-line webcam classification
```

---

## 🚀 Quick Start

### Option 1: Use Cloud App (No Installation)
Visit: [https://huggingface.co/spaces/ShaikFayaz6/sign-language-recognizer](https://huggingface.co/spaces/ShaikFayaz6/sign-language-recognizer)

### Option 2: Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/ShaikFayaz6/sign-language-recognizer.git
cd sign-language-recognizer

# 2. Create virtual environment
python -m venv venv_tf2

# 3. Activate virtual environment
# Windows PowerShell:
.\venv_tf2\Scripts\Activate.ps1
# Windows CMD:
.\venv_tf2\Scripts\activate.bat
# Linux/Mac:
source venv_tf2/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run Flask application
python webapp/app.py

# 6. Open browser
# Navigate to http://localhost:5000
```

---

## 📖 How It Works

### 1. Image Classification Flow
```
User uploads image → Image validation → Preprocessing (resize, normalize) 
→ TensorFlow inference → Top-5 predictions → Save to history → Display result
```

### 2. Webcam Classification Flow
```
Start webcam → Capture frame → Base64 encoding → Send to server 
→ Preprocessing → ML inference → Return predictions → Display in real-time
```

### 3. Model Architecture
- **Base Model:** InceptionV3 (pre-trained on ImageNet)
- **Transfer Learning:** Fine-tuned on ASL alphabet dataset
- **Training Steps:** 10,000
- **Input Size:** 299x299 pixels
- **Output Classes:** 29 (A-Z + Space + Delete + Nothing)

---

## 🎯 Supported Gestures

| Letters | Special |
|---------|---------|
| A B C D E F G H I J K L M N O P Q R S T U V W X Y Z | Space, Delete, Nothing |

---

## 📊 API Endpoints (Flask)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page |
| `/upload` | GET | Image upload page |
| `/webcam` | GET | Webcam capture page |
| `/history` | GET | Prediction history |
| `/api/predict` | POST | Classify image (JSON response) |
| `/api/export_csv` | GET | Download history as CSV |

---

## 🔧 Original CLI Commands

### Training (if you have the dataset)
```bash
python train.py \
  --bottleneck_dir=logs/bottlenecks \
  --how_many_training_steps=10000 \
  --model_dir=inception \
  --summaries_dir=logs/training_summaries/basic \
  --output_graph=logs/output_graph_improved.pb \
  --output_labels=logs/output_labels_improved.txt \
  --image_dir=./dataset
```

### Command-Line Classification
```bash
python classify.py path/to/image.jpg
```

### Webcam Demo (CLI)
```bash
python classify_webcam.py
```

---

## 📚 Documentation

- **Phase 3 Report:** Architecture extension, use cases, quality attributes
- **Phase 4 Report:** UI development, communication diagrams, deployment
- **Final Report:** Complete project summary with all changes documented

---

## 👥 Team

**Group 4 - Software Development for AI**
- Fayaz Shaik
- Harsha Koritala
- Mallikarjun Kotha
- Sai Grishyanth Magunta
- Sai Kiran Dasari

---

## 🙏 Acknowledgments

- Original framework: [Image Classification with TensorFlow](https://github.com/xuetsing/image-classification-tensorflow) by xuetsing
- InceptionV3 model: Google's TensorFlow team
- Hugging Face for cloud hosting

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

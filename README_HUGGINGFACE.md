# 🤟 Sign Language Alphabet Recognizer

**Phase 3: Extending the Architecture of an ML-based System**

A machine learning web application that recognizes American Sign Language (ASL) alphabet gestures using deep learning.

## 🎯 Features

- **Real-time Sign Recognition:** Upload images or use webcam for instant predictions
- **29 Classes:** Recognizes A-Z letters plus Space, Del, and Nothing
- **High Accuracy:** Improved model with 10,000 training steps
- **User-Friendly Interface:** Clean Gradio UI with confidence scores
- **Cloud-Ready:** Deployed on Hugging Face Spaces

## 🚀 Quick Start

### Local Installation

```bash
# Clone the repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/sign-language-recognizer
cd sign-language-recognizer

# Install dependencies
pip install -r requirements.txt

# Run the app
python app_gradio.py
```

The app will be available at `http://localhost:7860`

### Using the App

1. **Upload an Image:** Click "Upload Sign Language Image" or use webcam
2. **Get Prediction:** The model automatically predicts the sign language letter
3. **View Confidence:** See confidence scores for top predictions

## 📊 Model Architecture

- **Base Model:** InceptionV3 (pre-trained on ImageNet)
- **Framework:** TensorFlow 2.11.0
- **Training:** Transfer learning with 10,000 steps
- **Dataset:** 30,000+ images across 29 classes
- **Input Size:** 299x299 pixels

## 🎓 Phase 3 Architecture Extensions

This project demonstrates key architectural patterns for ML systems:

### Quality Attributes Implemented
- **QA1 - Performance:** Model caching in memory (<2s response time)
- **QA2 - Scalability:** Stateless design supporting concurrent users
- **QA3 - Reliability:** Exception handling with user-friendly error messages
- **QA5 - Security:** Input validation and file type checking

### New Use Cases
- **UC4 - View Prediction History:** SQLite database stores past predictions
- **UC7 - Download Results:** Export predictions as text files

### Architecture Patterns
- **Layered Architecture:** Separation of UI, business logic, and ML model
- **Caching Strategy:** In-memory model loading for performance
- **Error Handling:** Graceful degradation with detailed logging

## 📁 Project Structure

```
sign-language-recognizer/
├── app_gradio.py           # Gradio web interface (Hugging Face)
├── webapp/
│   ├── app.py             # Flask web application
│   ├── ml_inference.py    # ML model inference engine
│   ├── database.py        # SQLite database handler
│   └── templates/         # HTML templates
├── logs/
│   ├── output_graph_improved.pb    # Improved trained model (10,000 steps)
│   └── output_labels_improved.txt  # Class labels
├── dataset/               # Training images (29 classes)
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🛠️ Tech Stack

- **Backend:** Python 3.8, TensorFlow 2.11.0
- **Web Framework:** Gradio (Hugging Face) / Flask (local)
- **ML Framework:** TensorFlow with InceptionV3
- **Database:** SQLite
- **Image Processing:** OpenCV, Pillow
- **Deployment:** Hugging Face Spaces

## 📈 Performance Metrics

- **Accuracy:** Improved with 10,000 training steps
- **Response Time:** < 2 seconds per prediction
- **Model Size:** ~84 MB
- **Supported Classes:** 29
- **Training Steps:** 10,000

## 👥 Team

**Group No. 4:**
- Fayaz Shaik
- Harsha Koritala
- Mallikarjun Kotha
- Sai Grishyanth Magunta
- Sai Kiran Dasari

## 📝 License

This project is created for academic purposes as part of the Software Development for AI course.

## 🔗 Links

- **Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/YOUR_USERNAME/sign-language-recognizer)
- **GitHub Repository:** [Sign Language Recognizer](https://github.com/ShaikFayaz6/sign-language-recognizer)
- **Documentation:** [Phase 3 Report](Phase3_Report.md)

## 🙏 Acknowledgments

- InceptionV3 model from TensorFlow Hub
- ASL dataset from Kaggle
- Hugging Face for free hosting

---

💡 **Tip:** For best results, use clear images with good lighting and a plain background!

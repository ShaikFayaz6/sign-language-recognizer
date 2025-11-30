# Final Report: Sign Language Alphabet Recognizer

**Group 4:** Fayaz Shaik, Harsha Koritala, Mallikarjun Kotha, Sai Grishyanth Magunta, Sai Kiran Dasari  
**Course:** Software Development for AI  
**Project Completion Date:** November 2024

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Files Created](#2-files-created)
3. [Files Modified](#3-files-modified)
4. [Detailed Changes Summary](#4-detailed-changes-summary)
5. [Deployment Summary](#5-deployment-summary)
6. [Final Project State](#6-final-project-state)

---

## 1. Project Overview

This report documents all the changes made to transform the original CLI-based Sign Language Recognizer into a full-stack web application with cloud deployment capabilities.

### Original Project State
- CLI-only application for training and classification
- TensorFlow 1.x based
- Python 3.5 compatible
- No web interface
- No database for history
- No cloud deployment

### Final Project State
- Full-stack web application (Flask + Gradio)
- TensorFlow 2.x compatible
- Python 3.10+ support
- Web-based image upload and webcam classification
- SQLite database for prediction history
- Cloud deployed on Hugging Face Spaces

---

## 2. Files Created

### 2.1 `app_gradio.py` (New File)
**Purpose:** Cloud-deployable Gradio web application for Hugging Face Spaces

**What it does:**
- Provides a modern web UI using Gradio 4.0
- Supports image upload for classification
- Supports webcam capture for real-time recognition
- Maintains prediction history in SQLite database
- Provides CSV export functionality
- Displays top-5 predictions with confidence scores

**Key Components:**
```python
# Database operations for history
def init_db(), save_prediction(), get_history(), export_csv()

# ML model loading (TensorFlow 2.x compatible)
def load_graph(), load_labels()

# Classification logic
def classify_image(image_path)

# Gradio Interface with Blocks API
gr.Blocks() with tabs for Upload, Webcam, History
```

**Why it helps:**
- Enables cloud deployment without server management
- Provides accessible web interface for end users
- Persistent history storage across sessions

---

### 2.2 `webapp/` Directory (New Folder)
Complete Flask-based web application for local deployment.

#### 2.2.1 `webapp/app.py`
**Purpose:** Main Flask server with routing and API endpoints

**What it does:**
- Serves web pages (home, upload, webcam, history)
- Handles file uploads and image processing
- Provides REST API for predictions
- Handles webcam Base64 image processing
- Manages CSV export functionality

**Key Routes:**
| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Home page |
| `/upload` | GET/POST | Image upload interface |
| `/webcam` | GET | Webcam capture interface |
| `/history` | GET | View prediction history |
| `/api/predict` | POST | REST API for classification |
| `/api/export_csv` | GET | Download history as CSV |

**Why it helps:**
- Enables local deployment with full control
- RESTful API for programmatic access
- Clean separation of concerns

---

#### 2.2.2 `webapp/ml_inference.py`
**Purpose:** TensorFlow model integration layer

**What it does:**
- Loads TensorFlow 2.x graph from `.pb` file
- Loads class labels from text file
- Preprocesses images (resize, normalize)
- Runs inference and returns predictions
- Handles TensorFlow 1.x to 2.x compatibility

**Key Functions:**
```python
load_graph()     # Loads frozen model
load_labels()    # Loads 29 class labels
preprocess_image()  # Resize to 299x299, normalize
classify_image()    # Run inference, return top-5
```

**Why it helps:**
- Abstracts ML complexity from web layer
- Handles TensorFlow version differences
- Enables consistent predictions across interfaces

---

#### 2.2.3 `webapp/database.py`
**Purpose:** SQLite database operations for prediction history

**What it does:**
- Creates SQLite database (`predictions.db`)
- Stores predictions with timestamps
- Retrieves history for display
- Supports CSV export

**Database Schema:**
```sql
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    image_name TEXT,
    predicted_class TEXT,
    confidence REAL,
    top_5_predictions TEXT
)
```

**Why it helps:**
- Persists prediction history across sessions
- Enables history viewing and export features
- Lightweight, no external database required

---

#### 2.2.4 `webapp/templates/` (5 HTML Files)

| File | Purpose |
|------|---------|
| `base.html` | Base template with navigation, Bootstrap styling |
| `index.html` | Home page with feature overview |
| `upload.html` | Drag-and-drop image upload interface |
| `webcam.html` | Real-time webcam classification with JavaScript |
| `history.html` | Prediction history table with export button |

**Why it helps:**
- Consistent look and feel across pages
- Responsive design for various devices
- Interactive webcam interface with real-time feedback

---

#### 2.2.5 `webapp/static/css/style.css`
**Purpose:** Custom styling for Flask application

**What it does:**
- Extends Bootstrap styling
- Provides drag-and-drop visual feedback
- Styles prediction result cards
- Custom webcam interface styling

---

### 2.3 `Phase3_Report.md` (New File)
**Purpose:** Architecture extension documentation

**Contents:**
- Extended use cases (UC-04: View History, UC-07: Download Results)
- Quality attributes (Portability, Security, Testability, Usability, Scalability)
- Updated C4 diagrams (Context, Container, Component)
- Layered architecture design
- Technology stack decisions

**Why it helps:**
- Documents architectural decisions
- Provides traceability from requirements to implementation
- Serves as reference for future development

---

### 2.4 `Phase4_Report.md` (New File)
**Purpose:** UI development and integration documentation

**Contents:**
- PlantUML communication diagrams for all use cases:
  - UC-01: Classify Image (Upload)
  - UC-02: Classify from Webcam
  - UC-04: View History
  - UC-05: Cloud Deployment
  - UC-07: Download Results
- UI component specifications
- Integration testing results
- Deployment procedures

**Why it helps:**
- Documents message flows between components
- Provides implementation reference
- Records testing and deployment procedures

---

### 2.5 Deployment Configuration Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Container configuration for Docker deployment |
| `Procfile` | Heroku deployment configuration |
| `railway.json` | Railway deployment configuration |
| `render.yaml` | Render deployment configuration |
| `runtime.txt` | Python version specification |
| `README_HUGGINGFACE.md` | Hugging Face Spaces documentation |

**Why they help:**
- Enable deployment to multiple cloud platforms
- Document deployment procedures
- Provide configuration for CI/CD pipelines

---

## 3. Files Modified

### 3.1 `requirements.txt`

**Original Content:**
```
opencv-python
tensorflow
matplotlib
numpy
```

**Updated Content:**
```
tensorflow==2.11.0
matplotlib>=3.5.0
numpy>=1.21.0,<1.24.0
opencv-python-headless>=4.5.0,<4.8.0
flask>=2.3.0
werkzeug>=2.3.0
gunicorn>=20.1.0
python-dotenv>=0.19.0
gradio==4.37.2
Pillow>=9.0.0
```

**Changes Made:**
| Change | Reason |
|--------|--------|
| `tensorflow` → `tensorflow==2.11.0` | Specific version for compatibility |
| Added `flask>=2.3.0` | Web framework for local app |
| Added `gradio==4.37.2` | Cloud app framework |
| Added `gunicorn>=20.1.0` | Production WSGI server |
| Added `Pillow>=9.0.0` | Image processing |
| `opencv-python` → `opencv-python-headless` | Server compatibility (no GUI) |
| Added version constraints | Prevent breaking changes |

**Why it helps:**
- Ensures reproducible environment
- Adds web framework dependencies
- Prevents version conflicts

---

### 3.2 `train.py`

**Changes Made:**
- Updated TensorFlow 1.x imports to TensorFlow 2.x compatibility
- Changed `tf.gfile` to `tf.io.gfile`
- Changed `tf.logging` to `tf.compat.v1.logging`
- Added `tf.compat.v1.disable_eager_execution()` for graph mode
- Updated deprecated function calls

**Example Change:**
```python
# Before (TensorFlow 1.x)
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

# After (TensorFlow 2.x compatible)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
tf.compat.v1.disable_eager_execution()
```

**Why it helps:**
- Enables training with TensorFlow 2.x
- Maintains backward compatibility with trained models

---

### 3.3 `classify.py`

**Changes Made:**
- Updated TensorFlow 1.x imports to TensorFlow 2.x
- Changed graph loading to use `tf.compat.v1.GraphDef()`
- Updated session handling for TF2 compatibility
- Changed deprecated `tf.gfile` calls

**Key Updates:**
```python
# Before
with tf.gfile.GFile(model_path, 'rb') as f:
    graph_def = tf.GraphDef()
    
# After
with tf.io.gfile.GFile(model_path, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
```

**Why it helps:**
- CLI classification works with TensorFlow 2.x
- Consistent model loading across all interfaces

---

### 3.4 `classify_webcam.py`

**Changes Made:**
- Same TensorFlow 2.x compatibility updates as `classify.py`
- Updated OpenCV import handling
- Added error handling for webcam access

**Why it helps:**
- Webcam CLI works with updated dependencies
- Better error handling for missing camera

---

### 3.5 `README.md`

**Original Content:**
- Referenced Python 3.5
- Only CLI usage instructions
- No web interface documentation
- No cloud deployment information

**Updated Content:**
- References Python 3.10+
- Documents Flask and Gradio interfaces
- Includes cloud deployment links
- Complete API documentation
- Updated installation instructions
- Project structure documentation

**Why it helps:**
- Accurate documentation for current state
- Helps new users get started quickly
- Documents all features and endpoints

---

## 4. Detailed Changes Summary

### 4.1 TensorFlow Migration (1.x → 2.x)

| Component | TF 1.x Code | TF 2.x Code |
|-----------|-------------|-------------|
| Logging | `tf.logging` | `tf.compat.v1.logging` |
| File IO | `tf.gfile.GFile` | `tf.io.gfile.GFile` |
| Graph | `tf.GraphDef()` | `tf.compat.v1.GraphDef()` |
| Session | `tf.Session()` | `tf.compat.v1.Session()` |
| Flags | `tf.app.flags` | `tf.compat.v1.app.flags` |

### 4.2 Web Interface Implementation

| Layer | Technology | Files |
|-------|------------|-------|
| Presentation | HTML/CSS/JS | `webapp/templates/`, `webapp/static/` |
| Application | Flask | `webapp/app.py` |
| ML Integration | TensorFlow | `webapp/ml_inference.py` |
| Data | SQLite | `webapp/database.py` |
| Cloud UI | Gradio | `app_gradio.py` |

### 4.3 Database Schema

```sql
-- Created in database.py and app_gradio.py
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    image_name TEXT,
    predicted_class TEXT NOT NULL,
    confidence REAL NOT NULL,
    top_5_predictions TEXT
);
```

---

## 5. Deployment Summary

### 5.1 Local Deployment (Flask)
```bash
cd webapp
python app.py
# Runs on http://localhost:5000
```

### 5.2 Cloud Deployment (Hugging Face Spaces)
- **URL:** https://huggingface.co/spaces/ShaikFayaz6/sign-language-recognizer
- **Framework:** Gradio 4.0
- **Hardware:** CPU Basic (Free Tier)
- **Files Synced:** `app_gradio.py`, `requirements.txt`, `logs/`

### 5.3 Deployment Files Purpose

| File | Platform | Purpose |
|------|----------|---------|
| `Dockerfile` | Docker/Any | Container build instructions |
| `Procfile` | Heroku | Process definition |
| `railway.json` | Railway | Build and deploy config |
| `render.yaml` | Render | Service configuration |
| `runtime.txt` | Various | Python version (3.10) |

---

## 6. Final Project State

### 6.1 Complete File Inventory

**Created Files (11):**
1. `app_gradio.py` - Gradio cloud application
2. `webapp/app.py` - Flask server
3. `webapp/ml_inference.py` - ML integration
4. `webapp/database.py` - Database operations
5. `webapp/templates/base.html` - Base template
6. `webapp/templates/index.html` - Home page
7. `webapp/templates/upload.html` - Upload page
8. `webapp/templates/webcam.html` - Webcam page
9. `webapp/templates/history.html` - History page
10. `Phase3_Report.md` - Architecture documentation
11. `Phase4_Report.md` - UI documentation

**Modified Files (5):**
1. `requirements.txt` - Added web dependencies
2. `train.py` - TensorFlow 2.x compatibility
3. `classify.py` - TensorFlow 2.x compatibility
4. `classify_webcam.py` - TensorFlow 2.x compatibility
5. `README.md` - Updated documentation

### 6.2 Technology Stack Summary

| Component | Original | Final |
|-----------|----------|-------|
| Python | 3.5 | 3.10+ |
| TensorFlow | 1.x | 2.11.0 |
| Interface | CLI only | CLI + Flask + Gradio |
| Database | None | SQLite |
| Deployment | Local | Local + Cloud (HF Spaces) |

### 6.3 Feature Comparison

| Feature | Original | Final |
|---------|----------|-------|
| Image Classification | ✅ CLI | ✅ CLI + Web + Cloud |
| Webcam Classification | ✅ CLI | ✅ CLI + Web + Cloud |
| Prediction History | ❌ | ✅ SQLite + UI |
| CSV Export | ❌ | ✅ Web download |
| Cloud Access | ❌ | ✅ Hugging Face Spaces |
| REST API | ❌ | ✅ Flask API |

---

## Conclusion

This project successfully transformed a CLI-based TensorFlow 1.x sign language recognizer into a modern, full-stack web application with cloud deployment. The key achievements include:

1. **TensorFlow 2.x Migration:** Updated all ML code for compatibility
2. **Web Interface:** Built complete Flask application with modern UI
3. **Cloud Deployment:** Deployed to Hugging Face Spaces for public access
4. **History Feature:** Added SQLite database for prediction tracking
5. **Export Capability:** Enabled CSV download of results
6. **Documentation:** Created comprehensive Phase 3, Phase 4, and Final reports

**Live Demo:** https://huggingface.co/spaces/ShaikFayaz6/sign-language-recognizer

---

*Report prepared by Group 4 - Software Development for AI*

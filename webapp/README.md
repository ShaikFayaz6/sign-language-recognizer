# Sign Language Alphabet Recognizer - Web Application

## Phase 3 Implementation

This is the Flask-based web interface for the Sign Language Alphabet Recognizer project. It provides a user-friendly interface for uploading images, using webcam for real-time recognition, viewing prediction history, and downloading results.

## Features

✅ **Homepage** - Overview and navigation  
✅ **Upload Image** - Drag & drop or browse to upload sign language images  
✅ **Webcam Recognition** - Real-time sign language recognition using webcam  
✅ **Prediction History** - View past predictions with timestamps and confidence scores  
✅ **Download Results** - Export prediction results as text files  
✅ **Responsive Design** - Works on desktop, tablet, and mobile devices  

## Project Structure

```
webapp/
├── app.py                  # Main Flask application
├── ml_inference.py         # ML model loading and inference
├── database.py             # SQLite database operations
├── predictions.db          # SQLite database (auto-created)
├── temp_uploads/           # Temporary upload folder (auto-created)
├── templates/              # HTML templates
│   ├── base.html          # Base template with navbar/footer
│   ├── index.html         # Homepage
│   ├── upload.html        # Upload image page
│   ├── webcam.html        # Webcam recognition page
│   └── history.html       # History page
└── static/                 # Static assets
    └── css/
        └── style.css      # Main stylesheet
```

## Installation

### 1. Install Dependencies

Make sure you're in the project root directory and have activated the virtual environment:

```powershell
# Activate virtual environment
.\venv_tf2\Scripts\Activate.ps1

# Install Flask dependencies
pip install flask werkzeug
```

Or install from requirements.txt:

```powershell
pip install -r requirements.txt
```

### 2. Navigate to webapp folder

```powershell
cd webapp
```

### 3. Run the Application

```powershell
python app.py
```

The application will start on `http://localhost:5000`

## Usage

### 1. Homepage
- Navigate to `http://localhost:5000`
- View features and choose an option (Upload or Webcam)

### 2. Upload Image
- Go to **Upload** page
- Drag & drop an image or click to browse
- Supported formats: JPG, PNG, GIF, BMP (Max 10MB)
- Click **Classify Image**
- View prediction with confidence score
- Download results as text file

### 3. Webcam Recognition
- Go to **Webcam** page
- Click **Start Webcam** (allow camera permissions)
- Position your hand making a sign language gesture inside the frame
- Click **Capture & Classify**
- View real-time prediction results
- Download results if needed

### 4. View History
- Go to **History** page
- View last 20 predictions
- See predicted letters, confidence scores, and timestamps
- Clear history if needed

## API Endpoints

The application provides the following REST API endpoints:

### POST `/api/predict`
Classify an uploaded image or base64-encoded webcam snapshot

**Request:**
- `multipart/form-data` with `image` file, OR
- Form data with `image_data` (base64 encoded)

**Response:**
```json
{
  "success": true,
  "prediction": "A",
  "confidence": 0.95,
  "top_predictions": [
    {"label": "A", "confidence": 0.95},
    {"label": "S", "confidence": 0.03},
    ...
  ]
}
```

### POST `/api/download_result`
Download prediction result as text file

**Request:**
```json
{
  "prediction": "A",
  "confidence": 0.95
}
```

**Response:** Text file download

### GET `/api/history`
Get recent predictions (used internally)

### POST `/api/clear_history`
Clear all prediction history

### GET `/health`
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-11-17T10:30:00"
}
```

## Architecture

### Components

1. **Flask Web UI (app.py)**
   - Renders HTML templates
   - Handles file uploads
   - Routes requests to appropriate pages

2. **ML Inference Engine (ml_inference.py)**
   - Loads TensorFlow model at startup (cached in memory)
   - Performs predictions on images
   - Returns top-5 predictions with confidence scores

3. **Database Manager (database.py)**
   - SQLite database for prediction history
   - Stores: predicted_letter, confidence, timestamp
   - Provides CRUD operations

4. **Frontend (HTML/CSS/JavaScript)**
   - Responsive UI using vanilla JavaScript and jQuery
   - AJAX calls for asynchronous predictions
   - Drag & drop file upload
   - Webcam access via MediaDevices API

### Quality Attributes Addressed

- **QA1 (Performance)**: Model loaded once at startup and cached in memory
- **QA2 (Scalability)**: Stateless REST API, supports concurrent requests
- **QA3 (Availability)**: Try-catch error handling, health check endpoint
- **QA5 (Security)**: File type validation, size limits (10MB max)
- **UC4 (History)**: SQLite database stores last 20 predictions
- **UC7 (Download)**: Text file generation for results

## Technologies

- **Backend**: Python 3.8+, Flask 2.3+
- **ML Framework**: TensorFlow 2.11
- **Image Processing**: OpenCV, NumPy
- **Database**: SQLite3
- **Frontend**: HTML5, CSS3, JavaScript (jQuery)
- **Styling**: Custom responsive CSS

## Troubleshooting

### Model not loading
- Ensure `logs/output_graph.pb` and `logs/output_labels.txt` exist in parent directory
- Check file paths in `ml_inference.py`

### Webcam not working
- Grant camera permissions in browser
- Use HTTPS or localhost (required by browsers)
- Check if another application is using the webcam

### Port already in use
- Change port in `app.py`: `app.run(port=5001)`
- Or kill the process using port 5000

### Database errors
- Delete `predictions.db` and restart (auto-recreates)
- Check file permissions

## Development

### Enable Debug Mode
Debug mode is already enabled in `app.py`:
```python
app.run(debug=True)
```

### Add New Routes
Edit `app.py` and add new routes:
```python
@app.route('/new_page')
def new_page():
    return render_template('new_page.html')
```

### Modify Styles
Edit `static/css/style.css` for UI changes

## Future Enhancements

- [ ] Add user authentication
- [ ] Implement batch image processing
- [ ] Add real-time continuous webcam recognition
- [ ] Export history as CSV
- [ ] Add model retraining interface
- [ ] Implement API rate limiting
- [ ] Add dark mode toggle
- [ ] Mobile app wrapper (PWA)

## License

MIT License - See parent project for details

## Team

**Group 4 - Phase 3 Project**
- Fayaz Shaik
- Harsha Koritala
- Mallikarjun Kotha
- Sai Grishyanth Magunta
- Sai Kiran Dasari

---

**🤟 Happy Sign Language Recognition!**

"""
Gradio App for Sign Language Alphabet Recognizer
Adapted for Hugging Face Spaces deployment
Styled to match Flask webapp design
"""

import gradio as gr
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import os
import sys
import warnings
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings

tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

print("Starting app initialization...", file=sys.stderr)

# Load model and labels
print("Loading ML model...")
model_path = "logs/output_graph_improved.pb"
labels_path = "logs/output_labels_improved.txt"

# Load labels
with tf.io.gfile.GFile(labels_path, 'r') as f:
    label_lines = [line.rstrip() for line in f]

# Load frozen graph
with tf.io.gfile.GFile(model_path, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

# Create session
sess = tf.compat.v1.Session()
softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
print(f"✓ Model loaded successfully! Classes: {len(label_lines)}")

# History storage (in-memory for Hugging Face Spaces)
prediction_history = []

# Custom CSS matching Flask webapp style - Clean and Aligned
custom_css = """
/* ============================================
   Sign Language Recognizer - Gradio Custom CSS
   Clean, Aligned, Professional Design
   ============================================ */

/* Reset and Base */
* {
    box-sizing: border-box;
}

/* Main Container */
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    background: #f5f7fa !important;
    max-width: 1100px !important;
    margin: 0 auto !important;
    padding: 1rem !important;
}

/* Hide default Gradio footer */
footer {
    display: none !important;
}

/* ===== HEADER SECTION ===== */
.app-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important;
    color: #ffffff !important;
    padding: 1.5rem 2rem !important;
    border-radius: 12px !important;
    margin-bottom: 1rem !important;
    text-align: center !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
}

.app-header h1 {
    margin: 0 !important;
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    color: #ffffff !important;
}

.app-header .subtitle {
    margin-top: 0.5rem !important;
    font-size: 0.9rem !important;
    color: #4CAF50 !important;
    font-weight: 600 !important;
}

/* ===== HERO SECTION ===== */
.hero-banner {
    background: linear-gradient(135deg, #4CAF50 0%, #2196F3 100%) !important;
    color: #ffffff !important;
    padding: 1.5rem 2rem !important;
    border-radius: 12px !important;
    margin-bottom: 1rem !important;
    text-align: center !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.15) !important;
}

.hero-banner h2 {
    margin: 0 0 0.5rem 0 !important;
    font-size: 1.4rem !important;
    font-weight: 600 !important;
    color: #ffffff !important;
}

.hero-banner p {
    margin: 0 !important;
    font-size: 0.95rem !important;
    color: #ffffff !important;
}

/* Tech Badges */
.tech-badges {
    display: flex !important;
    gap: 0.5rem !important;
    justify-content: center !important;
    flex-wrap: wrap !important;
    margin-top: 1rem !important;
}

.tech-badge {
    background: rgba(255,255,255,0.25) !important;
    color: #ffffff !important;
    padding: 0.35rem 0.75rem !important;
    border-radius: 20px !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    backdrop-filter: blur(5px) !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
}

/* ===== TAB STYLING ===== */
.tabs > .tab-nav {
    background: white !important;
    border-radius: 12px 12px 0 0 !important;
    padding: 0.5rem !important;
    gap: 0.5rem !important;
}

.tabs > .tab-nav > button {
    border-radius: 8px !important;
    padding: 0.75rem 1.25rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    transition: all 0.2s ease !important;
}

.tabs > .tab-nav > button.selected {
    background: linear-gradient(135deg, #4CAF50, #45a049) !important;
    color: white !important;
}

.tabitem {
    background: white !important;
    border-radius: 0 0 12px 12px !important;
    padding: 1.5rem !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08) !important;
}

/* ===== SECTION CARD ===== */
.section-card {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 10px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    text-align: center;
}

.section-card .icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
}

.section-card h3 {
    margin: 0 0 0.25rem 0;
    color: #4CAF50;
    font-size: 1.1rem;
    font-weight: 600;
}

.section-card p {
    margin: 0;
    color: #666;
    font-size: 0.9rem;
}

/* ===== RESULT DISPLAY ===== */
.result-container {
    background: white;
    border: 2px solid #4CAF50;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    min-height: 200px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}

.result-container h3 {
    margin: 0 0 0.5rem 0;
    color: #4CAF50;
    font-size: 1rem;
    font-weight: 600;
}

.result-letter {
    font-size: 4.5rem;
    font-weight: 700;
    color: #4CAF50;
    line-height: 1;
    margin: 0.5rem 0;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}

.result-confidence {
    font-size: 1.1rem;
    color: #666;
}

.result-confidence strong {
    color: #4CAF50;
    font-size: 1.2rem;
}

/* ===== PREDICTIONS LIST ===== */
.predictions-container {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 1rem;
}

.predictions-container h4 {
    margin: 0 0 0.75rem 0;
    color: #333;
    font-size: 0.95rem;
    font-weight: 600;
}

.pred-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.5rem;
    padding: 0.5rem;
    background: white;
    border-radius: 8px;
}

.pred-row:last-child {
    margin-bottom: 0;
}

.pred-label {
    min-width: 50px;
    font-weight: 600;
    font-size: 0.9rem;
    color: #333;
}

.pred-bar-bg {
    flex: 1;
    height: 22px;
    background: #e9ecef;
    border-radius: 11px;
    overflow: hidden;
}

.pred-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #4CAF50, #2196F3);
    border-radius: 11px;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding-right: 8px;
    color: white;
    font-size: 0.75rem;
    font-weight: 600;
    min-width: 40px;
}

/* ===== EMPTY STATE ===== */
.empty-state {
    text-align: center;
    padding: 2rem;
    color: #888;
}

.empty-state .icon {
    font-size: 3rem;
    margin-bottom: 0.75rem;
    opacity: 0.7;
}

.empty-state h3 {
    margin: 0 0 0.5rem 0;
    color: #666;
    font-size: 1.1rem;
}

.empty-state p {
    margin: 0;
    font-size: 0.9rem;
}

/* ===== INSTRUCTIONS BOX ===== */
.instructions-card {
    background: white;
    border: 1px solid #e9ecef;
    border-radius: 10px;
    padding: 1.25rem;
    margin-top: 1rem;
}

.instructions-card h4 {
    margin: 0 0 0.75rem 0;
    color: #4CAF50;
    font-size: 1rem;
    font-weight: 600;
}

.instructions-card ol, .instructions-card ul {
    margin: 0 0 0 1.25rem;
    padding: 0;
    line-height: 1.7;
    font-size: 0.9rem;
    color: #555;
}

.tip-box {
    background: #fff8e1;
    border-left: 4px solid #ffc107;
    padding: 0.75rem 1rem;
    border-radius: 0 8px 8px 0;
    margin-top: 0.75rem;
    font-size: 0.85rem;
}

/* ===== HISTORY TABLE ===== */
.history-wrapper {
    overflow-x: auto;
}

.history-table {
    width: 100%;
    border-collapse: collapse;
    background: white;
    border-radius: 10px;
    overflow: hidden;
    font-size: 0.9rem;
}

.history-table th {
    background: #f8f9fa;
    color: #4CAF50;
    font-weight: 600;
    padding: 0.875rem 1rem;
    text-align: left;
    border-bottom: 2px solid #e9ecef;
}

.history-table td {
    padding: 0.75rem 1rem;
    border-bottom: 1px solid #f0f0f0;
    vertical-align: middle;
}

.history-table tr:hover {
    background: #f8f9fa;
}

.letter-badge {
    display: inline-block;
    width: 40px;
    height: 40px;
    line-height: 40px;
    text-align: center;
    background: linear-gradient(135deg, #4CAF50, #45a049);
    color: white;
    border-radius: 8px;
    font-weight: 700;
    font-size: 1.1rem;
}

.conf-bar {
    height: 24px;
    background: linear-gradient(90deg, #4CAF50, #2196F3);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 600;
    font-size: 0.8rem;
    min-width: 50px;
}

/* ===== MODEL INFO GRID ===== */
.info-section {
    background: white;
    border: 1px solid #e9ecef;
    border-radius: 10px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}

.info-section h4 {
    margin: 0 0 1rem 0;
    color: #4CAF50;
    font-size: 1rem;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.info-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.75rem;
}

@media (max-width: 600px) {
    .info-grid {
        grid-template-columns: 1fr;
    }
}

.info-card {
    background: #f8f9fa;
    padding: 0.875rem;
    border-radius: 8px;
    text-align: center;
}

.info-card strong {
    display: block;
    color: #4CAF50;
    font-size: 0.85rem;
    margin-bottom: 0.25rem;
}

.info-card span {
    color: #333;
    font-size: 0.9rem;
}

/* Letter Grid for About Page */
.letter-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    justify-content: center;
}

.letter-chip {
    width: 32px;
    height: 32px;
    line-height: 32px;
    text-align: center;
    background: #4CAF50;
    color: white;
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.85rem;
}

.letter-chip.blue { background: #2196F3; }
.letter-chip.red { background: #f44336; }
.letter-chip.gray { background: #757575; }

/* Feature List */
.feature-list {
    list-style: none;
    margin: 0;
    padding: 0;
}

.feature-list li {
    padding: 0.5rem 0;
    padding-left: 1.75rem;
    position: relative;
    font-size: 0.9rem;
    color: #555;
}

.feature-list li::before {
    content: "✓";
    position: absolute;
    left: 0;
    color: #4CAF50;
    font-weight: bold;
}

/* ===== FOOTER ===== */
.app-footer {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important;
    color: #ffffff !important;
    padding: 1.25rem !important;
    border-radius: 12px !important;
    margin-top: 1rem !important;
    text-align: center !important;
}

.app-footer p {
    margin: 0.25rem 0 !important;
    font-size: 0.9rem !important;
    color: #ffffff !important;
}

.app-footer .title {
    font-weight: 600 !important;
    font-size: 1rem !important;
    color: #ffffff !important;
}

.app-footer .team {
    font-size: 0.85rem !important;
    color: #e0e0e0 !important;
}

.app-footer .copyright {
    font-size: 0.8rem !important;
    color: #b0b0b0 !important;
    margin-top: 0.5rem !important;
}

/* ===== BUTTON STYLING ===== */
button.primary {
    background: linear-gradient(135deg, #4CAF50, #45a049) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 8px rgba(76,175,80,0.3) !important;
    transition: all 0.2s ease !important;
}

button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(76,175,80,0.4) !important;
}

button.secondary {
    background: #6c757d !important;
    color: white !important;
}

button.stop {
    background: linear-gradient(135deg, #f44336, #d32f2f) !important;
}

/* ===== IMAGE COMPONENT ===== */
.image-container {
    border: 2px dashed #dee2e6 !important;
    border-radius: 10px !important;
    transition: border-color 0.2s ease !important;
}

.image-container:hover {
    border-color: #4CAF50 !important;
}

/* Row alignment */
.gr-row {
    gap: 1rem !important;
}

.gr-column {
    display: flex;
    flex-direction: column;
}
"""

def predict_sign(image):
    """
    Predict sign language letter from image
    
    Args:
        image: PIL Image or numpy array
        
    Returns:
        tuple: (letter_html, predictions_html, confidence_dict)
    """
    try:
        # Handle None image (webcam not started)
        if image is None:
            empty_html = """
            <div class="empty-state">
                <div class="empty-icon">📷</div>
                <h3>No Image</h3>
                <p>Please upload an image or capture from webcam</p>
            </div>
            """
            return "", empty_html, {}
        
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Convert RGB to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Encode image as JPEG
        is_success, buffer = cv2.imencode(".jpg", image)
        if not is_success:
            return "❌", "<p>Failed to encode image</p>", {}
        
        image_data = buffer.tobytes()
        
        # Run prediction
        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
        
        # Get top predictions
        top_k = predictions[0].argsort()[-5:][::-1]
        
        # Create confidence dictionary for top 5
        confidence_dict = {
            label_lines[node_id]: float(predictions[0][node_id])
            for node_id in top_k
        }
        
        # Best prediction
        best_idx = top_k[0]
        prediction = label_lines[best_idx]
        confidence = float(predictions[0][best_idx]) * 100
        
        # Create letter display HTML - Clean aligned design
        letter_html = f"""
        <div class="result-container">
            <h3>🎯 Prediction Result</h3>
            <div class="result-letter">{prediction.upper()}</div>
            <div class="result-confidence">
                Confidence: <strong>{confidence:.1f}%</strong>
            </div>
        </div>
        """
        
        # Create predictions bar HTML - Clean aligned design
        predictions_html = '<div class="predictions-container"><h4>📊 Top 5 Predictions</h4>'
        for node_id in top_k:
            label = label_lines[node_id]
            score = float(predictions[0][node_id]) * 100
            predictions_html += f"""
            <div class="pred-row">
                <span class="pred-label">{label.upper()}</span>
                <div class="pred-bar-bg">
                    <div class="pred-bar-fill" style="width: {max(score, 8)}%;">{score:.1f}%</div>
                </div>
            </div>
            """
        predictions_html += '</div>'
        
        # Add to history
        prediction_history.insert(0, {
            'letter': prediction.upper(),
            'confidence': confidence,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        # Keep only last 20
        if len(prediction_history) > 20:
            prediction_history.pop()
        
        return letter_html, predictions_html, confidence_dict
            
    except Exception as e:
        error_html = f"""
        <div style="background: #ffebee; border: 2px solid #f44336; border-radius: 10px; padding: 2rem; text-align: center;">
            <div style="font-size: 3rem;">❌</div>
            <h3 style="color: #f44336;">Error Processing Image</h3>
            <p>{str(e)}</p>
        </div>
        """
        return "", error_html, {}

def get_history_html():
    """Generate history table HTML"""
    if not prediction_history:
        return """
        <div class="empty-state">
            <div class="icon">📋</div>
            <h3>No Predictions Yet</h3>
            <p>Your prediction history will appear here after you classify some images.</p>
        </div>
        """
    
    html = """
    <div class="history-wrapper">
        <table class="history-table">
            <thead>
                <tr>
                    <th style="width:50px">#</th>
                    <th style="width:100px">Letter</th>
                    <th>Confidence</th>
                    <th style="width:160px">Timestamp</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for i, item in enumerate(prediction_history, 1):
        conf = item['confidence']
        html += f"""
        <tr>
            <td>{i}</td>
            <td><span class="letter-badge">{item['letter']}</span></td>
            <td><div class="conf-bar" style="width: {max(conf, 15)}%;">{conf:.1f}%</div></td>
            <td>{item['timestamp']}</td>
        </tr>
        """
    
    html += """
            </tbody>
        </table>
        <p style="text-align: center; margin-top: 1rem; color: #666;">
            Total Predictions: """ + str(len(prediction_history)) + """
        </p>
    </div>
    """
    
    return html

def clear_history():
    """Clear prediction history"""
    prediction_history.clear()
    return get_history_html()

def refresh_history():
    """Refresh history display"""
    return get_history_html()

def download_history_csv():
    """Generate CSV file for download"""
    if not prediction_history:
        # Return empty file path indicator
        return gr.File(value=None, visible=True)
    
    import csv
    import tempfile
    
    # Create CSV content in temp directory
    csv_path = os.path.join(tempfile.gettempdir(), "sign_language_history.csv")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['#', 'Letter', 'Confidence (%)', 'Timestamp'])
        for i, item in enumerate(prediction_history, 1):
            writer.writerow([i, item['letter'], f"{item['confidence']:.2f}", item['timestamp']])
    
    return gr.File(value=csv_path, visible=True)

def download_last_result():
    """Generate a result image for the last prediction"""
    if not prediction_history:
        return gr.File(value=None, visible=True)
    
    from PIL import Image as PILImage, ImageDraw
    import tempfile
    
    last = prediction_history[0]
    
    # Create result image - larger and more visible
    img = PILImage.new('RGB', (500, 400), color='#ffffff')
    draw = ImageDraw.Draw(img)
    
    # Draw green header bar
    draw.rectangle([0, 0, 500, 80], fill='#4CAF50')
    
    # Draw header text (using default font, positioned manually)
    # Title area
    draw.rectangle([150, 20, 350, 60], fill='#4CAF50')
    
    # Draw large letter in center
    letter = last['letter']
    # Draw letter background circle
    draw.ellipse([175, 120, 325, 270], fill='#e8f5e9', outline='#4CAF50', width=3)
    
    # Draw confidence bar background
    draw.rectangle([50, 300, 450, 330], fill='#e0e0e0', outline='#bdbdbd')
    
    # Draw confidence bar fill
    conf_width = int(400 * (last['confidence'] / 100))
    draw.rectangle([50, 300, 50 + conf_width, 330], fill='#4CAF50')
    
    # Draw footer with timestamp
    draw.rectangle([0, 360, 500, 400], fill='#f5f5f5')
    
    # Save image
    result_path = os.path.join(tempfile.gettempdir(), "sign_language_result.png")
    img.save(result_path)
    
    return gr.File(value=result_path, visible=True)

# Create Gradio interface with custom styling
with gr.Blocks(title="Sign Language Alphabet Recognizer") as demo:
    
    # Inject CSS via style tag (compatible with Gradio 4.0.0)
    gr.HTML(f"<style>{custom_css}</style>")
    
    # Header
    gr.HTML("""
    <div class="app-header">
        <h1>🤟 Sign Language Recognizer</h1>
        <div class="subtitle">Phase 3: ML-Based System Architecture Extension</div>
    </div>
    """)
    
    # Hero Section
    gr.HTML("""
    <div class="hero-banner">
        <h2>Translate Sign Language to Text Instantly</h2>
        <p>Upload an image or use your webcam to recognize ASL alphabet gestures</p>
        <div class="tech-badges">
            <span class="tech-badge">🐍 Python</span>
            <span class="tech-badge">🧠 TensorFlow</span>
            <span class="tech-badge">🎨 Gradio</span>
            <span class="tech-badge">📷 OpenCV</span>
            <span class="tech-badge">☁️ Hugging Face</span>
        </div>
    </div>
    """)
    
    # Main Tabs
    with gr.Tabs() as tabs:
        
        # Upload Tab
        with gr.TabItem("📤 Upload", id=1):
            gr.HTML("""
            <div class="section-card">
                <div class="icon">📸</div>
                <h3>Upload & Classify</h3>
                <p>Upload an image of a hand gesture to get instant predictions</p>
            </div>
            """)
            
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    upload_image = gr.Image(
                        label="Drop image here or click to upload",
                        type="pil",
                        sources=["upload", "clipboard"],
                        height=300
                    )
                    upload_btn = gr.Button("🔍 Classify Sign", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    upload_result_letter = gr.HTML()
                    upload_result_bars = gr.HTML()
                    upload_confidence = gr.Label(num_top_classes=5, visible=False)
                    upload_download_btn = gr.Button("📥 Download Result", variant="secondary", size="sm")
                    upload_download_file = gr.File(label="📄 Click to download result", visible=True, interactive=False)
            
            upload_btn.click(
                fn=predict_sign,
                inputs=upload_image,
                outputs=[upload_result_letter, upload_result_bars, upload_confidence]
            )
            upload_image.change(
                fn=predict_sign,
                inputs=upload_image,
                outputs=[upload_result_letter, upload_result_bars, upload_confidence]
            )
            upload_download_btn.click(
                fn=download_last_result,
                inputs=None,
                outputs=upload_download_file
            )
        
        # Webcam Tab
        with gr.TabItem("📹 Webcam", id=2):
            gr.HTML("""
            <div class="section-card">
                <div class="icon">🎥</div>
                <h3>Real-time Webcam</h3>
                <p>Use your webcam to capture and classify sign language gestures</p>
            </div>
            """)
            
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    webcam_image = gr.Image(
                        label="Webcam Feed",
                        type="pil",
                        sources=["webcam"],
                        height=300,
                        streaming=False
                    )
                    webcam_btn = gr.Button("🔍 Classify Captured Image", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    webcam_result_letter = gr.HTML()
                    webcam_result_bars = gr.HTML()
                    webcam_confidence = gr.Label(num_top_classes=5, visible=False)
            
            # Instructions
            gr.HTML("""
            <div class="instructions-card">
                <h4>📋 How to Use</h4>
                <ol>
                    <li>Click the camera icon to start your webcam</li>
                    <li>Position your hand showing an ASL letter</li>
                    <li>Click capture to take a snapshot</li>
                    <li>Click "Classify Captured Image" to get prediction</li>
                </ol>
                <div class="tip-box">
                    💡 <strong>Tip:</strong> Use good lighting and a plain background for best results.
                </div>
            </div>
            """)
            
            webcam_btn.click(
                fn=predict_sign,
                inputs=webcam_image,
                outputs=[webcam_result_letter, webcam_result_bars, webcam_confidence]
            )
            webcam_image.change(
                fn=predict_sign,
                inputs=webcam_image,
                outputs=[webcam_result_letter, webcam_result_bars, webcam_confidence]
            )
        
        # History Tab
        with gr.TabItem("📊 History", id=3):
            gr.HTML("""
            <div class="section-card">
                <div class="icon">📋</div>
                <h3>Prediction History</h3>
                <p>View your recent predictions and confidence scores</p>
            </div>
            """)
            
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh", variant="secondary", size="sm")
                download_csv_btn = gr.Button("📥 Download CSV", variant="primary", size="sm")
                clear_btn = gr.Button("🗑️ Clear All", variant="stop", size="sm")
            
            history_display = gr.HTML(value=get_history_html())
            download_csv_file = gr.File(label="📄 Click to download CSV", visible=True, interactive=False)
            
            refresh_btn.click(fn=refresh_history, inputs=None, outputs=history_display)
            download_csv_btn.click(fn=download_history_csv, inputs=None, outputs=download_csv_file)
            clear_btn.click(fn=clear_history, inputs=None, outputs=history_display)
        
        # About Tab
        with gr.TabItem("ℹ️ About", id=4):
            gr.HTML("""
            <div class="info-section">
                <h4>📊 Model Information</h4>
                <div class="info-grid">
                    <div class="info-card">
                        <strong>Framework</strong>
                        <span>TensorFlow 2.11.0</span>
                    </div>
                    <div class="info-card">
                        <strong>Architecture</strong>
                        <span>InceptionV3 Transfer Learning</span>
                    </div>
                    <div class="info-card">
                        <strong>Training Steps</strong>
                        <span>10,000</span>
                    </div>
                    <div class="info-card">
                        <strong>Classes</strong>
                        <span>29 (A-Z + Space + Del + Nothing)</span>
                    </div>
                </div>
            </div>
            
            <div class="info-section">
                <h4>🎯 Supported Gestures</h4>
                <p style="margin-bottom: 0.75rem; font-size: 0.9rem; color: #666;">
                    This model recognizes American Sign Language (ASL) alphabet gestures:
                </p>
                <div class="letter-grid">
                    <span class="letter-chip">A</span><span class="letter-chip">B</span>
                    <span class="letter-chip">C</span><span class="letter-chip">D</span>
                    <span class="letter-chip">E</span><span class="letter-chip">F</span>
                    <span class="letter-chip">G</span><span class="letter-chip">H</span>
                    <span class="letter-chip">I</span><span class="letter-chip">J</span>
                    <span class="letter-chip">K</span><span class="letter-chip">L</span>
                    <span class="letter-chip">M</span><span class="letter-chip">N</span>
                    <span class="letter-chip">O</span><span class="letter-chip">P</span>
                    <span class="letter-chip">Q</span><span class="letter-chip">R</span>
                    <span class="letter-chip">S</span><span class="letter-chip">T</span>
                    <span class="letter-chip">U</span><span class="letter-chip">V</span>
                    <span class="letter-chip">W</span><span class="letter-chip">X</span>
                    <span class="letter-chip">Y</span><span class="letter-chip">Z</span>
                    <span class="letter-chip blue">Space</span>
                    <span class="letter-chip red">Del</span>
                    <span class="letter-chip gray">Nothing</span>
                </div>
            </div>
            
            <div class="info-section">
                <h4>✨ Features</h4>
                <ul class="feature-list">
                    <li>Web-based interface powered by Gradio</li>
                    <li>Real-time prediction via webcam</li>
                    <li>Image upload support (drag & drop)</li>
                    <li>Confidence scores visualization</li>
                    <li>Prediction history tracking</li>
                    <li>Cloud deployment on Hugging Face Spaces</li>
                </ul>
            </div>
            
            <div class="instructions-card">
                <h4>💡 Tips for Best Results</h4>
                <ul>
                    <li>Use clear, well-lit images</li>
                    <li>Use a plain, contrasting background</li>
                    <li>Ensure your hand gesture is clearly visible</li>
                    <li>Keep your hand steady when capturing</li>
                    <li>Center your hand in the frame</li>
                </ul>
            </div>
            """)
    
    # Footer
    gr.HTML("""
    <div class="app-footer">
        <p class="title">🤟 Sign Language Alphabet Recognizer</p>
        <p style="color: #ffffff !important;">Phase 3: ML-Based System Architecture Extension</p>
        <p class="team">
            <span style="color: #4CAF50 !important; font-weight: bold;">Group 4:</span> 
            <span style="color: #e0e0e0 !important;">Fayaz Shaik, Harsha Koritala, Mallikarjun Kotha, Sai Grishyanth Magunta, Sai Kiran Dasari</span>
        </p>
        <p class="copyright">© 2024 - Software Development for AI</p>
    </div>
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        ssr_mode=False
    )

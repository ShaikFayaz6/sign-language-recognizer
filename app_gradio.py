"""
Gradio App for Sign Language Alphabet Recognizer
Adapted for Hugging Face Spaces deployment
"""

import gradio as gr
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import os

# Disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.disable_eager_execution()

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

def predict_sign(image):
    """
    Predict sign language letter from image
    
    Args:
        image: PIL Image or numpy array
        
    Returns:
        tuple: (prediction, confidence_dict)
    """
    try:
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
            return "❌ Failed to encode image", {}
        
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
        
        # Format output
        output_text = f"""
🔤 **Predicted Letter:** {prediction.upper()}
📊 **Confidence:** {confidence:.1f}%

✨ The model is **{confidence:.1f}%** confident this is the letter **'{prediction.upper()}'**
        """
        
        return output_text, confidence_dict
            
    except Exception as e:
        error_msg = f"❌ Error processing image: {str(e)}"
        return error_msg, {}

# Create Gradio interface
with gr.Blocks(title="Sign Language Alphabet Recognizer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤟 Sign Language Alphabet Recognizer
    
    ### Phase 3: ML-Based System Architecture Extension
    
    **Upload an image** of an American Sign Language (ASL) hand gesture, and the model will predict the letter!
    
    **Supported Letters:** A-Z, Space, Del, Nothing (29 classes)
    
    ---
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input image
            input_image = gr.Image(
                label="Upload Sign Language Image",
                type="pil",
                sources=["upload", "webcam", "clipboard"],
                height=400
            )
            
            # Predict button
            predict_btn = gr.Button("🔍 Predict Sign", variant="primary", size="lg")
            
            # Example images - only show if files exist
            try:
                example_images = []
                for letter in ['A', 'B', 'C']:
                    folder = f"dataset/{letter}"
                    if os.path.exists(folder):
                        images = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png'))]
                        if images:
                            example_images.append([os.path.join(folder, images[0])])
                
                if example_images:
                    gr.Markdown("### 📸 Try These Examples:")
                    gr.Examples(
                        examples=example_images,
                        inputs=input_image,
                        label="Sample Images"
                    )
            except Exception as e:
                print(f"Warning: Could not load example images: {e}")
        
        with gr.Column(scale=1):
            # Prediction output
            output_text = gr.Markdown(label="Prediction Result")
            
            # Confidence scores
            output_confidence = gr.Label(
                label="Confidence Scores (Top 5)",
                num_top_classes=5
            )
    
    # Connect button to prediction function
    predict_btn.click(
        fn=predict_sign,
        inputs=input_image,
        outputs=[output_text, output_confidence]
    )
    
    # Auto-predict on image upload
    input_image.change(
        fn=predict_sign,
        inputs=input_image,
        outputs=[output_text, output_confidence]
    )
    
    gr.Markdown("""
    ---
    
    ### 📊 Model Information
    - **Framework:** TensorFlow 2.11.0
    - **Architecture:** InceptionV3 Transfer Learning
    - **Training Steps:** 10,000
    - **Accuracy:** Improved model
    - **Classes:** 29 (A-Z + Space + Del + Nothing)
    
    ### 🎯 Phase 3 Features
    - ✅ Web-based interface (Gradio)
    - ✅ Real-time prediction
    - ✅ Confidence scores
    - ✅ Multi-class classification
    - ✅ Cloud deployment ready
    
    ### 👥 Team
    **Group No. 4:** Fayaz Shaik, Harsha Koritala, Mallikarjun Kotha, Sai Grishyanth Magunta, Sai Kiran Dasari
    
    ---
    
    💡 **Tip:** For best results, use clear images with good lighting and a plain background.
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,  # Default Hugging Face Spaces port
        share=False
    )

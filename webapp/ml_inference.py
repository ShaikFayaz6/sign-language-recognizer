"""
ML Inference Engine
Handles model loading and predictions with caching
"""
import os
import sys
import tensorflow as tf
import numpy as np

# Disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.disable_eager_execution()


class MLInferenceEngine:
    """Machine Learning Inference Engine for Sign Language Recognition"""
    
    def __init__(self):
        # Get the directory where this script is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        
        self.model_path = os.path.join(parent_dir, "logs", "output_graph_improved.pb")
        self.labels_path = os.path.join(parent_dir, "logs", "output_labels_improved.txt")
        self.sess = None
        self.softmax_tensor = None
        self.label_lines = []
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and labels into memory (cached)"""
        try:
            # Load labels
            with tf.io.gfile.GFile(self.labels_path, 'r') as f:
                self.label_lines = [line.rstrip() for line in f]
            
            # Load frozen graph
            with tf.io.gfile.GFile(self.model_path, 'rb') as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(graph_def, name='')
            
            # Create session and get tensor
            self.sess = tf.compat.v1.Session()
            self.softmax_tensor = self.sess.graph.get_tensor_by_name('final_result:0')
            
            print(f"✓ Model loaded successfully from {self.model_path}")
            print(f"✓ Labels loaded: {len(self.label_lines)} classes")
            
        except Exception as e:
            print(f"✗ Error loading model: {str(e)}")
            self.sess = None
            self.softmax_tensor = None
    
    def is_loaded(self):
        """Check if model is loaded"""
        return self.sess is not None and self.softmax_tensor is not None
    
    def predict(self, image_data):
        """
        Predict sign language letter from image
        
        Args:
            image_data: bytes - Image data (JPEG/PNG bytes)
        
        Returns:
            dict: {
                'prediction': str,
                'confidence': float,
                'top_predictions': list of tuples (label, confidence)
            }
        """
        if not self.is_loaded():
            raise Exception("Model not loaded. Cannot make predictions.")
        
        try:
            # Run prediction
            predictions = self.sess.run(
                self.softmax_tensor,
                {'DecodeJpeg/contents:0': image_data}
            )
            
            # Get top predictions
            top_k = predictions[0].argsort()[-5:][::-1]  # Top 5 predictions
            
            # Format results
            top_predictions = [
                {
                    'label': self.label_lines[node_id],
                    'confidence': float(predictions[0][node_id])
                }
                for node_id in top_k
            ]
            
            # Best prediction
            best_prediction = top_predictions[0]
            
            result = {
                'prediction': best_prediction['label'],
                'confidence': best_prediction['confidence'],
                'top_predictions': top_predictions,
                'success': True
            }
            
            return result
        
        except Exception as e:
            raise Exception(f"Prediction error: {str(e)}")
    
    def __del__(self):
        """Cleanup session on deletion"""
        if self.sess:
            self.sess.close()

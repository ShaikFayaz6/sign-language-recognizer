"""
Database Module
Handles SQLite operations for prediction history
"""
import sqlite3
from datetime import datetime
import os


class Database:
    """SQLite Database Manager for Prediction History"""
    
    def __init__(self, db_path='predictions.db'):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database and create table if not exists"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    predicted_letter TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            print(f"✓ Database initialized: {self.db_path}")
            
        except Exception as e:
            print(f"✗ Database initialization error: {str(e)}")
    
    def is_connected(self):
        """Check if database file exists and is accessible"""
        return os.path.exists(self.db_path)
    
    def add_prediction(self, predicted_letter, confidence):
        """
        Add a new prediction to history
        
        Args:
            predicted_letter: str - Predicted sign language letter
            confidence: float - Confidence score (0.0 to 1.0)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO predictions (predicted_letter, confidence)
                VALUES (?, ?)
            ''', (predicted_letter, confidence))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error adding prediction: {str(e)}")
            raise
    
    def get_recent_predictions(self, limit=20):
        """
        Get recent predictions from history
        
        Args:
            limit: int - Maximum number of records to return
        
        Returns:
            list of dict: [{id, predicted_letter, confidence, timestamp}, ...]
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, predicted_letter, confidence, timestamp
                FROM predictions
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            # Convert to list of dicts
            predictions = [
                {
                    'id': row['id'],
                    'predicted_letter': row['predicted_letter'],
                    'confidence': row['confidence'],
                    'timestamp': row['timestamp']
                }
                for row in rows
            ]
            
            return predictions
        
        except Exception as e:
            print(f"Error fetching predictions: {str(e)}")
            return []
    
    def get_statistics(self):
        """Get prediction statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total predictions
            cursor.execute('SELECT COUNT(*) FROM predictions')
            total = cursor.fetchone()[0]
            
            # Most predicted letter
            cursor.execute('''
                SELECT predicted_letter, COUNT(*) as count
                FROM predictions
                GROUP BY predicted_letter
                ORDER BY count DESC
                LIMIT 1
            ''')
            most_common = cursor.fetchone()
            
            # Average confidence
            cursor.execute('SELECT AVG(confidence) FROM predictions')
            avg_confidence = cursor.fetchone()[0] or 0.0
            
            conn.close()
            
            return {
                'total_predictions': total,
                'most_common_letter': most_common[0] if most_common else None,
                'most_common_count': most_common[1] if most_common else 0,
                'average_confidence': avg_confidence
            }
        
        except Exception as e:
            print(f"Error getting statistics: {str(e)}")
            return None
    
    def clear_history(self):
        """Clear all prediction history"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM predictions')
            
            conn.commit()
            conn.close()
            
            print("✓ Prediction history cleared")
            
        except Exception as e:
            print(f"Error clearing history: {str(e)}")
            raise

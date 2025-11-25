# Use Python 3.8 slim image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY webapp/ ./webapp/
COPY logs/ ./logs/
COPY dataset/ ./dataset/

# Create temp_uploads directory
RUN mkdir -p webapp/temp_uploads

# Set environment variables
ENV FLASK_APP=webapp/app.py
ENV FLASK_ENV=production
ENV PORT=8080

# Expose port
EXPOSE 8080

# Run gunicorn
CMD gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 webapp.app:app

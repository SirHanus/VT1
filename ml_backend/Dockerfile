FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV and other libraries
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    label-studio-ml==1.0.9 \
    redis \
    rq \
    requests \
    flask \
    flask-cors \
    ultralytics \
    torch \
    torchvision \
    opencv-python-headless \
    numpy \
    Pillow

# Create the ML backend directory structure
RUN mkdir -p /app

# Copy model code and WSGI entry point
COPY model.py /app/model.py
COPY _wsgi.py /app/_wsgi.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Create a minimal __init__.py to make it a package
RUN echo "from .model import YOLOBackend" > /app/__init__.py

# Environment variables with defaults
ENV YOLO_MODEL=yolo11n-pose.pt
ENV CONF_THRESHOLD=0.25
ENV LOG_LEVEL=INFO

# Expose port
EXPOSE 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:9090/health', timeout=5)" || exit 1

# Start ML backend using Python directly
CMD ["python", "/app/_wsgi.py"]


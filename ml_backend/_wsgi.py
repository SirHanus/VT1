"""
WSGI application entry point for Label Studio ML Backend
"""

import sys

# Add the app directory to the path
sys.path.insert(0, "/app")

from label_studio_ml.api import init_app
from model import YOLOBackend

# Initialize the Flask app with the model class
application = init_app(model_class=YOLOBackend, model_dir="/app")

if __name__ == "__main__":
    application.run(host="0.0.0.0", port=9090, debug=False)

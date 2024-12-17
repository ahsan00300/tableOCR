# main.py
from flask import Flask
# from flask_cors import CORS

from app.api.table_extraction import ocr_blueprint
from app.api.health import health_blueprint

# Initialize the Flask application
app = Flask(__name__)

# Setup CORS
# CORS(app, origins=["*"])

# Register blueprints
app.register_blueprint(ocr_blueprint)
app.register_blueprint(health_blueprint)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8002, debug=True)
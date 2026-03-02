from flask import Flask
from flask_cors import CORS
from config import Config
import os


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    CORS(app)

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    from app.routes.analyze import analyze_bp
    app.register_blueprint(analyze_bp, url_prefix="/api")

    return app
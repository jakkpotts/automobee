from flask import Flask
from flask_cors import CORS
import os

def create_app():
    app = Flask(__name__)
    CORS(app)
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev')  # Use environment variable
    return app

app = create_app()
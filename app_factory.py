from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)
    app.config['SECRET_KEY'] = 'dev'  # Add secret key for sessions/csrf
    return app

app = create_app()
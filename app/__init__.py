from flask import Flask
from .routes import predict, retrain, home

def create_app():
    app = Flask(__name__)
    app.add_url_rule('/', 'home', home)
    app.add_url_rule('/predict', 'predict', predict, methods=['POST'])
    app.add_url_rule('/train', 'train', retrain, methods=['POST'])
    return app

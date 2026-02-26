from src.settings import PROJECT_DIR
import os
import joblib
import numpy as np

def loader(topic):
    topic = topic.strip()
    model_path = os.path.join(PROJECT_DIR, "src", "models",  topic, "optimized_text_classifier.pkl")
    label_encoder_path = os.path.join(PROJECT_DIR, "src", "models",  topic, "label_encoder.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(label_encoder_path):
        raise FileNotFoundError(f"Label encoder file not found: {label_encoder_path}")
    # Load the model (this is just a placeholder, replace with actual loading code)
    model = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)

    return model, label_encoder
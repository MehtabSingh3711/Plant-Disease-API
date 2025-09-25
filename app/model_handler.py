# app/model_handler.py
import os
import json
import tensorflow as tf
from vercel_blob import blob
from typing import Dict, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOADED_MODELS: Dict[str, tf.keras.Model] = {}
MODEL_CONFIGS: Dict[str, Any] = {}
CLASS_NAMES: Dict[str, Dict[str, str]] = {}

def load_configs():
    if not MODEL_CONFIGS:
        config_path = os.path.join(PROJECT_ROOT, 'config.json')
        print(f"Loading configs from absolute path: {config_path}")
        with open(config_path, 'r') as f:
            configs = json.load(f)
            MODEL_CONFIGS.update(configs)
            for plant_type, config in configs.items():
                classes_path = os.path.join(PROJECT_ROOT, config['classes_path'])
                with open(classes_path, 'r') as cf:
                    CLASS_NAMES[plant_type] = json.load(cf)

async def get_model(plant_type: str) -> tf.keras.Model:
    if plant_type in LOADED_MODELS:
        print(f"Model '{plant_type}' found in memory cache.")
        return LOADED_MODELS[plant_type]

    local_model_path = f"/tmp/{plant_type}_model.h5"

    if os.path.exists(local_model_path):
        print(f"Model '{plant_type}' found in /tmp cache. Loading...")
        model = tf.keras.models.load_model(local_model_path)
        LOADED_MODELS[plant_type] = model
        return model

    print(f"Model '{plant_type}' not found in cache. Downloading from Blob...")
    model_url = MODEL_CONFIGS[plant_type]['model_url']
    await blob.download(url=model_url, pathname=local_model_path)
    
    print(f"Download complete. Loading model '{plant_type}'...")
    model = tf.keras.models.load_model(local_model_path)
    LOADED_MODELS[plant_type] = model
    return model
# app/preprocessing.py
from PIL import Image
import numpy as np
import io
from typing import Dict, Any
import tensorflow as tf

def preprocess_image(image_bytes: bytes, config: Dict[str, Any]) -> np.ndarray:
    mode = config.get("preprocessing_mode")

    if mode == "internal":
        print("Preprocessing Mode: INTERNAL")
        img = tf.keras.utils.load_img(io.BytesIO(image_bytes))
        img_array = tf.keras.utils.img_to_array(img)
        processed_image = np.expand_dims(img_array, axis=0)
        return processed_image

    elif mode == "external":
        print("Preprocessing Mode: EXTERNAL")
        img_size = tuple(config["image_size"])
        normalization = config["normalization_type"]
        color_order = config.get("color_channel_order", "RGB") # Default to RGB

        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize(img_size)
        img_array = np.array(img)

        if color_order == "BGR":
            img_array = img_array[:, :, ::-1]

        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

        if normalization == "scale_zero_one":
            processed_image = img_array / 255.0
        elif normalization == "scale_minus_one_one":
            processed_image = (img_array / 127.5) - 1.0
        else:
            raise ValueError(f"Unknown normalization_type in config: {normalization}")
        
        return processed_image
    
    else:
        raise ValueError(f"A 'preprocessing_mode' ('internal' or 'external') must be set in config.json for this model.")
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from typing import List

from app.model_handler import get_model, load_configs, MODEL_CONFIGS, CLASS_NAMES
from app.preprocessing import preprocess_image

app = FastAPI(title="Plant Disease Detection API on Vercel")

@app.on_event("startup")
async def startup_event():
    load_configs()

@app.get("/api")
def read_root():
    return {"message": "Welcome to the Plant Disease Detection API. Use /api/predict to analyze an image."}

@app.post("/api/predict")
async def predict(
    plant_type: str = Form(...),
    file: UploadFile = File(...)
):
    if plant_type not in MODEL_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Invalid 'plant_type'.")

    model_config = MODEL_CONFIGS[plant_type]
    
    model = await get_model(plant_type)
    
    class_map = CLASS_NAMES[plant_type]
    
    image_bytes = await file.read()
    processed_image = preprocess_image(image_bytes, config=model_config)
    
    prediction = model.predict(processed_image)
    predicted_index = str(np.argmax(prediction[0]))
    
    predicted_disease = class_map.get(predicted_index, "Unknown Class")
    confidence = float(np.max(prediction[0]))
    
    return JSONResponse(content={
        "plant_type": plant_type,
        "predicted_disease": predicted_disease,
        "confidence": f"{confidence:.2%}",
    })
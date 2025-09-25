# app/main.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from typing import List

from .model_loader import load_all_models, MODELS, CLASS_NAMES, MODEL_CONFIGS, get_supported_plants
from .preprocessing import preprocess_image

app = FastAPI(title="Plant Disease Detection API")

@app.on_event("startup")
async def startup_event():
    load_all_models()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Plant Disease Detection API. Use /predict to analyze an image."}

@app.get("/supported_plants", response_model=List[str])
def list_supported_plants():
    """Returns a list of all plant types supported by the loaded configurations."""
    return get_supported_plants()

@app.post("/predict")
async def predict(
    plant_type: str = Form(...),
    file: UploadFile = File(...)
):
    supported_plants = get_supported_plants()
    if plant_type not in supported_plants:
        raise HTTPException(status_code=400, detail=f"Invalid 'plant_type'. Supported types are: {supported_plants}")

    try:
        model_config = MODEL_CONFIGS[plant_type]
        model = MODELS[plant_type]
        class_map = CLASS_NAMES[plant_type]
    except KeyError:
        raise HTTPException(status_code=500, detail=f"Model assets for '{plant_type}' not loaded correctly. Check server logs.")

    image_bytes = await file.read()
    try:
        processed_image = preprocess_image(image_bytes, config=model_config)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")
    
    try:
        prediction = model.predict(processed_image)
        predicted_index = str(np.argmax(prediction[0]))
        
        predicted_disease = class_map.get(predicted_index, f"Unknown Class Index: {predicted_index}")
        confidence = float(np.max(prediction[0]))
        
        return JSONResponse(content={
            "plant_type": plant_type,
            "predicted_disease": predicted_disease,
            "confidence": f"{confidence:.2%}",
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")
import time
import sqlite3
import torch
import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends
from fastapi.responses import JSONResponse, RedirectResponse
from torchvision import models, transforms
from PIL import Image
import io
from constants import CLASS_NAMES

# Load environment variables from .env file
load_dotenv()

# Get the security key; fallback to a default if not set (for safety in dev)
API_SECURITY_KEY = os.getenv("API_SECURITY_KEY", "default-dev-key")

app = FastAPI(title="Plant Disease Diagnosis API", version="1.0.0")

# Security Dependency definition
async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_SECURITY_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    return x_api_key

# Load model globally to avoid loading it on every request
device = torch.device("cpu")
model = models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Sequential(
    torch.nn.Dropout(0.2),
    torch.nn.Linear(model.last_channel, len(CLASS_NAMES))
)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_assets", "mobilenetv2_plant.pth")

if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
else:
    print(f"Warning: Model file not found at {MODEL_PATH}. Predictions will fail.")

# --- PREPROCESSING ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "plant_pathology.db")

def get_diagnosis_details(class_name: str) -> dict:
    """Query the SQLite database for diagnosis details."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT common_name, scientific_name, symptoms, organic_control, chemical_control, severity FROM disease_profiles WHERE class_key = ?",
            (class_name,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return {
            "common_name": "Unknown",
            "scientific_name": "Unknown",
            "symptoms": "No details available in the database.",
            "organic_control": "N/A",
            "chemical_control": "N/A",
            "severity": "Unknown"
        }
    except Exception as e:
        print(f"Database error: {e}")
        return {
            "error": "Failed to retrieve diagnosis details from the database."
        }

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

@app.post("/v1/predict")
async def predict(file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    start_time = time.time()
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    
    try:
        # Read the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocess the image
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            preds = model(input_tensor)
            probs = torch.softmax(preds, dim=1)[0]
            index = probs.argmax().item()
            confidence = probs[index].item()
            
        class_name = CLASS_NAMES[index]
        is_healthy = class_name.endswith("___healthy")
        
        # Get diagnosis details from the database
        diagnosis_details = get_diagnosis_details(class_name)
        
        inference_time_ms = (time.time() - start_time) * 1000
        
        # Expected JSON Response Structure
        response_data = {
            "status": "success",
            "data": {
                "prediction": {
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "is_healthy": is_healthy
                },
                "diagnosis": diagnosis_details
            },
            "metadata": {
                "inference_time_ms": round(inference_time_ms, 2)
            }
        }
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)


import os
import json
import torch
from torchvision import transforms
from PIL import Image
import timm
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import io

app = FastAPI(title="Tomato Disease Detection API")

# Global variables
model = None
class_names = []
treatment_map = {}
device = torch.device("cpu") # Render free tier usually CPU only

def load_resources():
    global model, class_names, treatment_map
    
    # Load Labels
    if os.path.exists("labels.txt"):
        with open("labels.txt", "r") as f:
            class_names = [line.strip() for line in f.readlines()]
    else:
        print("Warning: labels.txt not found.")
        
    # Load Treatments
    if os.path.exists("treatments.json"):
        with open("treatments.json", "r", encoding="utf-8") as f:
            treatment_map = json.load(f)
    else:
        print("Warning: treatments.json not found.")

    # Load Model
    if os.path.exists("tomato_disease_model.pth") and class_names:
        try:
            model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=len(class_names))
            state_dict = torch.load("tomato_disease_model.pth", map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            model = None
    else:
        print("Warning: Model file not found or labels missing.")

# Initialize on startup
load_resources()

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.get("/")
def read_root():
    return {"message": "Tomato Disease Detection API is running"}

@app.get("/treatments")
def get_treatments():
    return treatment_map

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top1_prob, top1_catid = torch.topk(probabilities, 1)
            
            predicted_label = class_names[top1_catid.item()]
            confidence = top1_prob.item()
            
            treatment = treatment_map.get(predicted_label, None)
            
            return {
                "filename": file.filename,
                "prediction": predicted_label,
                "confidence": confidence,
                "treatment": treatment
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# For local testing
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
import pickle
import numpy as np
from pathlib import Path
from os import getenv
import joblib

# Create FastAPI instance
app = FastAPI(
    title="Body Fat Prediction API",
    description="This API predicts body fat percentage based on body measurements.",
    version="1.0.0",
)

# Enable CORS for external requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Load Jinja2 templates
templates = Jinja2Templates(directory="api/templates")

# Load model
MODEL_PATH = Path(getenv("MODEL_PATH", "/app/models/model.pkl"))

if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)
else:
    model = None

# Define routes
@app.get("/")
def form_page(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict/", summary="Predict body fat percentage", description="Send body measurements to get a body fat percentage prediction.")
async def predict(
    request: Request,
    abdomen: float = Form(..., description="Abdomen circumference (cm)"),
    hip: float = Form(..., description="Hip circumference (cm)"),
    weight: float = Form(..., description="Weight (kg)"),
    thigh: float = Form(..., description="Thigh circumference (cm)"),
    knee: float = Form(..., description="Knee circumference (cm)"),
    biceps: float = Form(..., description="Biceps circumference (cm)"),
    neck: float = Form(..., description="Neck circumference (cm)"),
):
    if model is None:
        return {"error": "Model not found"}

    # Create array with features
    features = np.array([[abdomen, hip, weight, thigh, knee, biceps, neck]])
    print(features)

    # Ensure model has a predict function
    if not hasattr(model, "predict"):
        return {"error": f"Invalid model type: {type(model)}"}

    # Get prediction
    prediction = model.predict(features)[0]

    # Detect if the request wants JSON or HTML
    accept_type = request.headers.get("accept", "")

    if "application/json" in accept_type:  # API request
        return {"prediction": float(prediction)}

    # Otherwise, return the HTML page with the prediction result
    return templates.TemplateResponse("form.html", {
        "request": request,
        "prediction": prediction
    })

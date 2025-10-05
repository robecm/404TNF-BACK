from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from pydantic import BaseModel
from pathlib import Path
import os
import traceback

# --- Model Loading ---
# Build a robust path relative to this script's location (main.py)
SCRIPT_DIR = Path(__file__).parent.resolve()
MODEL_DIR = SCRIPT_DIR / "models"
model_path = MODEL_DIR / "exoplanet_model.pkl"
encoder_path = MODEL_DIR / "label_encoder.pkl"

model = None
label_encoder = None
model_loading_error = None

if not model_path.exists():
    model_loading_error = {"error_message": f"Model file not found at {model_path}"}
elif not encoder_path.exists():
    model_loading_error = {"error_message": f"Encoder file not found at {encoder_path}"}
else:
    try:
        model = joblib.load(model_path)
        label_encoder = joblib.load(encoder_path)
    except Exception as e:
        model_loading_error = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc()
        }
# --- End Model Loading ---


# --- Input Data Definition ---
class CandidateInput(BaseModel):
    koi_period: float
    koi_time0bk: float
    koi_impact: float
    koi_duration: float
    koi_depth: float
    koi_prad: float
    koi_teq: float
    koi_insol: float
    koi_model_snr: float
    koi_steff: float
    koi_slogg: float
    koi_srad: float
# --- End Input Data Definition ---


# --- FastAPI App Creation ---
app = FastAPI(
    title="Exoplanet Detection API - NASA Hackathon",
    description="An API to predict if an object of interest is an exoplanet based on physical characteristics.",
    version="1.0.0"
)

# --- Enable CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5174",  # your local frontend
        "http://localhost:5173",
        "https://404-tnf-front.vercel.app/",
        "https://404-tnf-front.vercel.app",
        "404-tnf-front.vercel.app",
        "exoptolemy.study", # production
        "www.exoptolemy.study",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --------------------------


@app.get("/")
def read_root():
    return {"message": "Welcome to the Exoplanet Detection API."}


@app.get("/api/health")
def health_check():
    return {"status": "healthy"}


@app.get("/debug-info", tags=["Status"])
def get_debug_info():
    """
    Provides detailed debugging information about the application environment.
    """
    def list_dir_safely(path_obj):
        try:
            if path_obj.exists() and path_obj.is_dir():
                return [p.name for p in path_obj.iterdir()]
            elif path_obj.exists():
                return f"'{path_obj}' exists but is not a directory."
            else:
                return f"Path '{path_obj}' does not exist."
        except Exception as e:
            return f"Error listing directory '{path_obj}': {str(e)}"

    return {
        "model_loading_error": model_loading_error,
        "environment_details": {
            "current_working_directory": os.getcwd(),
            "model_directory_path": str(MODEL_DIR),
            "model_file_path": str(model_path),
            "encoder_file_path": str(encoder_path),
        },
        "file_system_check": {
            "project_root_contents": list_dir_safely(Path(".")),
            "api_dir_contents": list_dir_safely(Path("api")),
            "models_dir_contents": list_dir_safely(MODEL_DIR),
            "model_file_exists": model_path.exists(),
            "encoder_file_exists": encoder_path.exists(),
        }
    }


# --- Prediction Endpoint ---
@app.post("/predict", tags=["Predictions"])
def get_prediction(data: CandidateInput):
    """
    Receives the 12 physical parameters of a candidate and returns a prediction.
    """
    if model_loading_error:
        return {"error": "Model is not loaded. Check server logs or the /debug-info endpoint.", "details": model_loading_error}
    if not model or not label_encoder:
        return {"error": "Model is not loaded for an unknown reason."}

    input_data = data.model_dump()
    feature_order = [
        'koi_period', 'koi_time0bk', 'koi_impact', 'koi_duration', 'koi_depth',
        'koi_prad', 'koi_teq', 'koi_insol', 'koi_model_snr', 'koi_steff',
        'koi_slogg', 'koi_srad'
    ]
    input_array = np.array([[input_data[feature] for feature in feature_order]])

    prediction_encoded = model.predict(input_array)[0]
    prediction_proba = model.predict_proba(input_array)[0]

    prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]
    confidence = float(prediction_proba[prediction_encoded])

    return {
        "verdict": prediction_label,
        "confidence": confidence
    }
# --- End Prediction Endpoint ---


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

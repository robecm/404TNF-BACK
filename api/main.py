from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
from pathlib import Path
import os
import traceback

# --- INICIO: Carga del Modelo (CORREGIDO) ---
# Construye la ruta de forma robusta, relativa a la ubicación de este script (main.py)
SCRIPT_DIR = Path(__file__).parent.resolve()
MODEL_DIR = SCRIPT_DIR / "models"
model_path = MODEL_DIR / "exoplanet_model.pkl"
encoder_path = MODEL_DIR / "label_encoder.pkl"

model = None
label_encoder = None
model_loading_error = None

# --- Comprobación explícita de la existencia de archivos ---
print(f"Current working directory: {os.getcwd()}")
print(f"Checking for model at: {model_path}")
if not model_path.exists():
    model_loading_error = {"error_message": f"Model file not found at {model_path}"}
    print(f"❌ ERROR: Model file not found at {model_path}")
elif not encoder_path.exists():
    model_loading_error = {"error_message": f"Encoder file not found at {encoder_path}"}
    print(f"❌ ERROR: Encoder file not found at {encoder_path}")
else:
    print("✅ Model and encoder files found. Attempting to load...")
    try:
        model = joblib.load(model_path)
        label_encoder = joblib.load(encoder_path)
        print("✅ Modelo y codificador cargados exitosamente.")
    except Exception as e:
        model_loading_error = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc()
        }
        print(f"❌ Error cargando modelos con joblib: {e}")
# --- FIN: Carga del Modelo ---


# --- INICIO: Definición de Datos de Entrada ---
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
# --- FIN: Definición de Datos de Entrada ---


# --- Creación de la Aplicación FastAPI ---
app = FastAPI(
    title="API de Detección de Exoplanetas - NASA Hackathon",
    description="Una API para predecir si un objeto de interés es un exoplaneta basado en características físicas.",
    version="1.0.0"
)


@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de Detección de Exoplanetas."}


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


# --- INICIO: Endpoint del Modelo de Predicción ---
@app.post("/predict", tags=["Predicciones"])
def get_prediction(data: CandidateInput):
    """
    Recibe los 12 parámetros físicos de un candidato y devuelve una predicción.
    """
    if model_loading_error:
        return {"error": "El modelo no está cargado. Revisa los logs del servidor o el endpoint /debug-info.", "details": model_loading_error}
    if not model or not label_encoder:
        return {"error": "El modelo no está cargado por una razón desconocida."}

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
        "veredicto": prediction_label,
        "confianza": confidence
    }
# --- FIN: Endpoint del Modelo de Predicción ---


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
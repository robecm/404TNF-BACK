from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
from pathlib import Path
import os
import traceback

# --- INICIO: Carga del Modelo ---
# Use a path relative to the project root, which is more reliable on Vercel.
# Vercel sets the working directory to the project root (`/var/task`).
# The path from the root to your models is 'api/models'.
MODEL_DIR = Path("api/models")

model_path = MODEL_DIR / "exoplanet_model.pkl"
encoder_path = MODEL_DIR / "label_encoder.pkl"

# Global variables to hold the models and any loading error
model = None
label_encoder = None
model_loading_error = None

try:
    # Log the current working directory and the resolved path for debugging
    print(f"Current working directory: {os.getcwd()}")
    print(f"Attempting to load model from: {model_path.resolve()}")

    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    print("✅ Modelo y codificador cargados exitosamente.")
except Exception as e:
    # Store the exception details for debugging
    model_loading_error = {
        "error_type": type(e).__name__,
        "error_message": str(e),
        "traceback": traceback.format_exc()
    }
    print(f"❌ Error cargando modelos: {e}")
# --- FIN: Carga del Modelo ---


# --- INICIO: Definición de Datos de Entrada ---
# Pydantic validará que los datos que lleguen a la API tengan este formato.
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
    # Helper to list directory contents safely
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
            "model_directory_path": str(MODEL_DIR.resolve()),
            "model_file_path": str(model_path.resolve()),
            "encoder_file_path": str(encoder_path.resolve()),
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
    if not model or not label_encoder:
        return {"error": "El modelo no está cargado. Revisa los logs del servidor o el endpoint /debug-info."}

    # Convertimos los datos de entrada a un array de NumPy
    # El orden de las características debe coincidir con el entrenamiento del modelo.
    input_data = data.model_dump() # Use model_dump() instead of deprecated dict()
    feature_order = [
        'koi_period', 'koi_time0bk', 'koi_impact', 'koi_duration', 'koi_depth',
        'koi_prad', 'koi_teq', 'koi_insol', 'koi_model_snr', 'koi_steff',
        'koi_slogg', 'koi_srad'
    ]
    input_array = np.array([[input_data[feature] for feature in feature_order]])


    # Realizamos la predicción y obtenemos las probabilidades
    prediction_encoded = model.predict(input_array)[0]
    prediction_proba = model.predict_proba(input_array)[0]

    # Decodificamos el resultado para obtener la etiqueta de texto (ej. "CANDIDATE")
    prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

    # Calculamos la confianza de la predicción
    confidence = float(prediction_proba[prediction_encoded])

    # Devolvemos el resultado en formato JSON
    return {
        "veredicto": prediction_label,
        "confianza": confidence
    }
# --- FIN: Endpoint del Modelo de Predicción ---


# Only run uvicorn locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
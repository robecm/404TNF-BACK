from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
from pathlib import Path

# --- INICIO: Carga del Modelo ---
# Detect absolute path for the current file
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

model_path = MODEL_DIR / "exoplanet_model.pkl"
encoder_path = MODEL_DIR / "label_encoder.pkl"

try:
    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    print("✅ Modelo y codificador cargados exitosamente.")
except Exception as e:
    print("❌ Error cargando modelos:", e)
    model = None
    label_encoder = None
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


@app.get("/model_status", tags=["Status"])
def model_status():
    """
    Verifica si el modelo y el codificador de etiquetas se cargaron correctamente.
    """
    model_loaded = model is not None
    encoder_loaded = label_encoder is not None

    if model_loaded and encoder_loaded:
        status_message = "Modelo y codificador cargados exitosamente."
    else:
        missing_files = []
        if not model_loaded:
            missing_files.append("exoplanet_model.pkl")
        if not encoder_loaded:
            missing_files.append("label_encoder.pkl")
        status_message = f"Error: No se pudieron cargar los siguientes archivos: {', '.join(missing_files)}"

    return {
        "model_loaded": model_loaded,
        "label_encoder_loaded": encoder_loaded,
        "status": status_message
    }


# --- INICIO: Endpoint del Modelo de Predicción ---
@app.post("/predict", tags=["Predicciones"])
def get_prediction(data: CandidateInput):
    """
    Recibe los 12 parámetros físicos de un candidato y devuelve una predicción.
    """
    if not model or not label_encoder:
        return {"error": "El modelo no está cargado. Revisa los logs del servidor."}

    # Convertimos los datos de entrada a un array de NumPy
    # El orden de las características debe coincidir con el entrenamiento del modelo.
    input_data = data.dict()
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
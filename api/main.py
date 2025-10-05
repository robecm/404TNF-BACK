from fastapi import FastAPI
import pandas as pd
import joblib
from pydantic import BaseModel

# --- INICIO: Carga del Modelo ---
# Cargamos el modelo y el codificador al iniciar la aplicación.
try:
    model = joblib.load('models/exoplanet_model.pkl')
    label_encoder = joblib.load('models/label_encoder.pkl')
    print("Modelo y codificador cargados exitosamente.")
except FileNotFoundError:
    model = None
    label_encoder = None
    print("ERROR: No se encontraron los archivos 'exoplanet_model.pkl' o 'label_encoder.pkl'.")
    print("Asegúrate de que los archivos del modelo estén en la misma carpeta que main.py.")
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


# --- INICIO: Endpoint del Modelo de Predicción ---
@app.post("/predict", tags=["Predicciones"])
def get_prediction(data: CandidateInput):
    """
    Recibe los 12 parámetros físicos de un candidato y devuelve una predicción.
    """
    if not model or not label_encoder:
        return {"error": "El modelo no está cargado. Revisa los logs del servidor."}

    # Convertimos los datos de entrada a un DataFrame de pandas
    input_df = pd.DataFrame([data.dict()])

    # Realizamos la predicción y obtenemos las probabilidades
    prediction_encoded = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

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
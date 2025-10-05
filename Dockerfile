FROM python:3.11-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 && rm -rf /var/lib/apt/lists/*

# Copiar requirements desde carpeta api
COPY api/requirements.txt /app/requirements.txt

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copiar todo el código
COPY . /app
WORKDIR /app

# Ejecutar FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]

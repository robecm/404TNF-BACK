# Usa una imagen base ligera con Python
FROM python:3.11-slim

# Define el directorio de trabajo
WORKDIR /app

# Copia los archivos del proyecto
COPY ./api /app

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Expone el puerto que usará FastAPI
EXPOSE 8080

# Comando para ejecutar la app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]


# main.py
from fastapi import FastAPI
from config import snow_conn

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World!"}

@app.get("/api/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
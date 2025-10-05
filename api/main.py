# main.py
from fastapi import FastAPI


app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World!"}

@app.get("/api/health")
def health_check():
    return {"status": "healthy"}

# Only run uvicorn locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

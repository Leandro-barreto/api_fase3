from fastapi import FastAPI
from routers import downloaddata

app = FastAPI()

app.include_router(downloaddata.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the ANEEL API"}

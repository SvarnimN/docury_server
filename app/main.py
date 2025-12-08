from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import file
from app.routers import chat

app = FastAPI()
app.include_router(file.router)
app.include_router(chat.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def index():
    return "Welcome to Docury."

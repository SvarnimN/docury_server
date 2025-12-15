from fastapi import APIRouter
from app.services.url import URLService
from app.services.vector import VectorService
from pydantic import BaseModel

router = APIRouter()
url_service = URLService()
vector_service = VectorService()

class Payload(BaseModel):
    url: str

@router.post("/url")
async def url(req: Payload):
    chunks = url_service.fetch_and_split(req.url)
    vector_service.add_documents(chunks)

    return {
        "status": "ok",
        "message": f"{req.url} fetched and indexed",
        "chunks": len(chunks),
    }

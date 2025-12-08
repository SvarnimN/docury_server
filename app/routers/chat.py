from fastapi import APIRouter
from pydantic import BaseModel
from app.services.rag import RAGService

class Payload(BaseModel):
    question: str
    session_id: str

router = APIRouter()

@router.post("/chat")
async def chat(req: Payload):
    rag_service = RAGService()
    response = rag_service.ask_question(req.question, req.session_id)

    return {"status": "ok", "response": response}

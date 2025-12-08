from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.document import DocumentService
from app.services.vector import VectorService

router = APIRouter()
doc_service = DocumentService()
vector_service = VectorService()

@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads supported")
    
    saved_path = doc_service.save_file(file)
    chunks = doc_service.load_and_split(saved_path)
    vector_service.add_documents(chunks)
    
    return {
        "status": "ok",
        "message": f"{file.filename} uploaded and indexed.",
        "chunks": len(chunks)
    }

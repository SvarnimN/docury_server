import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import Configs

class DocumentService:
    def __init__(
            self, upload_dir=Configs.UPLOAD_DIR,
            chunk_size=Configs.CHUNK_SIZE,
            chunk_overlap=Configs.CHUNK_OVERLAP
        ):
        self.upload_dir = upload_dir
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def save_file(self, upload_file) -> str:
        os.makedirs(Configs.UPLOAD_DIR, exist_ok=True)

        path = os.path.join(self.upload_dir, upload_file.filename)
        file_bytes = upload_file.file.read()

        with open(path, "wb") as f:
            f.write(file_bytes)
        
        return path

    def load_and_split(self, filepath: str):
        loader = PyPDFLoader(filepath)
        doc = loader.load()
        chunks = self.splitter.split_documents(doc)
        
        for d_index, chunk in enumerate(chunks):
            if "source" not in chunk.metadata:
                chunk.metadata["source"] = filepath
            chunk.metadata["chunk_index"] = d_index

        return chunks

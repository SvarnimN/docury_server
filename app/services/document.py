import os
from typing import List
from fastapi import UploadFile
from langchain_docling import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.config import Configs


class DocumentService:
    def __init__(
        self,
        upload_dir=Configs.UPLOAD_DIR,
        chunk_size=Configs.CHUNK_SIZE,
        chunk_overlap=Configs.CHUNK_OVERLAP,
    ):
        self.upload_dir = upload_dir
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    async def save_file(self, upload_file: UploadFile) -> str:
        os.makedirs(self.upload_dir, exist_ok=True)

        path = os.path.join(self.upload_dir, upload_file.filename)

        with open(path, "wb") as f:
            f.write(await upload_file.read())

        return path

    def load_and_split(self, filepath: str) -> List[Document]:
        loader = DoclingLoader(
            file_path=filepath
        )

        docs = loader.load()
        chunks = self.splitter.split_documents(docs)

        for idx, chunk in enumerate(chunks):
            chunk.metadata["source"] = filepath
            chunk.metadata["chunk_index"] = idx

            page_numbers = set()

            dl_meta = getattr(chunk, "dl_meta", None)
            if dl_meta:
                for doc_item in dl_meta.get("doc_items", []):
                    for prov in doc_item.get("prov", []):
                        page_no = prov.get("page_no")
                        if page_no is not None:
                            page_numbers.add(page_no)

            chunk.metadata["page_numbers"] = sorted(page_numbers) if page_numbers else None

        return chunks

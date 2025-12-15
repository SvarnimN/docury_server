import os
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import Configs


class URLService:
    def __init__(
            self,
            chunk_size=Configs.CHUNK_SIZE,
            chunk_overlap=Configs.CHUNK_OVERLAP
        ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def fetch_and_split(self, url: str):
        loader = SeleniumURLLoader(urls=[url])

        docs = loader.load()

        chunks = self.splitter.split_documents(docs)
        
        for idx, chunk in enumerate(chunks):
            chunk.metadata["source"] = url
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["page"] = chunk.metadata["title"]
        
        return chunks

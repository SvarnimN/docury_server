import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents import Document
from app.config import Configs


class VectorService:
    def __init__(self, store_dir: str = Configs.VECTOR_DIR):
        os.makedirs(store_dir, exist_ok=True)

        self.embeddings = FastEmbedEmbeddings(
            model="BAAI/bge-small-en-v1.5"
        )

        self.store_dir = store_dir
        self.index_path = os.path.join(store_dir, "faiss_index")
        self.vector_store: FAISS | None = None

    def load_index(self) -> FAISS | None:
        if os.path.exists(self.index_path):
            self.vector_store = FAISS.load_local(
                self.index_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        return self.vector_store

    def add_documents(self, docs: List[Document]) -> None:
        if not docs:
            return

        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(
                docs,
                self.embeddings,
            )
        else:
            self.vector_store.add_documents(docs)

        self.vector_store.save_local(self.index_path)

    def get_retriever(self, k: int = 3):
        if self.vector_store is None:
            self.load_index()

        return self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": max(k * 3, 10),
            },
        )

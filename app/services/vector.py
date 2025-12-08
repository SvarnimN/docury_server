import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from app.config import Configs

class VectorService:
    def __init__(self, store_dir=Configs.VECTOR_DIR):
        os.makedirs(store_dir, exist_ok=True)
        
        self.embeddings = FastEmbedEmbeddings(model="BAAI/bge-small-zh-v1.5")
        self.store_dir = store_dir
        self.index_path = f"{store_dir}/faiss_index"
        self.vector_store = None

    def load_index(self):
        if os.path.exists(self.index_path):
            self.vector_store = FAISS.load_local(
                self.index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        
        return self.vector_store
    
    def add_documents(self, docs):
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
        else:
            self.vector_store.add_documents(docs)
        
        self.vector_store.save_local(self.index_path)

    def get_retriever(self, k=3):
        if self.vector_store is None:
            self.load_index()
        
        return self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k}
        )


from langchain.vectorstores import FAISS
from langchain.schema import Document
from src.models.embeddings import get_embeddings
from config.settings import VECTOR_STORE_PATH
import os

class DocumentRetriever:
    def __init__(self):
        self.embeddings = get_embeddings()
        self.vector_store = self._initialize_vector_store()

    def _initialize_vector_store(self):
        """Load or create vector store"""
        if os.path.exists(VECTOR_STORE_PATH):
            return FAISS.load_local(VECTOR_STORE_PATH, self.embeddings)
        else:
            return FAISS.from_documents([Document(page_content="")], self.embeddings)

    def update_knowledge_base(self, documents):
        """Add new documents to vector store"""
        self.vector_store.add_documents(documents)
        self.vector_store.save_local(VECTOR_STORE_PATH)

    def retrieve(self, query: str, k: int = 3):
        """Retrieve top k relevant documents"""
        return self.vector_store.similarity_search(query, k=k)

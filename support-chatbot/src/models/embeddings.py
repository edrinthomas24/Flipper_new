from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from config.settings import EMBEDDING_MODEL, OPENAI_API_KEY

def get_embeddings():
    """Initialize embedding model based on configuration"""
    if EMBEDDING_MODEL.startswith("text-embedding"):
        return OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY
        )
    else:
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={'normalize_embeddings': True}
        )

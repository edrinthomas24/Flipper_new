
from decouple import config

# API Keys
OPENAI_API_KEY = config("OPENAI_API_KEY", default="")
HUGGINGFACEHUB_API_TOKEN = config("HUGGINGFACEHUB_API_TOKEN", default="")

# Model Configuration
EMBEDDING_MODEL = "text-embedding-3-small"  # Alternatives: "all-MiniLM-L6-v2"
LLM_MODEL = "gpt-4-1106-preview"            # Alternatives: "HuggingFaceH4/zephyr-7b-beta"

# Paths
VECTOR_STORE_PATH = "data/vector_store/"
KB_DOCUMENTS_PATH = "data/kb_documents/"

# Performance
MAX_TOKENS = 4096
TEMPERATURE = 0.3

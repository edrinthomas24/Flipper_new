from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub
from config.settings import LLM_MODEL, OPENAI_API_KEY, HUGGINGFACEHUB_API_TOKEN, TEMPERATURE

def get_llm():
    """Initialize LLM based on configuration"""
    if LLM_MODEL.startswith("gpt"):
        return ChatOpenAI(
            model=LLM_MODEL,
            temperature=TEMPERATURE,
            openai_api_key=OPENAI_API_KEY
        )
    else:
        return HuggingFaceHub(
            repo_id=LLM_MODEL,
            huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
            model_kwargs={"temperature": TEMPERATURE}
        )

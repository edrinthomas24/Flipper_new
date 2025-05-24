from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.chains.retrieval import DocumentRetriever
from src.chains.response import ResponseGenerator
from src.chains.summary import TicketSummarizer
from src.utils.logger import setup_logger
import os

# Initialize components
logger = setup_logger("support-chatbot")
retriever = DocumentRetriever()
chatbot = ResponseGenerator(retriever.vector_store.as_retriever())
summarizer = TicketSummarizer()

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

class SummaryRequest(BaseModel):
    ticket_text: str

@app.on_event("startup")
async def startup_event():
    """Initialize knowledge base on startup"""
    from src.utils.file_loader import bulk_load_documents
    from config.settings import KB_DOCUMENTS_PATH
    
    if os.path.exists(KB_DOCUMENTS_PATH):
        documents = bulk_load_documents(KB_DOCUMENTS_PATH)
        if documents:
            retriever.update_knowledge_base(documents)
            logger.info(f"Loaded {len(documents)} documents into knowledge base")

@app.post("/query")
async def handle_query(request: QueryRequest):
    """Endpoint for answering support questions"""
    try:
        response = chatbot.generate_response(request.question)
        logger.info(f"Query: {request.question} | Sources: {response['sources']}")
        return response
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize")
async def summarize_ticket(request: SummaryRequest):
    """Endpoint for ticket summarization"""
    try:
        summary = summarizer.summarize(request.ticket_text)
        logger.info(f"Summarized ticket: {request.ticket_text[:50]}...")
        return {"summary": summary}
    except Exception as e:
        logger.error(f"Summary failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

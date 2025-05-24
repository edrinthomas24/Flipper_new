from langchain.chains import RetrievalQA
from src.models.llm import get_llm
from config.prompts import RESPONSE_PROMPT

class ResponseGenerator:
    def __init__(self, retriever):
        self.llm = get_llm()
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": RESPONSE_PROMPT},
            return_source_documents=True
        )

    def generate_response(self, question: str):
        """Generate response with sources"""
        result = self.qa_chain({"query": question})
        return {
            "answer": result["result"],
            "sources": [doc.metadata.get("source", "") for doc in result["source_documents"]]
        }

from langchain.chains import LLMChain
from src.models.llm import get_llm
from config.prompts import SUMMARY_PROMPT

class TicketSummarizer:
    def __init__(self):
        self.llm = get_llm()
        self.chain = LLMChain(
            llm=self.llm,
            prompt=SUMMARY_PROMPT
        )

    def summarize(self, ticket_text: str):
        """Generate concise ticket summary"""
        return self.chain.run(ticket_text=ticket_text)

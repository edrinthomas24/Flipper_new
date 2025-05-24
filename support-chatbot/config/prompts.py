from langchain.prompts import PromptTemplate

# Response Generation Prompt
RESPONSE_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    As an internal support assistant, answer the question using ONLY the provided context.
    If unsure, say "I don't have information about that in my knowledge base."

    Context: {context}
    
    Question: {question}
    
    Answer (concise, markdown-supported):
    """
)

# Summary Prompt
SUMMARY_PROMPT = PromptTemplate(
    input_variables=["ticket_text"],
    template="""
    Summarize this support ticket for agent review:
    {ticket_text}
    
    Include:
    - Key issue (1 sentence)
    - Urgency (Low/Medium/High)
    - Suggested knowledge base section
    - Critical details (IDs, contacts)
    """
)

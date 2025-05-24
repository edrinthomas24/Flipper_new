from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document

def load_document(file_path: str) -> List[Document]:
    """Load single document based on file type"""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".csv"):
        loader = CSVLoader(file_path)
    else:
        loader = TextLoader(file_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return loader.load_and_split(text_splitter)

def bulk_load_documents(directory: str) -> List[Document]:
    """Load all supported documents from a directory"""
    from glob import glob
    documents = []
    
    for ext in ["*.pdf", "*.txt", "*.docx", "*.csv"]:
        for file_path in glob(f"{directory}/{ext}"):
            try:
                documents.extend(load_document(file_path))
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
    
    return documents

import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config import settings


def get_retriever():
    """
    Load the FAISS index from disk and return a LangChain retriever.
    The embedding model here MUST match the one used in ingest.py.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )
    vectorstore = FAISS.load_local(
        folder_path=settings.faiss_index_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever(
        search_type='similarity',
        search_kwargs={'k': settings.top_k}
    )

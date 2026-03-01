import os
import sys
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config import settings



def ingest(file_path: str) -> None:
    """
    Load a PDF or TXT file, chunk it, embed it, and save to FAISS.
    The embedding model MUST match app/services/retriever.py.
    """
    print(f'Loading: {file_path}')

    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path, encoding='utf-8')
    else:
        raise ValueError(f'Unsupported file type: {file_path}. Use .pdf or .txt')

    docs = loader.load()


    for d in docs:
        d.page_content = (d.page_content.encode('utf-8', 'ignore').decode('utf-8', 'ignore'))
        
        d.metadata['source'] = os.path.basename(file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    chunks = splitter.split_documents(docs)

    print(f'Chunked into {len(chunks)} pieces')

    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(settings.faiss_index_path, exist_ok=True)
    vectorstore.save_local(settings.faiss_index_path)

    print(f'Ingested {len(chunks)} chunks from {os.path.basename(file_path)}')
    print(f'Saved to {settings.faiss_index_path}/')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python ingest.py <path_to_file.pdf_or_.txt>')
        sys.exit(1)
    ingest(sys.argv[1])

"""
Naive RAG baseline — the simplest possible RAG system.
No CRAG grading. No refinement. No web fallback.
This is what we are comparing against.
"""
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.services.retriever import get_retriever
from app.config import settings

sonnet = ChatAnthropic(model=settings.sonnet_model, temperature=0)

answer_prompt = ChatPromptTemplate.from_messages([
    ('system', 'Answer ONLY using the provided context. If unsure, say so.'),
    ('human', 'Context:\n{context}\n\nQuestion: {question}')])

chain = answer_prompt | sonnet | StrOutputParser()


def naive_rag_answer(question: str) -> dict:
    """
    Retrieve top-k chunks and pass directly to LLM.
    Returns dict with answer and contexts for RAGAS scoring.
    """
    retriever = get_retriever()
    docs      = retriever.invoke(question)
    context   = '\n\n'.join(d.page_content for d in docs)
    answer    = chain.invoke({'context': context, 'question': question})
    return {
        'answer':   answer,
        'contexts': [d.page_content for d in docs]}

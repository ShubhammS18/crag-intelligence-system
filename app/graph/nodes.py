import re
from typing import List
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from app.config import settings
from app.graph.state import GraphState
from app.models import DocEvalScore, WebQuery, KeepOrDrop
from app.services.retriever import get_retriever
from app.services.web_search import search_web

# Models
haiku  = ChatAnthropic(model=settings.haiku_model,  temperature=0)
sonnet = ChatAnthropic(model=settings.sonnet_model, temperature=0)



# NODE 1: retrieve
def retrieve(state: GraphState) -> dict:
    retriever = get_retriever()
    docs = retriever.invoke(state['question'])
    return {'docs': docs}


# NODE 2: eval_each_doc
# Scores every retrieved chunk 0.0-1.0 using Haiku.
# Assigns verdict: CORRECT / AMBIGUOUS / INCORRECT
# Populates good_docs with any chunk scoring > lower_th
doc_eval_prompt = ChatPromptTemplate.from_messages([
    ('system',
        'You are a strict retrieval evaluator for RAG.\n'
        'You will be given ONE retrieved chunk and a question.\n'
        'Return a relevance score in [0.0, 1.0].\n'
        '- 1.0: chunk alone is sufficient to answer fully/mostly\n'
        '- 0.0: chunk is irrelevant\n'
        'Be conservative with high scores.\n'
        'Also return a short reason.\n'
        'Output JSON only.'),
    ('human', 'Question: {question}\n\nChunk:\n{chunk}'),
])

doc_eval_chain = doc_eval_prompt | haiku.with_structured_output(DocEvalScore)


def eval_each_doc(state: GraphState) -> dict:
    q      = state['question']
    scores = []
    good   = []

    for doc in state['docs']:
        out = doc_eval_chain.invoke({
            'question': q,
            'chunk':    doc.page_content
        })
        scores.append(out.score)
        
        
        if out.score > settings.lower_th:
            good.append(doc)

    # CORRECT: at least one chunk scored above upper_th
    if any(s > settings.upper_th for s in scores):
        return {
            'good_docs' : good,
            'verdict' : 'CORRECT',
            'reason' : f'At least one chunk scored > {settings.upper_th}.',
        }

    # INCORRECT: every chunk scored below lower_th
    if scores and all(s < settings.lower_th for s in scores):
        return {
            'good_docs' : [],
            'verdict' : 'INCORRECT',
            'reason': f'All chunks scored < {settings.lower_th}.',
        }
    
    # AMBIGUOUS: in between
    return {
        'good_docs': good,
        'verdict' : 'AMBIGUOUS',
        'reason' : f'No chunk > {settings.upper_th}, but not all < {settings.lower_th}.',
    }



# NODE 3: rewrite_query
# Only called on AMBIGUOUS or INCORRECT path.
# Converts the user question into a short keyword web query.
rewrite_prompt = ChatPromptTemplate.from_messages([
    ('system',
        'Rewrite the user question into a web search query of keywords.\n'
        'Rules:\n'
        '- Keep it short (6-14 words).\n'
        '- If question implies recency, add (last 30 days).\n'
        '- Do NOT answer the question.\n'
        '- Return JSON with a single key: query'),
    ('human', 'Question: {question}'),
])

rewrite_chain = rewrite_prompt | haiku.with_structured_output(WebQuery)


def rewrite_query(state: GraphState) -> dict:
    out = rewrite_chain.invoke({'question': state['question']})
    return {'web_query': out.query}


# NODE 4: web_search
# Uses the rewritten web_query to fetch Tavily results.
def web_search(state: GraphState) -> dict:
    q = state.get('web_query') or state['question']
    results = search_web(q)
    return {'web_docs': results}


# NODE 5: refine
# Called on ALL paths.
# Assembles the right docs based on verdict
# decomposes to sentences, LLM judges each sentence keep/drop.
filter_prompt = ChatPromptTemplate.from_messages([
    ('system',
        'You are a strict relevance filter.\n'
        'Return keep=true ONLY if the sentence directly helps answer the question.\n'
        'Use ONLY the sentence itself. Output JSON only.'),
    ('human', 'Question: {question}\n\nSentence:\n{sentence}'),
])

filter_chain = filter_prompt | haiku.with_structured_output(KeepOrDrop)


def _decompose_to_sentences(text: str) -> List[str]:
    """Split text into individual sentences. Min length 20 chars."""
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def refine(state: GraphState) -> dict:
    q = state['question']
    verdict = state.get('verdict', 'INCORRECT')

    # Choose doc source based on CRAG verdict
    if verdict == 'CORRECT':
        docs_to_use = state['good_docs']   
    elif verdict == 'INCORRECT':
        docs_to_use = state.get('web_docs', [])
    else:  # AMBIGUOUS
        docs_to_use = state['good_docs'] + state.get('web_docs', [])  # both

    context = '\n\n'.join(d.page_content for d in docs_to_use).strip()
    strips  = _decompose_to_sentences(context)

    kept = []
    for sentence in strips:
        result = filter_chain.invoke({'question': q, 'sentence': sentence})
        if result.keep:
            kept.append(sentence)

    refined_context = '\n'.join(kept).strip()

    return {
        'strips' : strips,
        'kept_strips' : kept,
        'refined_context' : refined_context,
    }


# NODE 6: generate
answer_prompt = ChatPromptTemplate.from_messages([
    ('system',
        'You are a corporate intelligence analyst.\n'
        'Answer ONLY using the provided context.\n'
        "If the context is empty or insufficient, say: I don't know."),
    ('human', 'Question: {question}\n\nContext:\n{context}'),
])


def generate(state: GraphState) -> dict:
    out = (answer_prompt | sonnet).invoke({
        'question' : state['question'],
        'context' : state['refined_context']
    })
    return {'answer': out.content}


from fastapi import FastAPI, HTTPException
from app.models import AskRequest, AskResponse
from app.graph.graph import compiled_graph
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('crag-api')

app = FastAPI(
    title='CRAG Intelligence API',
    description='Corrective RAG with knowledge refinement and LangSmith observability',
    version='3.0')


@app.get('/health')
def health():
    return {'status': 'ok', 'version': '3.0'}


@app.post('/ask', response_model=AskResponse)
def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail='Question cannot be empty')

    logger.info(f'Received question: {request.question[:80]}')

    # Seed all state keys with safe empty defaults
    # LangGraph requires every TypedDict key to be present at invoke
    result = compiled_graph.invoke({
        'question': request.question,
        'docs':  [],
        'good_docs':  [],
        'verdict':  '',
        'reason':  '',
        'web_query':  '',
        'web_docs':  [],
        'strips':  [],
        'kept_strips':   [],
        'refined_context':  '',
        'answer':  '',
    })

    logger.info(f'Verdict: {result["verdict"]} | Kept strips: {len(result.get("kept_strips", []))}')

    return AskResponse(
        answer=result['answer'],
        verdict=result['verdict'],
        reason=result['reason'],
        kept_strips=result.get('kept_strips', []),
    )

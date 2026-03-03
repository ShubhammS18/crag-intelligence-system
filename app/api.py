import time
from fastapi import FastAPI, HTTPException
from app.models import AskRequest, AskResponse
from app.graph.graph import compiled_graph
from app.prompts import PROMPT_VERSION
from app.request_logger import log_request, compute_metrics
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('crag-api')

app = FastAPI(
    title='CRAG Intelligence API',
    description='Corrective RAG with knowledge refinement and LangSmith observability',
    version='3.1')


# Anthropic pricing as of March 2026 
# Source: https://www.anthropic.com/pricing
# Haiku  input:  $0.80  per 1M tokens
# Haiku  output: $4.00  per 1M tokens
# Sonnet input:  $3.00  per 1M tokens
# Sonnet output: $15.00 per 1M tokens
HAIKU_IN    = 0.80  / 1_000_000
HAIKU_OUT   = 4.00  / 1_000_000
SONNET_IN   = 3.00  / 1_000_000
SONNET_OUT  = 15.00 / 1_000_000


def _estimate_cost(verdict: str, n_kept_strips: int) -> float:
    """
    Estimate request cost from verdict and number of kept strips.

    Token estimates per path:
    - eval_each_doc: 4 chunks x ~300 tokens in + ~50 out = Haiku
    - refine filter:  n_sentences x ~150 tokens in + ~20 out = Haiku
    - rewrite_query:  ~100 in + ~30 out = Haiku (non-correct only)
    - generate:       ~1000 in + ~300 out = Sonnet

    This is an estimate. LangSmith has exact token counts per node.
    """
    # Base: eval_each_doc (4 chunks)
    cost = (4 * 300 * HAIKU_IN) + (4 * 50 * HAIKU_OUT)

    # Refine filter — estimate 3 sentences per kept strip on average
    n_sentences = max(n_kept_strips * 3, 5)
    cost += (n_sentences * 150 * HAIKU_IN) + (n_sentences * 20 * HAIKU_OUT)

    # Rewrite query — only on non-correct paths
    if verdict in ('AMBIGUOUS', 'INCORRECT'):
        cost += (100 * HAIKU_IN) + (30 * HAIKU_OUT)

    # Generate — always Sonnet
    cost += (1000 * SONNET_IN) + (300 * SONNET_OUT)

    return round(cost, 6)


@app.get('/health')
def health():
    return {
        'status':  'ok',
        'version': '3.1',
        'prompt_version': PROMPT_VERSION }


@app.get('/metrics')
def metrics():
    """
    Returns aggregated metrics computed from request_log.jsonl.
    Covers: failure rate, citation coverage, latency P50/P95,
    avg cost per request, verdict distribution.
    """
    return compute_metrics()


@app.post('/ask', response_model=AskResponse)
def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail='Question cannot be empty')

    logger.info(f'question="{request.question[:80]}" prompt_version={PROMPT_VERSION}')

    start = time.perf_counter()
    error   = None
    verdict = 'INCORRECT'
    n_kept  = 0
    result  = {}

    try:
        result = compiled_graph.invoke({
            'question':  request.question,
            'docs':      [],
            'good_docs': [],
            'verdict':   '',
            'reason':    '',
            'web_query': '',
            'web_docs':  [],
            'strips':    [],
            'kept_strips':  [],
            'refined_context': '',
            'answer':    '' })
        verdict = result.get('verdict', 'INCORRECT')
        n_kept  = len(result.get('kept_strips', []))
    
    except Exception as e:
        error = e
        logger.error(f'Graph execution failed: {e}')
    
    finally:
        latency_ms = int((time.perf_counter() - start) * 1000)
        cost = _estimate_cost(verdict, n_kept)
        log_request(
            question=request.question,
            verdict=verdict,
            latency_ms=latency_ms,
            kept_strips_count=n_kept,
            estimated_cost_usd=cost,
            prompt_version=PROMPT_VERSION,
            error=error)

    if error:
        raise HTTPException(status_code=500, detail=str(error))

    logger.info(
        f'verdict={verdict} kept={n_kept} '
        f'latency_ms={latency_ms} cost_usd={cost}')

    return AskResponse(
        answer=result['answer'],
        verdict=verdict,
        reason=result['reason'],
        kept_strips=result.get('kept_strips', []),
        latency_ms=latency_ms,
        estimated_cost_usd=cost)
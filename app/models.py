from pydantic import BaseModel
from typing import List


# FastAPI request/response models
class AskRequest(BaseModel):
    question : str


class AskResponse(BaseModel):
    answer : str
    verdict : str       # CORRECT | AMBIGUOUS | INCORRECT
    reason : str          
    kept_strips : List[str]
    latency_ms: int     # total wall-clock time for this request
    estimated_cost_usd: float    # rough cost estimate — LangSmith has exact



# Internal structured output schemas for LLM calls
class DocEvalScore(BaseModel):
    """
    Output schema for eval_each_doc node.
    LLM scores each chunk 0.0-1.0 with a short reason.
    """
    score : float
    reason : str


class WebQuery(BaseModel):
    """
    Output schema for rewrite_query node.
    LLM converts the question into a keyword web search string.
    """
    query : str


class KeepOrDrop(BaseModel):
    """
    Output schema for sentence-level filter in refine node.
    LLM decides whether each sentence directly answers the question.
    """
    keep : bool

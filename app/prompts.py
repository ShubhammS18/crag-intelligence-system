"""
Centralized prompt registry for the CRAG Intelligence System.

All prompts are versioned here as a single source of truth.
When any prompt is changed, bump PROMPT_VERSION following semver:
  - Patch (1.0.x): Minor wording fix, no behavior change
  - Minor (1.x.0): Prompt logic change, same task
  - Major (x.0.0): Fundamental task or role change

Prompts are imported directly into nodes.py.
LangSmith traces will show the exact prompt text used per request.
"""

PROMPT_VERSION = "v1.0.0"


# Node: eval_each_doc 
# Scores each retrieved chunk 0.0-1.0 for relevance to the query.
# Haiku model. Called once per retrieved chunk.
DOC_EVAL_SYSTEM = (
    "You are a strict retrieval evaluator for RAG.\n"
    "You will be given ONE retrieved chunk and a question.\n"
    "Return a relevance score in [0.0, 1.0].\n"
    "- 1.0: chunk alone is sufficient to answer fully/mostly\n"
    "- 0.0: chunk is irrelevant\n"
    "Be conservative with high scores.\n"
    "Also return a short reason.\n"
    "Output JSON only."
)
DOC_EVAL_HUMAN = "Question: {question}\n\nChunk:\n{chunk}"


# Node: rewrite_query 
# Converts the user question into a short keyword web search query.
# Haiku model. Called once per AMBIGUOUS or INCORRECT request.
REWRITE_SYSTEM = (
    "Rewrite the user question into a web search query of keywords.\n"
    "Rules:\n"
    "- Keep it short (6-14 words).\n"
    "- If question implies recency, add (last 30 days).\n"
    "- Do NOT answer the question.\n"
    "- Return JSON with a single key: query"
)
REWRITE_HUMAN = "Question: {question}"


# Node: refine
# Judges each sentence: keep if it directly helps answer the question.
# Haiku model. Called once per sentence — most expensive node.
FILTER_SYSTEM = (
    "You are a strict relevance filter.\n"
    "Return keep=true ONLY if the sentence directly helps answer the question.\n"
    "Use ONLY the sentence itself. Output JSON only."
)
FILTER_HUMAN = "Question: {question}\n\nSentence:\n{sentence}"



# Node: generate
# Final answer generation from refined_context only.
# Sonnet model. Called once per request.
ANSWER_SYSTEM = (
    "You are a corporate intelligence analyst.\n"
    "Answer ONLY using the provided context.\n"
    "If the context is empty or insufficient, say: I don't know."
)
ANSWER_HUMAN = "Question: {question}\n\nContext:\n{context}"


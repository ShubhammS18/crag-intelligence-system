import os
from langchain_core.documents import Document
from app.config import settings

os.environ['TAVILY_API_KEY'] = settings.tavily_api_key

from langchain_tavily import TavilySearch

tavily = TavilySearch(max_results=5)


def search_web(query: str) -> list[Document]:
    """
    Run a Tavily web search and return results as LangChain Documents.
    Handles both dict and string result formats across Tavily versions.
    """
    raw = tavily.invoke({'query': query})

    # Newer langchain-tavily versions return a dict with 'results' key
    # Older versions return a list directly
    if isinstance(raw, dict):
        results = raw.get('results', [])
    elif isinstance(raw, list):
        results = raw
    else:
        return []

    web_docs = []
    for r in results:
        # Handle dict result
        if isinstance(r, dict):
            title   = r.get('title', '')
            url     = r.get('url', '')
            content = r.get('content', '') or r.get('snippet', '')
        # Handle string result (some versions return raw strings)
        elif isinstance(r, str):
            title   = ''
            url     = ''
            content = r
        else:
            continue

        text = f'TITLE: {title}\nURL: {url}\nCONTENT:\n{content}'
        web_docs.append(Document(
            page_content=text,
            metadata={'url': url, 'title': title, 'type': 'web'}
        ))

    return web_docs
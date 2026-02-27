import os
from langchain_core.documents import Document
from app.config import settings

os.environ['TAVILY_API_KEY'] = settings.tavily_api_key

from langchain_tavily import TavilySearch

tavily = TavilySearch(max_results=5)


def search_web(query: str) -> list[Document]:
    """
    Run a Tavily web search and return results as LangChain Documents.
    Each Document has page_content = title + url + content.
    """
    results = tavily.invoke({'query': query})
    web_docs = []
    for r in results or []:
        title   = r.get('title', '')
        url     = r.get('url', '')
        content = r.get('content', '') or r.get('snippet', '')
        text    = f'TITLE: {title}\nURL: {url}\nCONTENT:\n{content}'
        web_docs.append(Document(
            page_content=text,
            metadata={'url': url, 'title': title, 'type': 'web'}
        ))
    return web_docs

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API keys 
    anthropic_api_key: str
    tavily_api_key:    str

    # LangSmith observability
    langchain_tracing_v2: str = 'true'
    langchain_endpoint:   str = 'https://api.smith.langchain.com'
    langchain_api_key:    str = ''
    langchain_project:    str = 'crag-intelligence-system'

    # Model names 
    haiku_model:  str = 'claude-haiku-3-5-20241022'
    sonnet_model: str = 'claude-sonnet-3-5-20241022'

    # Retrieval 
    top_k:         int   = 4
    chunk_size:    int   = 900
    chunk_overlap: int   = 150

    # CRAG thresholds
    upper_th: float = 0.7
    lower_th: float = 0.3

    # Storage 
    faiss_index_path: str = 'vector_store'

    class Config:
        env_file = '.env'


settings = Settings()

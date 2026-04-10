"""应用配置管理"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # 应用
    app_env: str = "dev"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # DeepSeek LLM
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    # Embedding
    embedding_provider: str = "online"  # online | local
    embedding_model: str = "BAAI/bge-m3"
    embedding_api_key: str = ""
    embedding_api_base: str = "https://api.siliconflow.cn/v1"

    # Reranker
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_api_key: str = ""
    reranker_api_base: str = "https://api.siliconflow.cn/v1"

    # FAISS
    faiss_index_dir: str = "./data/faiss_index"

    # 数据库
    db_url: str = "sqlite+aiosqlite:///./data/app.db"

    # RAG 参数
    retrieval_top_k: int = 5
    rerank_top_n: int = 3
    rrf_bm25_weight: float = 0.7
    rrf_dense_weight: float = 0.3
    history_max_rounds: int = 5
    history_max_tokens: int = 2000

    # 默认实验分组
    default_variant: str = "G3"

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent.parent

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()

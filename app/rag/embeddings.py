"""Embedding 管理 - 支持在线(Silicon Flow)和本地两种模式"""

from langchain_openai import OpenAIEmbeddings
from app.core.config import settings
from app.core.logging import logger


def get_embeddings() -> OpenAIEmbeddings:
    """根据配置获取 Embedding 实例"""
    if settings.embedding_provider == "online":
        logger.info(f"Using online embedding: {settings.embedding_model} via {settings.embedding_api_base}")
        return OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.embedding_api_key,
            openai_api_base=settings.embedding_api_base,
        )
    else:
        # 本地模式：使用 HuggingFace sentence-transformers
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            logger.info(f"Using local embedding: {settings.embedding_model}")
            return HuggingFaceEmbeddings(model_name=settings.embedding_model)
        except ImportError:
            raise RuntimeError(
                "本地 Embedding 需要安装 sentence-transformers: pip install sentence-transformers torch"
            )

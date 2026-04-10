"""BM25 索引共享单例，供 BM25/Hybrid 策略复用"""

from __future__ import annotations

import jieba
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from app.rag.retriever import get_vectorstore

_docs_cache: list[Document] | None = None
_bm25_cache: BM25Okapi | None = None


def get_bm25_index() -> tuple[list[Document], BM25Okapi]:
    """获取 BM25 索引（懒加载单例），同时返回文档列表"""
    global _docs_cache, _bm25_cache
    if _bm25_cache is None:
        vs = get_vectorstore()
        _docs_cache = list(vs.docstore._dict.values())
        tokenized = [list(jieba.cut(doc.page_content)) for doc in _docs_cache]
        _bm25_cache = BM25Okapi(tokenized)
    return _docs_cache, _bm25_cache

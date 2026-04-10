"""策略 A: BM25 稀疏检索"""

import time

import jieba

from app.rag.strategies import register
from app.rag.strategies.base import BaseRetrieverStrategy, RetrievalResult
from app.rag.strategies._bm25_index import get_bm25_index
from langchain_core.documents import Document


@register("bm25")
class BM25Strategy(BaseRetrieverStrategy):

    @property
    def name(self) -> str:
        return "bm25"

    @property
    def display_name(self) -> str:
        return "A: BM25"

    async def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        start = time.perf_counter()

        docs, bm25 = get_bm25_index()
        tokenized_query = list(jieba.cut(query))
        scores = bm25.get_scores(tokenized_query)

        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results = []
        for idx in top_indices:
            doc = docs[idx]
            results.append(
                Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "score": float(scores[idx])},
                )
            )

        elapsed = (time.perf_counter() - start) * 1000
        return RetrievalResult(
            documents=results,
            latency_ms=elapsed,
            trace={"method": "bm25", "corpus_size": len(docs)},
        )

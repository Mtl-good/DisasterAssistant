"""策略 B: Dense 向量检索 (FAISS)"""

import time

from app.rag.strategies import register
from app.rag.strategies.base import BaseRetrieverStrategy, RetrievalResult
from app.rag.retriever import get_vectorstore


@register("dense")
class DenseStrategy(BaseRetrieverStrategy):

    @property
    def name(self) -> str:
        return "dense"

    @property
    def display_name(self) -> str:
        return "B: Dense (FAISS)"

    async def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        start = time.perf_counter()

        vs = get_vectorstore()
        docs_with_scores = vs.similarity_search_with_score(query, k=top_k)

        results = []
        for doc, score in docs_with_scores:
            doc.metadata["score"] = float(score)
            results.append(doc)

        elapsed = (time.perf_counter() - start) * 1000
        return RetrievalResult(
            documents=results,
            latency_ms=elapsed,
            trace={"method": "dense_faiss"},
        )

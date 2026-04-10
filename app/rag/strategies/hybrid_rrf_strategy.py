"""策略 C: Hybrid RRF (BM25 + Dense 倒数排名融合)"""

import time

import jieba
from langchain_core.documents import Document

from app.core.config import settings
from app.rag.strategies import register
from app.rag.strategies.base import BaseRetrieverStrategy, RetrievalResult
from app.rag.strategies._bm25_index import get_bm25_index
from app.rag.retriever import get_vectorstore


@register("hybrid_rrf")
class HybridRRFStrategy(BaseRetrieverStrategy):

    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k
        self.bm25_weight = settings.rrf_bm25_weight
        self.dense_weight = settings.rrf_dense_weight

    @property
    def name(self) -> str:
        return "hybrid_rrf"

    @property
    def display_name(self) -> str:
        return "C: Hybrid (RRF)"

    async def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        start = time.perf_counter()

        expand_k = top_k * 6

        # --- BM25 路 ---
        docs, bm25 = get_bm25_index()
        tokenized_query = list(jieba.cut(query))
        bm25_scores = bm25.get_scores(tokenized_query)
        bm25_top = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[:expand_k]

        # --- Dense 路 ---
        vs = get_vectorstore()
        dense_results = vs.similarity_search_with_score(query, k=expand_k)

        # --- RRF 加权融合 ---
        doc_map: dict[str, Document] = {}
        rrf_scores: dict[str, float] = {}

        for rank, idx in enumerate(bm25_top):
            doc = docs[idx]
            key = doc.page_content
            doc_map[key] = doc
            rrf_scores[key] = rrf_scores.get(key, 0) + self.bm25_weight / (self.rrf_k + rank + 1)

        for rank, (doc, _score) in enumerate(dense_results):
            key = doc.page_content
            doc_map[key] = doc
            rrf_scores[key] = rrf_scores.get(key, 0) + self.dense_weight / (self.rrf_k + rank + 1)

        sorted_keys = sorted(
            rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True
        )[:top_k]

        results = []
        for key in sorted_keys:
            doc = doc_map[key]
            results.append(
                Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "score": rrf_scores[key]},
                )
            )

        elapsed = (time.perf_counter() - start) * 1000
        return RetrievalResult(
            documents=results,
            latency_ms=elapsed,
            trace={
                "method": "hybrid_rrf",
                "rrf_k": self.rrf_k,
                "bm25_weight": self.bm25_weight,
                "dense_weight": self.dense_weight,
                "bm25_candidates": len(bm25_top),
                "dense_candidates": len(dense_results),
            },
        )

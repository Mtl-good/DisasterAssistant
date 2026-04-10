"""策略 D: Hybrid RRF + Reranker (混合检索 + BGE 精排)"""

import time

import jieba
from langchain_core.documents import Document

from app.core.config import settings
from app.rag.strategies import register
from app.rag.strategies.base import BaseRetrieverStrategy, RetrievalResult
from app.rag.strategies._bm25_index import get_bm25_index
from app.rag.retriever import get_vectorstore
from app.rag.reranker import rerank


@register("hybrid_rrf_rerank")
class HybridRRFRerankStrategy(BaseRetrieverStrategy):

    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k
        self.bm25_weight = settings.rrf_bm25_weight
        self.dense_weight = settings.rrf_dense_weight

    @property
    def name(self) -> str:
        return "hybrid_rrf_rerank"

    @property
    def display_name(self) -> str:
        return "D: Hybrid+Rerank"

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

        # 取 top candidates 送入 Reranker（扩大候选池）
        rerank_k = max(top_k * 4, 20)
        sorted_keys = sorted(
            rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True
        )[:rerank_k]

        candidates = []
        for key in sorted_keys:
            doc = doc_map[key]
            candidates.append(
                Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "rrf_score": rrf_scores[key]},
                )
            )

        # --- Rerank 精排 ---
        reranked = await rerank(query, candidates, top_n=top_k)

        elapsed = (time.perf_counter() - start) * 1000
        return RetrievalResult(
            documents=reranked,
            latency_ms=elapsed,
            trace={
                "method": "hybrid_rrf_rerank",
                "rrf_k": self.rrf_k,
                "bm25_weight": self.bm25_weight,
                "dense_weight": self.dense_weight,
                "candidates_before_rerank": len(candidates),
            },
        )

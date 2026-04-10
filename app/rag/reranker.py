"""Reranker - 通过 Silicon Flow API 调用 bge-reranker-v2-m3"""

import httpx
from langchain_core.documents import Document

from app.core.config import settings
from app.core.logging import logger


async def rerank(query: str, docs: list[Document], top_n: int | None = None) -> list[Document]:
    """使用 BGE Reranker 对检索结果重排序，返回 top_n 个"""
    n = top_n or settings.rerank_top_n
    if not docs:
        return []

    pairs = [doc.page_content for doc in docs]

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{settings.reranker_api_base}/rerank",
                headers={
                    "Authorization": f"Bearer {settings.reranker_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.reranker_model,
                    "query": query,
                    "documents": pairs,
                    "top_n": n,
                    "return_documents": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        results_indices = data.get("results", [])
        reranked = []
        for item in results_indices:
            idx = item["index"]
            doc = docs[idx]
            doc.metadata["rerank_score"] = item.get("relevance_score", 0)
            reranked.append(doc)

        logger.info(f"Rerank: {len(docs)} -> {len(reranked)} docs")
        return reranked

    except Exception as e:
        logger.warning(f"Reranker failed, falling back to original order: {e}")
        return docs[:n]

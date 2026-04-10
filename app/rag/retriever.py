"""向量检索器 - FAISS"""

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.core.config import settings
from app.core.logging import logger
from app.rag.embeddings import get_embeddings


_vectorstore: FAISS | None = None


def get_vectorstore() -> FAISS:
    """获取 FAISS 向量库实例（懒加载单例）"""
    global _vectorstore
    if _vectorstore is None:
        index_path = Path(settings.faiss_index_dir)
        if not index_path.exists():
            raise RuntimeError(
                f"FAISS 索引不存在: {index_path}，请先运行 python scripts/build_index.py 构建索引"
            )
        embeddings = get_embeddings()
        _vectorstore = FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info(f"FAISS index loaded from {index_path}")
    return _vectorstore


def build_vectorstore(docs: list[Document]) -> FAISS:
    """从文档列表构建 FAISS 索引并持久化"""
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    index_path = Path(settings.faiss_index_dir)
    index_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_path))
    logger.info(f"FAISS index saved to {index_path} ({len(docs)} docs)")
    return vectorstore


def retrieve(query: str, top_k: int | None = None) -> list[Document]:
    """向量检索 top_k 个最相关文档"""
    k = top_k or settings.retrieval_top_k
    vs = get_vectorstore()
    docs = vs.similarity_search_with_score(query, k=k)
    results = []
    for doc, score in docs:
        doc.metadata["score"] = float(score)
        results.append(doc)
    return results

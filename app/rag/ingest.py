"""索引构建入口"""

from pathlib import Path

from app.core.config import settings
from app.core.logging import logger
from app.rag.splitter import split_handbook
from app.rag.retriever import build_vectorstore


def ingest_handbook(handbook_path: str | None = None) -> int:
    """读取手册、切分、构建 FAISS 索引"""
    path = Path(handbook_path) if handbook_path else settings.project_root / "data" / "raw" / "EarthQuakeHandBook.md"

    if not path.exists():
        raise FileNotFoundError(f"手册文件不存在: {path}")

    text = path.read_text(encoding="utf-8")
    logger.info(f"读取手册: {path} ({len(text)} chars)")

    docs = split_handbook(text)
    build_vectorstore(docs)

    return len(docs)

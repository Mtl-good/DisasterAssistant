"""检索策略抽象基类"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from langchain_core.documents import Document


@dataclass
class RetrievalResult:
    """检索结果"""

    documents: list[Document]
    latency_ms: float = 0.0
    trace: dict = field(default_factory=dict)


class BaseRetrieverStrategy(ABC):
    """检索策略抽象基类"""

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """执行检索，返回排序后的文档列表"""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """策略唯一标识"""
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        """策略展示名称"""
        ...

"""策略注册表"""

from __future__ import annotations

from app.rag.strategies.base import BaseRetrieverStrategy, RetrievalResult  # noqa: F401

_REGISTRY: dict[str, type[BaseRetrieverStrategy]] = {}


def register(strategy_id: str):
    """装饰器：注册策略类"""

    def wrapper(cls):
        _REGISTRY[strategy_id] = cls
        return cls

    return wrapper


def get_strategy(strategy_id: str, **kwargs) -> BaseRetrieverStrategy:
    """根据 ID 获取策略实例"""
    if strategy_id not in _REGISTRY:
        raise ValueError(
            f"Unknown strategy: {strategy_id}, available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[strategy_id](**kwargs)


def list_strategies() -> list[str]:
    """列出所有已注册策略 ID"""
    return list(_REGISTRY.keys())


# 导入策略模块以触发注册
from app.rag.strategies import bm25_strategy as _bm25  # noqa: E402, F401
from app.rag.strategies import dense_strategy as _dense  # noqa: E402, F401
from app.rag.strategies import hybrid_rrf_strategy as _hybrid  # noqa: E402, F401
from app.rag.strategies import hybrid_rrf_rerank_strategy as _hybrid_rr  # noqa: E402, F401

"""聊天服务 - 编排 RAG 流程"""

import json
import time
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import logger
from app.services.session_service import add_message, get_recent_history, get_session, rename_session
from app.rag.chain import build_chain


async def chat_stream(
    db: AsyncSession,
    session_id: str,
    query: str,
    variant: str | None = None,
) -> AsyncGenerator[str, None]:
    """处理用户提问，SSE 流式返回"""
    variant = variant or settings.default_variant
    start_time = time.time()

    # 校验会话存在
    session = await get_session(db, session_id)
    if not session:
        yield _sse("error", {"code": "SESSION_NOT_FOUND", "message": "会话不存在"})
        return

    # 保存用户消息
    await add_message(db, session_id, "user", query)

    # 自动生成会话标题（首条消息时）
    if session.title == "新会话":
        auto_title = query[:20] + ("..." if len(query) > 20 else "")
        await rename_session(db, session_id, auto_title)

    # 获取历史上下文
    history = await get_recent_history(db, session_id, settings.history_max_rounds)
    history_pairs = []
    for msg in history:
        if msg.role == "user":
            history_pairs.append(("user", msg.content))
        elif msg.role == "assistant":
            history_pairs.append(("assistant", msg.content))

    # 构建 RAG 链并执行
    try:
        chain = build_chain(variant)
        full_text = ""

        async for chunk in chain.astream(query, history_pairs):
            if isinstance(chunk, str):
                full_text += chunk
                yield _sse("token", {"text": chunk})

        latency_ms = int((time.time() - start_time) * 1000)
        yield _sse("meta", {"latency_ms": latency_ms, "variant": variant})

        # 保存 assistant 消息
        await add_message(
            db,
            session_id,
            "assistant",
            full_text,
            metadata_json={
                "latency_ms": latency_ms,
                "variant": variant,
            },
        )

    except Exception as e:
        logger.error(f"RAG chain error: {e}", exc_info=True)
        error_code = _classify_error(e)
        yield _sse("error", {"code": error_code, "message": _error_message(error_code)})

    yield _sse("done", {})


def _sse(event: str, data: dict) -> str:
    """格式化 SSE 事件"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _classify_error(e: Exception) -> str:
    """分类错误类型"""
    err_str = str(e).lower()
    if "timeout" in err_str:
        return "LLM_TIMEOUT"
    if "rate" in err_str and "limit" in err_str:
        return "LLM_RATE_LIMIT"
    if "quota" in err_str or "balance" in err_str or "insufficient" in err_str:
        return "LLM_QUOTA"
    return "INTERNAL_ERROR"


def _error_message(code: str) -> str:
    """错误码对应的用户提示"""
    messages = {
        "LLM_TIMEOUT": "生成服务暂时不可用，请稍后重试",
        "LLM_RATE_LIMIT": "请求过于频繁，请稍后重试",
        "LLM_QUOTA": "服务暂时不可用，请联系管理员",
        "INTERNAL_ERROR": "系统内部错误，请稍后重试",
    }
    return messages.get(code, "未知错误")

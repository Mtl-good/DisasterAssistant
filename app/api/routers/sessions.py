"""会话管理 API"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_db
from app.schemas.session import (
    SessionCreate, SessionUpdate, SessionOut,
    SessionListItem, SessionListOut,
    MessageOut, MessageListOut,
)
from app.services import session_service

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.post("", response_model=SessionOut)
async def create_session(body: SessionCreate = None, db: AsyncSession = Depends(get_db)):
    """新建会话"""
    title = body.title if body and body.title else None
    session = await session_service.create_session(db, title)
    return session


@router.get("", response_model=SessionListOut)
async def list_sessions(keyword: str | None = None, db: AsyncSession = Depends(get_db)):
    """会话列表/搜索"""
    items = await session_service.list_sessions(db, keyword)
    return SessionListOut(
        items=[SessionListItem(**item) for item in items],
        total=len(items),
    )


@router.patch("/{session_id}", response_model=SessionOut)
async def rename_session(session_id: str, body: SessionUpdate, db: AsyncSession = Depends(get_db)):
    """重命名会话"""
    session = await session_service.rename_session(db, session_id, body.title)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    return session


@router.delete("/{session_id}")
async def delete_session(session_id: str, db: AsyncSession = Depends(get_db)):
    """删除会话"""
    ok = await session_service.delete_session(db, session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"success": True}


@router.get("/{session_id}/messages", response_model=MessageListOut)
async def get_messages(
    session_id: str,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """获取会话消息"""
    messages, total = await session_service.get_messages(db, session_id, limit, offset)
    return MessageListOut(
        items=[MessageOut.model_validate(m) for m in messages],
        total=total,
    )

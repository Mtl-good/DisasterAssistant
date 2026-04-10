"""会话管理服务"""

from datetime import datetime, timezone

from sqlalchemy import select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.session import Session
from app.models.message import Message


async def create_session(db: AsyncSession, title: str | None = None) -> Session:
    """新建会话"""
    session = Session(title=title or "新会话")
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


async def list_sessions(db: AsyncSession, keyword: str | None = None) -> list[dict]:
    """列出会话（支持关键词搜索），按更新时间倒序"""
    stmt = (
        select(
            Session,
            func.count(Message.id).label("message_count"),
        )
        .outerjoin(Message, Message.session_id == Session.id)
        .group_by(Session.id)
        .order_by(Session.updated_at.desc())
    )
    if keyword:
        stmt = stmt.where(Session.title.contains(keyword))

    result = await db.execute(stmt)
    rows = result.all()
    return [
        {
            "id": row.Session.id,
            "title": row.Session.title,
            "created_at": row.Session.created_at,
            "updated_at": row.Session.updated_at,
            "message_count": row.message_count,
        }
        for row in rows
    ]


async def get_session(db: AsyncSession, session_id: str) -> Session | None:
    """获取单个会话"""
    return await db.get(Session, session_id)


async def rename_session(db: AsyncSession, session_id: str, title: str) -> Session | None:
    """重命名会话"""
    session = await db.get(Session, session_id)
    if not session:
        return None
    session.title = title
    session.updated_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(session)
    return session


async def delete_session(db: AsyncSession, session_id: str) -> bool:
    """删除会话（级联删除消息）"""
    session = await db.get(Session, session_id)
    if not session:
        return False
    await db.delete(session)
    await db.commit()
    return True


async def get_messages(
    db: AsyncSession,
    session_id: str,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[Message], int]:
    """获取会话消息"""
    count_stmt = select(func.count(Message.id)).where(Message.session_id == session_id)
    total = (await db.execute(count_stmt)).scalar() or 0

    stmt = (
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.asc())
        .offset(offset)
        .limit(limit)
    )
    result = await db.execute(stmt)
    messages = list(result.scalars().all())
    return messages, total


async def add_message(
    db: AsyncSession,
    session_id: str,
    role: str,
    content: str,
    metadata_json: dict | None = None,
) -> Message:
    """添加消息"""
    msg = Message(
        session_id=session_id,
        role=role,
        content=content,
        metadata_json=metadata_json,
    )
    db.add(msg)
    # 同时更新会话的 updated_at
    session = await db.get(Session, session_id)
    if session:
        session.updated_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(msg)
    return msg


async def get_recent_history(db: AsyncSession, session_id: str, max_rounds: int = 5) -> list[Message]:
    """获取最近 N 轮对话历史（用于 Query 改写）"""
    stmt = (
        select(Message)
        .where(Message.session_id == session_id)
        .where(Message.role.in_(["user", "assistant"]))
        .order_by(Message.created_at.desc())
        .limit(max_rounds * 2)
    )
    result = await db.execute(stmt)
    messages = list(result.scalars().all())
    messages.reverse()  # 按时间正序
    return messages

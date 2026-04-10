"""问答 API - SSE 流式输出"""

from fastapi import APIRouter, Depends
from starlette.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_db
from app.schemas.chat import ChatQuery
from app.services.chat_service import chat_stream

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/query")
async def query(body: ChatQuery, db: AsyncSession = Depends(get_db)):
    """问答接口 - SSE 流式返回"""

    async def event_generator():
        async for sse_msg in chat_stream(db, body.session_id, body.query):
            yield sse_msg

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

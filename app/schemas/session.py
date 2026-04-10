"""会话相关 Schema"""

from datetime import datetime
from pydantic import BaseModel, Field


class SessionCreate(BaseModel):
    title: str | None = None


class SessionUpdate(BaseModel):
    title: str


class SessionOut(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class SessionListItem(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


class SessionListOut(BaseModel):
    items: list[SessionListItem]
    total: int


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    metadata_json: dict | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class MessageListOut(BaseModel):
    items: list[MessageOut]
    total: int

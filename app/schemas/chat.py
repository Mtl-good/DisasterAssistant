"""问答相关 Schema"""

from pydantic import BaseModel


class ChatQuery(BaseModel):
    session_id: str
    query: str

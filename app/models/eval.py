"""评测数据模型"""

import uuid
from datetime import datetime

from sqlalchemy import String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, utcnow


class EvalCase(Base):
    __tablename__ = "eval_cases"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    question: Mapped[str] = mapped_column(Text)
    gold_points_json: Mapped[dict] = mapped_column(JSON)
    gold_sections_json: Mapped[dict] = mapped_column(JSON)
    category: Mapped[str] = mapped_column(String(50))  # 震前/震时/震后/特殊场景


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    variant: Mapped[str] = mapped_column(String(10))  # G1/G2/G3
    case_id: Mapped[str] = mapped_column(String(36), ForeignKey("eval_cases.id"))
    answer_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    metrics_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

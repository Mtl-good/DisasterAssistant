"""评测 API"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_db
from app.services.eval_service import run_eval, get_eval_report

router = APIRouter(prefix="/api/eval", tags=["eval"])


@router.post("/run")
async def eval_run(variant: str = "G3", db: AsyncSession = Depends(get_db)):
    """运行评测"""
    if variant not in ("G1", "G2", "G3"):
        raise HTTPException(status_code=422, detail="variant 必须为 G1/G2/G3")
    results = await run_eval(db, variant)
    return {"variant": variant, "total": len(results), "results": results}


@router.get("/report")
async def eval_report(variant: str = "G3", db: AsyncSession = Depends(get_db)):
    """获取评测报告"""
    if variant not in ("G1", "G2", "G3"):
        raise HTTPException(status_code=422, detail="variant 必须为 G1/G2/G3")
    report = await get_eval_report(db, variant)
    return report

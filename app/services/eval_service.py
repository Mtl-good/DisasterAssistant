"""评测服务"""

import time

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.eval import EvalCase, EvalRun
from app.rag.chain import build_chain
from app.core.logging import logger


async def run_eval(db: AsyncSession, variant: str) -> list[dict]:
    """对所有评测用例运行指定 variant 的 RAG 链"""
    stmt = select(EvalCase).order_by(EvalCase.category)
    result = await db.execute(stmt)
    cases = list(result.scalars().all())

    if not cases:
        return []

    chain = build_chain(variant)
    runs = []

    for case in cases:
        start = time.time()
        try:
            result_data = await chain.ainvoke(case.question, [])
        except Exception as e:
            logger.error(f"Eval error for case {case.id}: {e}")
            result_data = {"answer": f"ERROR: {e}", "sources": [], "rewritten_query": None}

        latency_ms = int((time.time() - start) * 1000)

        # 存储完整结果
        answer_json = {
            "raw_text": result_data["answer"],
            "sources": result_data["sources"],
            "rewritten_query": result_data.get("rewritten_query"),
        }

        # 计算检索指标
        metrics = _compute_retrieval_metrics(case, result_data["sources"], latency_ms)

        run = EvalRun(
            variant=variant,
            case_id=case.id,
            answer_json=answer_json,
            metrics_json=metrics,
        )
        db.add(run)
        runs.append({
            "case_id": case.id,
            "question": case.question,
            "variant": variant,
            "metrics": metrics,
        })

    await db.commit()
    return runs


def _compute_retrieval_metrics(
    case: EvalCase,
    sources: list[dict],
    latency_ms: int,
) -> dict:
    """计算检索层指标：Hit@k, MRR, Recall@k"""
    gold_sections = case.gold_sections_json if isinstance(case.gold_sections_json, list) else []

    # 对每个检索结果，检查是否命中任一 gold section（子串匹配）
    hit = 0
    mrr = 0.0
    gold_hit_count = 0

    for gold in gold_sections:
        # 取 gold section 中的关键片段（前 30 个非空字符作为匹配锚点）
        gold_anchor = _extract_anchor(gold)
        if not gold_anchor:
            continue

        for i, src in enumerate(sources):
            content = src.get("content", "") if isinstance(src, dict) else ""
            if gold_anchor in content:
                hit = 1
                if mrr == 0.0:
                    mrr = 1.0 / (i + 1)
                gold_hit_count += 1
                break

    recall = gold_hit_count / len(gold_sections) if gold_sections else 0

    return {
        "hit": hit,
        "mrr": mrr,
        "recall": recall,
        "latency_ms": latency_ms,
    }


def _extract_anchor(gold_section: str) -> str:
    """从 gold section 提取匹配锚点

    取第一行非标记文本中有实质内容的部分（跳过标题标记）。
    """
    for line in gold_section.split("\n"):
        line = line.strip().lstrip("#").lstrip("-").lstrip("*").strip()
        if len(line) >= 6:
            return line
    return gold_section.strip()[:50]


async def get_eval_report(db: AsyncSession, variant: str) -> dict:
    """生成评测报告"""
    stmt = select(EvalRun).where(EvalRun.variant == variant).order_by(EvalRun.created_at.desc())
    result = await db.execute(stmt)
    runs = list(result.scalars().all())

    if not runs:
        return {"variant": variant, "total_cases": 0, "runs": []}

    latencies = []
    hits = 0
    mrr_sum = 0.0
    recall_sum = 0.0

    for run in runs:
        metrics = run.metrics_json or {}
        latencies.append(metrics.get("latency_ms", 0))
        hits += metrics.get("hit", 0)
        mrr_sum += metrics.get("mrr", 0.0)
        recall_sum += metrics.get("recall", 0.0)

    total = len(runs)
    latencies.sort()

    return {
        "variant": variant,
        "total_cases": total,
        "hit_at_k": hits / total if total else 0,
        "mrr": mrr_sum / total if total else 0,
        "recall": recall_sum / total if total else 0,
        "avg_latency_ms": sum(latencies) / total if total else 0,
        "p95_latency_ms": latencies[int(total * 0.95)] if total else 0,
        "runs": [
            {
                "id": r.id,
                "variant": r.variant,
                "case_id": r.case_id,
                "answer_json": r.answer_json,
                "metrics_json": r.metrics_json,
            }
            for r in runs
        ],
    }

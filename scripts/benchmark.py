#!/usr/bin/env python
"""RAG 检索策略对照实验评测脚本

用法:
    python scripts/benchmark.py                          # 跑全部 4 组
    python scripts/benchmark.py --strategies bm25,dense  # 只跑指定策略
    python scripts/benchmark.py --top-k 3                # 改 top_k
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

# 项目根目录加入 sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def load_eval_cases(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    """去除 Markdown 加粗标记和多余空白"""
    return text.replace("**", "").replace("*", "").strip()


def extract_key_phrases(gold_section: str, min_len: int = 6) -> list[str]:
    """从 gold_section 中提取用于匹配的关键短语"""
    normalized = normalize_text(gold_section)
    phrases = []
    for line in normalized.split("\n"):
        line = line.strip().lstrip("- ").lstrip("•").strip()
        if len(line) >= min_len:
            phrases.append(line[:40] if len(line) > 40 else line)
    return phrases


def is_relevant(doc_content: str, gold_section: str) -> bool:
    """判断文档是否命中 gold_section（子串匹配）"""
    normalized_doc = normalize_text(doc_content)
    for phrase in extract_key_phrases(gold_section):
        if phrase in normalized_doc:
            return True
    return False


# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------

def compute_case_metrics(
    documents: list,
    gold_sections: list[str],
    latency_ms: float,
    top_k: int = 5,
) -> dict:
    """计算单题检索指标"""
    docs = documents[:top_k]

    hit = 0
    first_rank = 0
    relevance = []

    for rank, doc in enumerate(docs):
        rel = any(is_relevant(doc.page_content, g) for g in gold_sections)
        relevance.append(1 if rel else 0)
        if rel and hit == 0:
            hit = 1
            first_rank = rank + 1

    # MRR
    mrr = 1.0 / first_rank if first_rank > 0 else 0.0

    # Recall@k
    matched = 0
    for gold in gold_sections:
        if any(is_relevant(doc.page_content, gold) for doc in docs):
            matched += 1
    recall = matched / len(gold_sections) if gold_sections else 0.0

    # nDCG@k
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(relevance))
    ideal = sorted(relevance, reverse=True)
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {
        "hit": hit,
        "mrr": mrr,
        "recall": recall,
        "ndcg": ndcg,
        "latency_ms": latency_ms,
        "first_rank": first_rank,
    }


# ---------------------------------------------------------------------------
# Markdown 报告生成
# ---------------------------------------------------------------------------

def generate_report(
    all_results: dict[str, list[dict]],
    all_summaries: dict[str, dict],
    eval_cases: list[dict],
    top_k: int,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []

    lines.append("# RAG 检索策略对照实验结果\n")
    lines.append("## 实验概况\n")
    lines.append(f"- 测试时间：{now}")
    lines.append(f"- 测试题数：{len(eval_cases)}")
    lines.append(f"- Top K：{top_k}")
    lines.append("")

    # ---- 总览表 ----
    ranked = sorted(all_summaries.items(), key=lambda x: x[1]["mrr"], reverse=True)

    lines.append("## 指标对比总览\n")
    lines.append(
        "| 排名 | 策略 | Hit@{k} | Recall@{k} | MRR | nDCG@{k} | 平均延迟 | P95 延迟 |".format(
            k=top_k
        )
    )
    lines.append("|------|------|---------|------------|-----|----------|---------|---------|")

    for rank, (sid, s) in enumerate(ranked, 1):
        lines.append(
            f"| {rank} "
            f"| {s['display_name']} "
            f"| {s['hit_at_k']:.1%} "
            f"| {s['recall_at_k']:.1%} "
            f"| {s['mrr']:.3f} "
            f"| {s['ndcg_at_k']:.3f} "
            f"| {s['avg_latency_ms']:.0f}ms "
            f"| {s['p95_latency_ms']:.0f}ms |"
        )

    lines.append("")

    # ---- 综合分析 ----
    lines.append("## 综合分析\n")
    best_q = ranked[0]
    best_l = min(ranked, key=lambda x: x[1]["avg_latency_ms"])
    best_cp = max(ranked, key=lambda x: x[1]["mrr"] / max(x[1]["avg_latency_ms"], 1))

    lines.append(
        f"- **最高质量**：{best_q[1]['display_name']}"
        f"（MRR: {best_q[1]['mrr']:.3f}）"
    )
    lines.append(
        f"- **最低延迟**：{best_l[1]['display_name']}"
        f"（平均: {best_l[1]['avg_latency_ms']:.0f}ms）"
    )
    lines.append(
        f"- **最佳性价比**：{best_cp[1]['display_name']}"
        f"（MRR: {best_cp[1]['mrr']:.3f}, 延迟: {best_cp[1]['avg_latency_ms']:.0f}ms）"
    )
    lines.append("")

    # ---- 各题详细结果 ----
    lines.append("## 各题详细结果\n")

    for sid, case_results in all_results.items():
        s = all_summaries[sid]
        lines.append(f"### {s['display_name']}\n")
        lines.append("| # | 问题 | Hit | MRR | 延迟 | Top1 文档片段 |")
        lines.append("|---|------|-----|-----|------|-------------|")

        for i, cr in enumerate(case_results):
            q = cr["question"]
            if len(q) > 30:
                q = q[:30] + "..."
            hit_s = "Y" if cr["hit"] else "N"
            snippet = "-"
            if cr.get("top1_content"):
                snippet = cr["top1_content"][:50].replace("\n", " ") + "..."
            lines.append(
                f"| {i + 1} | {q} | {hit_s} | {cr['mrr']:.3f} "
                f"| {cr['latency_ms']:.0f}ms | {snippet} |"
            )

        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

async def run_benchmark(strategy_ids: list[str], top_k: int = 5):
    from app.rag.strategies import get_strategy

    eval_path = project_root / "data" / "eval_cases.json"
    eval_cases = load_eval_cases(eval_path)
    print(f"已加载 {len(eval_cases)} 道评测题 ({eval_path})")

    all_results: dict[str, list[dict]] = {}
    all_summaries: dict[str, dict] = {}

    for sid in strategy_ids:
        print(f"\n{'=' * 60}")
        print(f"  策略: {sid}")
        print(f"{'=' * 60}")

        strategy = get_strategy(sid)
        case_results: list[dict] = []

        for i, case in enumerate(eval_cases):
            query = case["question"]
            golds = case.get("gold_sections", [])

            result = await strategy.retrieve(query, top_k=top_k)
            metrics = compute_case_metrics(result.documents, golds, result.latency_ms, top_k)

            top1 = result.documents[0].page_content if result.documents else ""
            case_results.append(
                {
                    "case_id": case["case_id"],
                    "question": query,
                    **metrics,
                    "top1_content": top1,
                }
            )

            tag = "HIT " if metrics["hit"] else "MISS"
            print(
                f"  [{i + 1:02d}/{len(eval_cases)}] {tag} "
                f"| MRR={metrics['mrr']:.3f} "
                f"| {metrics['latency_ms']:.0f}ms "
                f"| {query[:40]}..."
            )

        # 汇总
        n = len(case_results)
        latencies = sorted(cr["latency_ms"] for cr in case_results)
        p95_idx = min(int(n * 0.95), n - 1)

        summary = {
            "display_name": strategy.display_name,
            "hit_at_k": sum(cr["hit"] for cr in case_results) / n,
            "recall_at_k": sum(cr["recall"] for cr in case_results) / n,
            "mrr": sum(cr["mrr"] for cr in case_results) / n,
            "ndcg_at_k": sum(cr["ndcg"] for cr in case_results) / n,
            "avg_latency_ms": sum(latencies) / n,
            "p95_latency_ms": latencies[p95_idx],
        }

        all_results[sid] = case_results
        all_summaries[sid] = summary

        print(
            f"\n  汇总: Hit@{top_k}={summary['hit_at_k']:.1%} "
            f"| Recall@{top_k}={summary['recall_at_k']:.1%} "
            f"| MRR={summary['mrr']:.3f} "
            f"| nDCG@{top_k}={summary['ndcg_at_k']:.3f} "
            f"| Avg={summary['avg_latency_ms']:.0f}ms "
            f"| P95={summary['p95_latency_ms']:.0f}ms"
        )

    # 生成报告
    report = generate_report(all_results, all_summaries, eval_cases, top_k)

    output_path = project_root / "ans.md"
    output_path.write_text(report, encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"  报告已写入: {output_path}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="RAG 检索策略对照实验")
    parser.add_argument(
        "--strategies",
        type=str,
        default=None,
        help="逗号分隔的策略 ID (默认: 全部 4 种)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top K (默认: 5)",
    )
    args = parser.parse_args()

    if args.strategies:
        ids = [s.strip() for s in args.strategies.split(",")]
    else:
        ids = ["bm25", "dense", "hybrid_rrf", "hybrid_rrf_rerank"]

    print(f"待评测策略: {ids}")
    print(f"Top K: {args.top_k}")

    asyncio.run(run_benchmark(ids, args.top_k))


if __name__ == "__main__":
    main()

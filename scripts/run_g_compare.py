#!/usr/bin/env python
"""G1/G2/G3 对比实验脚本

用法:
    python scripts/run_g_compare.py                  # 跑全部 3 组
    python scripts/run_g_compare.py --variants G1,G3 # 只跑指定分组
    python scripts/run_g_compare.py --no-judge        # 跳过 LLM-as-Judge（仅检索指标）
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

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def load_eval_cases(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    return text.replace("**", "").replace("*", "").strip()


def extract_key_phrases(gold_section: str, min_len: int = 6) -> list[str]:
    normalized = normalize_text(gold_section)
    phrases = []
    for line in normalized.split("\n"):
        line = line.strip().lstrip("- ").lstrip("•").strip()
        if len(line) >= min_len:
            phrases.append(line[:40] if len(line) > 40 else line)
    return phrases


def is_relevant(doc_content: str, gold_section: str) -> bool:
    normalized_doc = normalize_text(doc_content)
    for phrase in extract_key_phrases(gold_section):
        if phrase in normalized_doc:
            return True
    return False


# ---------------------------------------------------------------------------
# 检索指标计算
# ---------------------------------------------------------------------------

def compute_retrieval_metrics(
    sources: list[dict],
    gold_sections: list[str],
) -> dict:
    """计算检索层指标"""
    hit = 0
    first_rank = 0
    relevance = []

    for rank, src in enumerate(sources):
        content = src.get("content", "")
        rel = any(is_relevant(content, g) for g in gold_sections)
        relevance.append(1 if rel else 0)
        if rel and hit == 0:
            hit = 1
            first_rank = rank + 1

    mrr = 1.0 / first_rank if first_rank > 0 else 0.0

    matched = 0
    for gold in gold_sections:
        if any(is_relevant(src.get("content", ""), gold) for src in sources):
            matched += 1
    recall = matched / len(gold_sections) if gold_sections else 0.0

    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(relevance))
    ideal = sorted(relevance, reverse=True)
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {
        "hit": hit,
        "mrr": mrr,
        "recall": recall,
        "ndcg": ndcg,
    }


# ---------------------------------------------------------------------------
# LLM-as-Judge 评估
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """你是一个严格的评测专家。请根据以下标准对 RAG 系统的回答进行评分。

## 评测问题
{question}

## 标准答案要点（gold_sections）
{gold_sections}

## 系统回答
{answer}

## 评分标准

请从以下 3 个维度评分（每项 1-5 分）：

### 1. 完整性（completeness）
- 5分：覆盖了标准答案中所有关键要点
- 4分：覆盖了大部分关键要点（>80%）
- 3分：覆盖了约一半关键要点
- 2分：只覆盖了少量要点（<30%）
- 1分：几乎没有覆盖标准答案的要点

### 2. 准确性（accuracy）
- 5分：所有事实陈述都有参考资料支持，无编造信息
- 4分：极少数表述略有偏差，但无明显错误
- 3分：有 1-2 处事实错误或缺乏证据的陈述
- 2分：有多处事实错误或编造信息
- 1分：大量编造或错误信息

### 3. 可执行性（actionability）
- 5分：每条建议都具体、可操作（如"用手臂护住头部和颈部"）
- 4分：大部分建议具体可操作
- 3分：部分建议较为笼统（如"注意安全"）
- 2分：多数建议笼统，缺乏具体步骤
- 1分：几乎全是笼统表述，无法指导行动

## 输出格式
请严格按以下 JSON 格式输出，不要输出其他内容：
{{"completeness": <1-5>, "accuracy": <1-5>, "actionability": <1-5>, "hallucination_count": <非负整数，回答中无证据支持的事实陈述条数>}}"""


async def judge_answer(
    llm,
    question: str,
    gold_sections: list[str],
    answer: str,
    model: str,
) -> dict:
    """用 LLM 对回答进行质量评估"""
    gold_text = "\n---\n".join(gold_sections)
    prompt = JUDGE_PROMPT.format(
        question=question,
        gold_sections=gold_text,
        answer=answer,
    )

    try:
        resp = await llm.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        # 提取 JSON
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(raw[start:end])
    except Exception as e:
        print(f"    [WARN] Judge failed: {e}")

    return {"completeness": 0, "accuracy": 0, "actionability": 0, "hallucination_count": -1}


# ---------------------------------------------------------------------------
# 引用正确率计算（G3 特有）
# ---------------------------------------------------------------------------

def compute_citation_precision(answer: str, sources: list[dict]) -> float | None:
    """检测回答中的引用标记，校验引用内容是否存在于检索文档中

    返回 None 表示回答中没有引用标记。
    """
    import re
    # 匹配 > 来源：xxx 格式的引用
    citations = re.findall(r"来源[：:]\s*(.+?)(?:\n|$)", answer)
    if not citations:
        return None

    correct = 0
    source_texts = " ".join(s.get("section", "") + " " + s.get("content", "") for s in sources)

    for citation in citations:
        # 检查引用中的关键词是否出现在检索文档中
        keywords = [w.strip() for w in citation.replace(">", " ").split() if len(w.strip()) >= 2]
        if keywords and any(kw in source_texts for kw in keywords):
            correct += 1

    return correct / len(citations) if citations else None


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------

def generate_report(
    all_results: dict[str, list[dict]],
    all_summaries: dict[str, dict],
    eval_cases: list[dict],
    has_judge: bool,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []

    lines.append("# G1/G2/G3 对比实验结果\n")
    lines.append("## 实验概况\n")
    lines.append(f"- 测试时间：{now}")
    lines.append(f"- 测试题数：{len(eval_cases)}")
    lines.append(f"- LLM-as-Judge：{'是' if has_judge else '否'}")
    lines.append("")

    # ---- 功能特性矩阵 ----
    lines.append("## 各组功能特性\n")
    lines.append("| 功能模块 | G1 (基线) | G2 (检索优化) | G3 (全量优化) |")
    lines.append("|---------|-----------|-------------|-------------|")
    lines.append("| 向量检索 | Y | Y | Y |")
    lines.append("| Query 改写 | - | Y | Y |")
    lines.append("| Reranker | - | Y | Y |")
    lines.append("| 结构化输出约束 | - | - | Y |")
    lines.append("| 引用校验 | - | - | Y |")
    lines.append("| 追问机制 | - | - | Y |")
    lines.append("")

    # ---- 指标对比总览 ----
    variants_order = ["G1", "G2", "G3"]
    available = [v for v in variants_order if v in all_summaries]

    lines.append("## 指标对比总览\n")

    # 检索指标表
    lines.append("### 检索层指标\n")
    lines.append("| 指标 | " + " | ".join(available) + " |")
    lines.append("|------|" + "|".join(["------"] * len(available)) + "|")

    for metric, label, fmt in [
        ("hit_at_k", "Hit@K", ".1%"),
        ("recall", "Recall@K", ".1%"),
        ("mrr", "MRR", ".3f"),
        ("ndcg", "nDCG@K", ".3f"),
        ("avg_latency_ms", "平均延迟", ".0f"),
        ("p95_latency_ms", "P95 延迟", ".0f"),
    ]:
        cells = []
        for v in available:
            val = all_summaries[v].get(metric, 0)
            if "latency" in metric:
                cells.append(f"{val:{fmt}}ms")
            else:
                cells.append(f"{val:{fmt}}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines.append("")

    # G1 作为基线的提升量
    if "G1" in all_summaries and len(available) > 1:
        lines.append("### 相对 G1 基线的提升\n")
        lines.append("| 指标 | " + " | ".join(v for v in available if v != "G1") + " |")
        lines.append("|------|" + "|".join(["------"] * (len(available) - 1)) + "|")

        g1 = all_summaries["G1"]
        for metric, label, is_pct in [
            ("hit_at_k", "Hit@K", True),
            ("recall", "Recall@K", True),
            ("mrr", "MRR", False),
            ("ndcg", "nDCG@K", False),
        ]:
            cells = []
            for v in available:
                if v == "G1":
                    continue
                diff = all_summaries[v].get(metric, 0) - g1.get(metric, 0)
                if is_pct:
                    cells.append(f"+{diff:.1%}" if diff >= 0 else f"{diff:.1%}")
                else:
                    cells.append(f"+{diff:.3f}" if diff >= 0 else f"{diff:.3f}")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")

    # 生成质量指标表（如果有 Judge）
    if has_judge:
        lines.append("### 生成质量指标（LLM-as-Judge）\n")
        lines.append("| 指标 | " + " | ".join(available) + " |")
        lines.append("|------|" + "|".join(["------"] * len(available)) + "|")

        for metric, label in [
            ("completeness", "完整性 (1-5)"),
            ("accuracy", "准确性 (1-5)"),
            ("actionability", "可执行性 (1-5)"),
            ("hallucination_rate", "幻觉率"),
            ("citation_precision", "引用正确率"),
        ]:
            cells = []
            for v in available:
                val = all_summaries[v].get(metric)
                if val is None:
                    cells.append("N/A")
                elif metric == "hallucination_rate":
                    cells.append(f"{val:.1%}")
                elif metric == "citation_precision":
                    cells.append(f"{val:.1%}")
                else:
                    cells.append(f"{val:.2f}")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")

    # ---- 各题详细结果 ----
    lines.append("## 各组详细结果\n")

    for v in available:
        case_results = all_results[v]
        lines.append(f"### {v}\n")

        if has_judge:
            lines.append("| # | 问题 | Hit | MRR | 完整性 | 准确性 | 可执行 | 延迟 |")
            lines.append("|---|------|-----|-----|--------|--------|--------|------|")
        else:
            lines.append("| # | 问题 | Hit | MRR | Recall | 延迟 |")
            lines.append("|---|------|-----|-----|--------|------|")

        for i, cr in enumerate(case_results):
            q = cr["question"]
            if len(q) > 30:
                q = q[:30] + "..."
            hit_s = "Y" if cr["hit"] else "N"

            if has_judge:
                lines.append(
                    f"| {i + 1} | {q} | {hit_s} | {cr['mrr']:.3f} "
                    f"| {cr.get('completeness', 0)} | {cr.get('accuracy', 0)} "
                    f"| {cr.get('actionability', 0)} | {cr['latency_ms']:.0f}ms |"
                )
            else:
                lines.append(
                    f"| {i + 1} | {q} | {hit_s} | {cr['mrr']:.3f} "
                    f"| {cr['recall']:.1%} | {cr['latency_ms']:.0f}ms |"
                )

        lines.append("")

    # ---- 典型案例对比 ----
    if len(available) >= 2 and has_judge:
        lines.append("## 典型案例对比分析\n")
        lines.append(_generate_case_analysis(all_results, available, eval_cases))

    return "\n".join(lines)


def _generate_case_analysis(
    all_results: dict[str, list[dict]],
    variants: list[str],
    eval_cases: list[dict],
) -> str:
    """找出差异最大的案例进行对比分析"""
    lines: list[str] = []
    n_cases = len(eval_cases)

    # 计算每题的分组差异度
    diffs = []
    for i in range(n_cases):
        scores = []
        for v in variants:
            cr = all_results[v][i]
            # 综合得分 = 检索 hit + judge 分数
            total = cr.get("hit", 0) * 2 + cr.get("completeness", 0) + cr.get("accuracy", 0)
            scores.append(total)
        diff = max(scores) - min(scores)
        diffs.append((i, diff))

    # 取差异最大的 5 题
    diffs.sort(key=lambda x: x[1], reverse=True)
    top_cases = diffs[:5]

    for rank, (idx, diff) in enumerate(top_cases, 1):
        case = eval_cases[idx]
        lines.append(f"### 案例 {rank}：{case['question']}\n")

        for v in variants:
            cr = all_results[v][idx]
            hit_s = "命中" if cr["hit"] else "未命中"
            lines.append(f"**{v}**：检索{hit_s} (MRR={cr['mrr']:.3f})")
            if cr.get("completeness"):
                lines.append(
                    f"  - 完整性={cr['completeness']} 准确性={cr['accuracy']} "
                    f"可执行性={cr['actionability']}"
                )
            answer_preview = cr.get("answer_preview", "")
            if answer_preview:
                lines.append(f"  - 回答摘要：{answer_preview}")
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

async def run_g_compare(
    variants: list[str],
    use_judge: bool = True,
):
    from openai import AsyncOpenAI
    from app.core.config import settings
    from app.rag.chain import build_chain

    eval_path = project_root / "data" / "eval_cases.json"
    eval_cases = load_eval_cases(eval_path)
    print(f"已加载 {len(eval_cases)} 道评测题")
    print(f"待评测分组：{variants}")
    print(f"LLM-as-Judge：{'开启' if use_judge else '关闭'}")

    # 用于 Judge 的 LLM 客户端
    judge_llm = None
    if use_judge:
        judge_llm = AsyncOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            timeout=60.0,
        )

    all_results: dict[str, list[dict]] = {}
    all_summaries: dict[str, dict] = {}

    for variant in variants:
        print(f"\n{'=' * 60}")
        print(f"  分组: {variant}")
        print(f"{'=' * 60}")

        chain = build_chain(variant)
        case_results: list[dict] = []

        for i, case in enumerate(eval_cases):
            question = case["question"]
            golds = case.get("gold_sections", [])

            # 执行 RAG
            start = time.time()
            try:
                result = await chain.ainvoke(question, [])
            except Exception as e:
                print(f"    [ERROR] case {i + 1}: {e}")
                result = {"answer": f"ERROR: {e}", "sources": [], "rewritten_query": None}
            latency_ms = (time.time() - start) * 1000

            # 检索指标
            ret_metrics = compute_retrieval_metrics(result["sources"], golds)

            # LLM-as-Judge
            judge_scores = {}
            if use_judge and judge_llm and not result["answer"].startswith("ERROR"):
                judge_scores = await judge_answer(
                    judge_llm,
                    question,
                    golds,
                    result["answer"],
                    settings.deepseek_model,
                )

            # 引用正确率（所有 variant 都可以算，有引用才有值）
            cit_prec = compute_citation_precision(result["answer"], result["sources"])

            # 计算幻觉相关
            answer_facts = _count_facts(result["answer"])
            halluc_count = judge_scores.get("hallucination_count", 0)
            if halluc_count < 0:
                halluc_count = 0

            cr = {
                "case_id": case["case_id"],
                "question": question,
                "answer_preview": result["answer"][:100].replace("\n", " "),
                "latency_ms": latency_ms,
                **ret_metrics,
                **judge_scores,
                "citation_precision": cit_prec,
                "hallucination_count": halluc_count,
                "fact_count": answer_facts,
            }
            case_results.append(cr)

            hit_s = "HIT " if ret_metrics["hit"] else "MISS"
            judge_info = ""
            if judge_scores:
                judge_info = (
                    f" | 完整={judge_scores.get('completeness', '?')}"
                    f" 准确={judge_scores.get('accuracy', '?')}"
                    f" 可执行={judge_scores.get('actionability', '?')}"
                )
            print(
                f"  [{i + 1:02d}/{len(eval_cases)}] {hit_s}"
                f" | MRR={ret_metrics['mrr']:.3f}"
                f" | {latency_ms:.0f}ms"
                f"{judge_info}"
                f" | {question[:35]}..."
            )

        # 汇总
        n = len(case_results)
        latencies = sorted(cr["latency_ms"] for cr in case_results)
        p95_idx = min(int(n * 0.95), n - 1)

        summary: dict = {
            "hit_at_k": sum(cr["hit"] for cr in case_results) / n,
            "recall": sum(cr["recall"] for cr in case_results) / n,
            "mrr": sum(cr["mrr"] for cr in case_results) / n,
            "ndcg": sum(cr["ndcg"] for cr in case_results) / n,
            "avg_latency_ms": sum(latencies) / n,
            "p95_latency_ms": latencies[p95_idx],
        }

        if use_judge:
            valid_judges = [cr for cr in case_results if cr.get("completeness", 0) > 0]
            if valid_judges:
                nj = len(valid_judges)
                summary["completeness"] = sum(cr["completeness"] for cr in valid_judges) / nj
                summary["accuracy"] = sum(cr["accuracy"] for cr in valid_judges) / nj
                summary["actionability"] = sum(cr["actionability"] for cr in valid_judges) / nj

                total_facts = sum(cr.get("fact_count", 1) for cr in valid_judges)
                total_halluc = sum(cr.get("hallucination_count", 0) for cr in valid_judges)
                summary["hallucination_rate"] = total_halluc / max(total_facts, 1)

            # 引用正确率
            cit_cases = [cr for cr in case_results if cr.get("citation_precision") is not None]
            if cit_cases:
                summary["citation_precision"] = sum(cr["citation_precision"] for cr in cit_cases) / len(cit_cases)

        all_results[variant] = case_results
        all_summaries[variant] = summary

        print(f"\n  汇总: Hit@K={summary['hit_at_k']:.1%} | MRR={summary['mrr']:.3f} | Avg={summary['avg_latency_ms']:.0f}ms")
        if use_judge and "completeness" in summary:
            print(
                f"  Judge: 完整性={summary['completeness']:.2f} "
                f"准确性={summary['accuracy']:.2f} "
                f"可执行性={summary['actionability']:.2f} "
                f"幻觉率={summary.get('hallucination_rate', 0):.1%}"
            )

    # 生成报告
    report = generate_report(all_results, all_summaries, eval_cases, use_judge)
    output_path = project_root / "g_compare.md"
    output_path.write_text(report, encoding="utf-8")

    # 同时保存原始数据为 JSON（便于后续分析）
    raw_data = {
        "timestamp": datetime.now().isoformat(),
        "variants": variants,
        "has_judge": use_judge,
        "summaries": all_summaries,
        "results": {v: results for v, results in all_results.items()},
    }
    raw_path = project_root / "g_compare_raw.json"
    raw_path.write_text(json.dumps(raw_data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"  报告已写入: {output_path}")
    print(f"  原始数据: {raw_path}")
    print(f"{'=' * 60}")


def _count_facts(answer: str) -> int:
    """粗略统计回答中的事实陈述条数（按列表项和句子数计算）"""
    lines = answer.strip().split("\n")
    count = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 列表项
        if line.startswith(("-", "•", "*")) or (len(line) >= 2 and line[0].isdigit() and line[1] in ".）)"):
            count += 1
        elif len(line) > 10:
            count += 1
    return max(count, 1)


def main():
    parser = argparse.ArgumentParser(description="G1/G2/G3 对比实验")
    parser.add_argument(
        "--variants",
        type=str,
        default="G1,G2,G3",
        help="逗号分隔的分组 (默认: G1,G2,G3)",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="跳过 LLM-as-Judge 评估（仅检索指标）",
    )
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",")]
    for v in variants:
        if v not in ("G1", "G2", "G3"):
            print(f"错误: 无效的 variant '{v}'，必须为 G1/G2/G3")
            sys.exit(1)

    asyncio.run(run_g_compare(variants, use_judge=not args.no_judge))


if __name__ == "__main__":
    main()

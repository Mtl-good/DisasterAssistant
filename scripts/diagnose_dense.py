"""诊断 Dense 向量检索质量

检查项:
1. FAISS 索引维度与 embedding 维度是否一致
2. 在线 API embedding 质量抽样
3. FAISS 返回的 L2 距离分布
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import json
import numpy as np


def check_faiss_index():
    """检查 FAISS 索引基本信息"""
    import faiss as faiss_lib
    from app.core.config import settings

    index_path = Path(settings.faiss_index_dir) / "index.faiss"
    index = faiss_lib.read_index(str(index_path))

    print("=" * 60)
    print("  1. FAISS 索引信息")
    print("=" * 60)
    print(f"  索引维度 (d):     {index.d}")
    print(f"  文档总数 (ntotal): {index.ntotal}")
    print(f"  索引类型:         {type(index).__name__}")
    return index.d, index.ntotal


def check_embedding_dim():
    """检查当前 embedding 模型输出维度"""
    from app.rag.embeddings import get_embeddings
    from app.core.config import settings

    print("\n" + "=" * 60)
    print("  2. Embedding 模型检查")
    print("=" * 60)
    print(f"  Provider:  {settings.embedding_provider}")
    print(f"  Model:     {settings.embedding_model}")
    print(f"  API Base:  {settings.embedding_api_base}")

    emb = get_embeddings()
    test_text = "地震发生时应该如何避险"
    vec = emb.embed_query(test_text)
    dim = len(vec)
    norm = float(np.linalg.norm(vec))

    print(f"  输出维度:  {dim}")
    print(f"  向量模长:  {norm:.4f}")
    print(f"  前5个值:   {[f'{v:.4f}' for v in vec[:5]]}")
    return dim


def check_distance_distribution():
    """检查查询与文档的 L2 距离分布"""
    from app.rag.retriever import get_vectorstore

    print("\n" + "=" * 60)
    print("  3. L2 距离分布分析")
    print("=" * 60)

    eval_path = project_root / "data" / "eval_cases.json"
    with open(eval_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    vs = get_vectorstore()

    # 取前5个问题做抽样
    sample_cases = cases[:5]
    for i, case in enumerate(sample_cases):
        q = case["question"]
        results = vs.similarity_search_with_score(q, k=10)
        distances = [score for _, score in results]
        print(f"\n  Q{i+1}: {q[:40]}...")
        print(f"    Top10 L2 距离: min={min(distances):.4f}, max={max(distances):.4f}, "
              f"mean={np.mean(distances):.4f}")
        print(f"    各距离: {[f'{d:.3f}' for d in distances]}")

        # 检查距离区分度
        if len(distances) >= 2:
            gap = distances[-1] - distances[0]
            print(f"    Top1-Top10 差距: {gap:.4f} "
                  f"({'区分度好' if gap > 0.1 else '区分度差 ⚠'})")


def check_sample_retrieval():
    """用已知答案检查 Dense 检索命中情况"""
    from app.rag.retriever import get_vectorstore

    print("\n" + "=" * 60)
    print("  4. 抽样检索命中检查")
    print("=" * 60)

    eval_path = project_root / "data" / "eval_cases.json"
    with open(eval_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    vs = get_vectorstore()

    for i, case in enumerate(cases[:5]):
        q = case["question"]
        golds = case.get("gold_sections", [])
        results = vs.similarity_search_with_score(q, k=5)

        print(f"\n  Q{i+1}: {q[:50]}")

        hit = False
        for rank, (doc, score) in enumerate(results):
            content_preview = doc.page_content[:60].replace("\n", " ")
            # 简单匹配检查
            match = any(
                g.replace("**", "")[:30] in doc.page_content.replace("**", "")
                for g in golds if len(g) > 10
            )
            tag = "MATCH" if match else "     "
            if match:
                hit = True
            print(f"    [{rank+1}] L2={score:.3f} {tag} | {content_preview}...")

        print(f"    => {'HIT' if hit else 'MISS ⚠'}")


def main():
    print("\n" + "#" * 60)
    print("#  Dense 向量检索诊断报告")
    print("#" * 60)

    # 1. 检查 FAISS 索引
    try:
        index_dim, ntotal = check_faiss_index()
    except Exception as e:
        print(f"  FAISS 索引检查失败: {e}")
        index_dim = None

    # 2. 检查 embedding 维度
    try:
        emb_dim = check_embedding_dim()
    except Exception as e:
        print(f"  Embedding 检查失败: {e}")
        emb_dim = None

    # 3. 维度一致性
    if index_dim and emb_dim:
        print("\n" + "=" * 60)
        print("  维度一致性检查")
        print("=" * 60)
        if index_dim == emb_dim:
            print(f"  PASS: 索引维度({index_dim}) == Embedding维度({emb_dim})")
        else:
            print(f"  FAIL ⚠: 索引维度({index_dim}) != Embedding维度({emb_dim})")
            print(f"  这是 Dense 检索失效的根本原因！")
            print(f"  需要用当前 embedding 模型重建索引。")
            return

    # 4. 距离分布
    try:
        check_distance_distribution()
    except Exception as e:
        print(f"  距离分布检查失败: {e}")

    # 5. 抽样检索
    try:
        check_sample_retrieval()
    except Exception as e:
        print(f"  抽样检索检查失败: {e}")

    print("\n" + "#" * 60)
    print("#  诊断完成")
    print("#" * 60)


if __name__ == "__main__":
    main()

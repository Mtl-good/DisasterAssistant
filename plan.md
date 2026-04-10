# RAG 检索策略对照实验计划

## 1. 背景与目标

### 1.1 现状

当前系统已实现三个 variant：

| Variant | 向量检索 | Query 改写 | Reranker | 结构化输出 | 引用校验 | 追问机制 |
|---------|---------|-----------|----------|-----------|---------|---------|
| G1 | Y | - | - | - | - | - |
| G2 | Y | Y | Y | - | - | - |
| G3 | Y | Y | Y | Y | Y | Y |

**问题**：G1/G2/G3 把"检索策略改进"和"生成约束改进"混在一起，无法精确归因哪部分提升来自检索层，哪部分来自生成层。

### 1.2 目标

- 解耦检索层与生成层，**单独对比 4 种检索策略**的效果
- 通过统一评测脚本，用**数值指标**（百分比）直接对比各策略
- 找到一个综合性最强的检索方案作为默认配置

### 1.3 回答你的三个疑问

> **Q1：每组的方式都要重写吗？**
>
> **不需要。** 采用**策略模式（Strategy Pattern）**，每种检索方式只需实现一个 Python 类（继承统一接口），通过配置切换。评测脚本自动遍历全部 4 种策略，一键跑完所有实验。

> **Q2：测评的结果我要怎么拿到？**
>
> 结果自动写入项目根目录的 **`ans.md`** 文件，包含：
> 1. 各策略的指标对比排行榜（百分比数值）
> 2. 每道测试题的详细检索结果
> 3. 综合分析与推荐结论

> **Q3：实验做几组合适？**
>
> **4 组**，覆盖从简单到复杂的检索范式递进关系：
> - 策略 A（BM25）：纯关键词基线
> - 策略 B（Dense）：纯语义基线
> - 策略 C（Hybrid）：关键词 + 语义融合
> - 策略 D（Hybrid + Rerank）：融合 + 精排

---

## 2. 四种检索策略详解

| 策略 | ID | 核心原理 | 优势 | 劣势 |
|------|----|---------|------|------|
| **A: BM25** | `bm25` | 基于词频(TF)和逆文档频率(IDF)的关键词匹配，使用 jieba 中文分词 | 无需 GPU、速度最快、术语精确匹配好、可解释性强 | 语义理解弱、无法处理同义词和改述 |
| **B: Dense** | `dense` | 将 query 和文档编码为向量（BGE-M3），用 FAISS 计算余弦相似度 | 语义理解强、可处理同义词和改述 | 对罕见术语弱、依赖 embedding 模型质量 |
| **C: Hybrid (RRF)** | `hybrid_rrf` | BM25 + Dense 双路召回，用 RRF（Reciprocal Rank Fusion）算法融合排名 | 鲁棒性高、互补优势、对分值尺度不敏感 | 实现略复杂 |
| **D: Hybrid + Rerank** | `hybrid_rrf_rerank` | 在策略 C 基础上，用 BGE-Reranker-v2-m3 交叉编码器精排 | 目前公认最强通用方案、精确度最高 | 延迟最高、增加 API 成本 |

### 递进关系

```
A (BM25)              ─── 纯稀疏检索基线
B (Dense)             ─── 纯密集检索基线
C (Hybrid RRF)        ─── A + B 融合，验证互补效果
D (Hybrid + Rerank)   ─── C + 精排，验证精排增益
```

---

## 3. 架构设计：可插拔的检索策略

### 3.1 核心思路

**策略模式 + 注册表** → 每种检索方式是一个独立 Python 类，评测脚本通过策略 ID 自动实例化和调用。

### 3.2 目录结构

```
app/rag/
├── strategies/                          # 新增：策略目录
│   ├── __init__.py                     # 注册表 + 工厂函数
│   ├── base.py                         # 抽象基类 + RetrievalResult
│   ├── bm25_strategy.py                # 策略 A: BM25
│   ├── dense_strategy.py               # 策略 B: Dense (FAISS)
│   ├── hybrid_rrf_strategy.py          # 策略 C: Hybrid RRF
│   └── hybrid_rrf_rerank_strategy.py   # 策略 D: Hybrid RRF + Reranker
├── retriever.py                         # 现有：保持不变（被 Dense 策略复用）
├── reranker.py                          # 现有：保持不变（被 Rerank 策略复用）
└── ...

scripts/
├── benchmark.py                         # 新增：一键评测脚本

data/
├── eval_cases.json                      # 现有：20 道测试题

项目根目录/
├── ans.md                               # 新增：评测结果输出
```

### 3.3 抽象基类

```python
# app/rag/strategies/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from langchain_core.documents import Document


@dataclass
class RetrievalResult:
    """检索结果"""
    documents: list[Document]           # 排序后的文档列表
    latency_ms: float = 0.0             # 检索耗时（毫秒）
    trace: dict = field(default_factory=dict)  # 检索过程详情


class BaseRetrieverStrategy(ABC):
    """检索策略抽象基类"""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """执行检索，返回排序后的文档列表"""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """策略唯一标识（如 'bm25', 'dense'）"""
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        """策略展示名称（如 'BM25 稀疏检索'）"""
        ...
```

### 3.4 策略注册表

```python
# app/rag/strategies/__init__.py
from app.rag.strategies.base import BaseRetrieverStrategy

_REGISTRY: dict[str, type[BaseRetrieverStrategy]] = {}

def register(strategy_id: str):
    """装饰器：注册策略类"""
    def wrapper(cls):
        _REGISTRY[strategy_id] = cls
        return cls
    return wrapper

def get_strategy(strategy_id: str, **kwargs) -> BaseRetrieverStrategy:
    """根据 ID 获取策略实例"""
    if strategy_id not in _REGISTRY:
        raise ValueError(f"Unknown strategy: {strategy_id}, available: {list(_REGISTRY.keys())}")
    return _REGISTRY[strategy_id](**kwargs)

def list_strategies() -> list[str]:
    """列出所有已注册策略 ID"""
    return list(_REGISTRY.keys())

# 导入所有策略模块以触发注册
from app.rag.strategies import bm25_strategy
from app.rag.strategies import dense_strategy
from app.rag.strategies import hybrid_rrf_strategy
from app.rag.strategies import hybrid_rrf_rerank_strategy
```

### 3.5 各策略实现要点

**策略 A - BM25**：
- 使用 `rank_bm25` 库 + `jieba` 中文分词
- 在初始化时加载所有文档并构建 BM25 索引
- 新增依赖：`rank_bm25`, `jieba`

**策略 B - Dense**：
- 复用现有 `app/rag/retriever.py` 中的 FAISS 向量库
- 直接调用 `similarity_search_with_score`

**策略 C - Hybrid RRF**：
- 内部同时持有 BM25 索引和 FAISS 向量库
- 双路各召回 top_k*2 个候选
- 用 RRF 公式融合：`score(d) = Σ 1/(k + rank_i(d))`，默认 k=60

**策略 D - Hybrid RRF + Rerank**：
- 继承策略 C 的混合召回
- 对 RRF 融合后的 top_k*2 结果，调用现有 `app/rag/reranker.py` 的 BGE Reranker 精排
- 返回精排后的 top_k 结果

---

## 4. 评测指标体系

### 4.1 检索层指标（不经过 LLM，纯测检索质量）

| 指标 | 计算方式 | 含义 | 值域 |
|------|---------|------|------|
| **Hit@5** | 命中题数 / 总题数 × 100% | top_5 结果中是否包含 gold_section 的内容 | 0~100% |
| **Recall@5** | Σ(每题命中的 gold 数 / 每题 gold 总数) / N × 100% | top_5 结果覆盖了多少标准答案来源 | 0~100% |
| **MRR** | Σ(1/首个命中排名) / N | 第一个相关结果的平均排名倒数 | 0~1 |
| **nDCG@5** | DCG@5 / IDCG@5 | 考虑排名位置权重的检索质量 | 0~1 |

### 4.2 效率指标

| 指标 | 计算方式 | 含义 |
|------|---------|------|
| **Avg Latency** | 总检索耗时 / 题数 | 平均每题检索延迟（ms） |
| **P95 Latency** | 第 95 百分位延迟 | 尾部延迟（ms） |

### 4.3 命中判定规则

对每道测试题的每个 `gold_section`，检查 top_k 检索结果中是否有文档的 `page_content` **包含** gold_section 中的关键文本片段（子串匹配）。

---

## 5. 评测脚本设计

### 5.1 单一脚本 `scripts/benchmark.py`

```
执行流程：
┌─────────────────────────────────────────────┐
│         python scripts/benchmark.py          │
├─────────────────────────────────────────────┤
│                                              │
│  1. 读取 data/eval_cases.json（20 题）        │
│  2. 初始化 4 个策略实例                        │
│  3. 对每个策略：                               │
│     ├── 对每道题执行 retrieve()                │
│     ├── 计算 Hit/Recall/MRR/nDCG             │
│     └── 记录 latency                         │
│  4. 汇总各策略指标                             │
│  5. 生成 ans.md 写入项目根目录                  │
│                                              │
└─────────────────────────────────────────────┘
```

### 5.2 使用方式

```bash
# 运行全部 4 组实验，结果写入 ans.md
python scripts/benchmark.py

# 只运行指定策略
python scripts/benchmark.py --strategies bm25,dense
```

### 5.3 ans.md 输出格式

```markdown
# RAG 检索策略对照实验结果

## 实验概况
- 测试时间：2026-03-03 14:30:00
- 测试题数：20
- Top K：5

## 指标对比总览

| 排名 | 策略 | Hit@5 | Recall@5 | MRR | nDCG@5 | 平均延迟 | P95 延迟 |
|------|------|-------|----------|-----|--------|---------|---------|
| 1 | D: Hybrid+Rerank | 95.0% | 87.5% | 0.891 | 0.856 | 420ms | 680ms |
| 2 | C: Hybrid RRF | 90.0% | 82.5% | 0.845 | 0.812 | 180ms | 310ms |
| 3 | B: Dense | 80.0% | 72.5% | 0.756 | 0.723 | 120ms | 210ms |
| 4 | A: BM25 | 65.0% | 55.0% | 0.612 | 0.578 | 30ms | 50ms |

## 综合分析
- 最高质量：D (Hybrid+Rerank)
- 最低延迟：A (BM25)
- 最佳性价比：C (Hybrid RRF)

## 各题详细结果

### 策略 A: BM25

| # | 问题 | Hit | MRR | 延迟 | 命中的文档片段 |
|---|------|-----|-----|------|--------------|
| 1 | 为了防震，客厅里的... | Y | 1.000 | 25ms | [来源1] A. 客厅... |
| 2 | 家庭应急物资中... | N | 0.000 | 30ms | - |
...

### 策略 B: Dense
（同上格式）

### 策略 C: Hybrid RRF
（同上格式）

### 策略 D: Hybrid + Rerank
（同上格式）
```

---

## 6. 实施步骤

### Step 1：搭建策略抽象层

**目标**：创建可插拔的检索策略架构

| 文件 | 操作 | 说明 |
|------|------|------|
| `app/rag/strategies/__init__.py` | 新建 | 注册表 + 工厂函数 |
| `app/rag/strategies/base.py` | 新建 | 抽象基类 + RetrievalResult 数据类 |

### Step 2：实现 4 种检索策略

| 文件 | 操作 | 说明 |
|------|------|------|
| `app/rag/strategies/bm25_strategy.py` | 新建 | 策略 A: BM25（rank_bm25 + jieba） |
| `app/rag/strategies/dense_strategy.py` | 新建 | 策略 B: Dense（复用现有 FAISS 检索） |
| `app/rag/strategies/hybrid_rrf_strategy.py` | 新建 | 策略 C: BM25+Dense RRF 融合 |
| `app/rag/strategies/hybrid_rrf_rerank_strategy.py` | 新建 | 策略 D: Hybrid RRF + BGE Reranker |

**新增依赖**：`rank_bm25`、`jieba`

### Step 3：构建评测脚本

| 文件 | 操作 | 说明 |
|------|------|------|
| `scripts/benchmark.py` | 新建 | 一键评测脚本（读取 eval_cases.json → 跑 4 组 → 输出 ans.md） |

### Step 4：运行评测

```bash
pip install rank_bm25 jieba       # 安装依赖
python scripts/benchmark.py       # 一键运行，结果写入 ans.md
```

---

## 7. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| BM25 中文分词质量影响检索效果 | 中 | 使用 jieba 精确模式，必要时加载自定义词典 |
| Reranker API 限流或超时 | 低 | 已有 fallback 逻辑（降级为原始排序），记录失败率 |
| 评测集 20 题统计显著性有限 | 中 | 先用 20 题验证框架可用性，后续可扩展题目 |
| 策略 D 延迟较高 | 低 | 评测时记录延迟数据，作为决策参考 |

---

## 8. SESSION_ID（供 /ccg:execute 使用）

- CODEX_SESSION: `019ca99d-e576-7722-9243-0bfe5574e15c`
- GEMINI_SESSION: 无（API Key 未配置）

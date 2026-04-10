# 地震应急问答系统技术文档（开发执行版）

版本：v2.1  
项目路径：`D:\Python\DisasterAssistant`  
适用对象：后端开发、算法开发、前端开发、测试  
核心约束：**Python 实现 + LangChain + DeepSeek API**

---

## 1. 项目目标

构建一个面向普通用户的地震应急问答系统，覆盖震前、震时、震后场景，支持多轮对话与会话管理。
系统基于本地手册知识库回答用户问题，通过 RAG 技术提供准确、结构化的应急指导。
毕业设计重点验证 RAG 优化效果：检索准确率、幻觉率、响应速度、引用可靠性。

---

## 2. 范围与边界

### 2.1 功能范围（本期必须）
- Web 问答页面（聊天式）。
- 多轮对话（同会话记忆上下文）。
- 会话管理：新建、查看历史、重命名、删除、搜索。
- 回答固定结构：
  1) 先做什么（立即行动）
  2) 后续建议动作
  3) 来源引用
- 本地知识库：`EarthQuakeHandBook.md`。
- 低置信度时：动态追问 1-2 个关键槽位（地点/人群/阶段）。

### 2.2 非目标（本期不做）
- 语音输入/输出。
- 移动端原生 App。
- 自动报警联动。
- 会话导出（CSV/PDF）。

---

## 3. 技术栈（Python + LangChain）

### 3.1 后端
- Python 3.12
- FastAPI（HTTP API + SSE 流式输出）
- LangChain（RAG 编排）
- Pydantic（输入输出 Schema）
- SQLAlchemy + Alembic（数据模型与迁移）
- SQLite（开发）/PostgreSQL（部署）

### 3.2 RAG 与模型
- Embedding：`BAAI/bge-m3`
  - 在线：Silicon Flow API（开发阶段推荐，免费额度充足，OpenAI 兼容格式）
  - 本地：`sentence-transformers` 加载（需 GPU 或 8GB+ 内存）
  - 通过 `EMBEDDING_PROVIDER` 环境变量切换，两种方式使用同一模型，向量一致无需重建索引
- Vector DB：FAISS（`faiss-cpu`，纯文件持久化，无额外进程依赖）
- Reranker：`bge-reranker-v2-m3`（G2/G3 实验组必需，通过 Silicon Flow API 调用）
- LLM：DeepSeek Chat API（主生成模型）

### 3.3 前端（轻量）
- FastAPI + Jinja2 模板 + 原生 JS（避免额外 Node 依赖）
- 页面包含：聊天区、会话列表、来源展示区

### 3.4 核心依赖清单

```text
# Web 框架
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
jinja2>=3.1.0
python-multipart>=0.0.9
sse-starlette>=2.0.0

# 数据库
sqlalchemy>=2.0.0
alembic>=1.13.0
aiosqlite>=0.20.0

# RAG / LLM
langchain>=0.3.0
langchain-community>=0.3.0
langchain-openai>=0.2.0
langchain-text-splitters>=0.3.0
faiss-cpu>=1.8.0
openai>=1.50.0

# 工具
pydantic>=2.9.0
pydantic-settings>=2.5.0
python-dotenv>=1.0.0
httpx>=0.27.0
```

> **说明**：Reranker 通过 Silicon Flow API（OpenAI 兼容格式）调用，无需额外安装 `sentence-transformers` 或 `FlagEmbedding`。若后续需要本地 Embedding，再追加 `sentence-transformers>=3.0.0` 和 `torch`。

### 3.5 前端页面设计

#### 布局结构（左右分栏）

```text
+--------------------+----------------------------------------+
|                    |            系统标题栏                    |
|   会话列表侧边栏    +----------------------------------------+
|                    |                                        |
|  [+ 新建会话]       |            聊天消息区                   |
|  ──────────        |   (assistant 消息含来源折叠卡片)         |
|  搜索框             |   (追问消息以特殊卡片样式展示)            |
|  ──────────        |                                        |
|  会话1 (当前)       |                                        |
|  会话2              |                                        |
|  会话3              +----------------------------------------+
|                    |  [输入框]                    [发送按钮]   |
+--------------------+----------------------------------------+
```

#### 核心交互行为
- **新建会话**：点击后创建空会话，标题由首条提问自动生成（取前 20 个字）
- **切换会话**：点击侧边栏会话项，主区域加载该会话的历史消息
- **发送消息**：点击发送或按 Enter 键发送；Shift+Enter 换行
- **流式显示**：接收 `token` 事件时逐步追加文本；接收 `structured` 事件后在消息底部渲染来源折叠卡片
- **追问展示**：当返回 `clarify_questions` 时，以带问号图标的卡片展示，用户点击或直接输入回答
- **会话管理**：右键或悬浮菜单显示重命名/删除选项
- **空状态**：无会话时显示引导提示"点击左上角新建会话开始提问"
- **加载状态**：发送后输入框禁用，显示加载动画，流式完成后恢复

---

## 4. 系统架构

```text
Browser UI
  -> FastAPI API Layer
      -> Session Service (SQLite/Postgres)
      -> RAG Orchestrator (LangChain)
          -> Query Preprocessor (改写 + 槽位识别)
          -> Local Retriever (FAISS Vector Store)
          -> Reranker (G2/G3 启用)
          -> Answer Generator (DeepSeek)
          -> Citation Builder
      -> Error Handler (统一错误处理)
      -> Logs / Metrics / Eval Store
```

---

## 5. 代码目录规划

```text
DisasterAssistant/
  app/
    main.py
    api/
      routers/
        chat.py
        sessions.py
        eval.py
    core/
      config.py
      logging.py
      db.py
    models/
      session.py
      message.py
      eval.py
    schemas/
      chat.py
      session.py
      eval.py
    rag/
      ingest.py
      splitter.py
      embeddings.py
      retriever.py
      reranker.py
      chain.py
      prompt.py
      citation.py
    services/
      chat_service.py
      session_service.py
      eval_service.py
    web/
      templates/
        index.html
      static/
        app.js
        app.css
  data/
    raw/EarthQuakeHandBook.md
    processed/
  scripts/
    build_index.py
    run_eval.py
  tests/
    test_api.py
    test_retrieval.py
    test_generation.py
  TECH_SPEC.md
```

---

## 6. 数据模型（最小可用）

### 6.1 sessions
- `id` (PK)
- `title`
- `created_at`
- `updated_at`

### 6.2 messages
- `id` (PK)
- `session_id` (FK sessions.id)
- `role` (`user`/`assistant`/`system`)
- `content`
- `metadata_json`（引用来源、耗时等结构化信息）
- `created_at`

> **说明**：chunk 元信息（章节路径、版本等）直接存储在 FAISS docstore 的 metadata 中，不再单独建表，避免数据双写和一致性问题。

### 6.3 eval_cases
- `id` (PK)
- `question`
- `gold_points_json`
- `gold_sections_json`
- `category`（震前/震时/震后/特殊场景）

### 6.4 eval_runs
- `id` (PK)
- `variant`（G1/G2/G3）
- `case_id` (FK eval_cases.id)
- `answer_json`
- `metrics_json`
- `created_at`

---

## 7. API 契约

## 7.1 会话接口

**新建会话** `POST /api/sessions`
```json
// 请求体（可选，不传则自动生成标题）
{ "title": "地震避险咨询" }
// 响应体
{ "id": "uuid", "title": "地震避险咨询", "created_at": "2026-02-19T10:00:00Z", "updated_at": "2026-02-19T10:00:00Z" }
```

**会话列表/搜索** `GET /api/sessions?keyword=地震`
```json
// 响应体
{
  "items": [
    { "id": "uuid", "title": "地震避险咨询", "created_at": "...", "updated_at": "...", "message_count": 5 }
  ],
  "total": 1
}
```

**重命名** `PATCH /api/sessions/{id}`
```json
// 请求体
{ "title": "新标题" }
// 响应体
{ "id": "uuid", "title": "新标题", "updated_at": "..." }
```

**删除** `DELETE /api/sessions/{id}`
```json
// 响应体
{ "success": true }
```

**会话消息** `GET /api/sessions/{id}/messages?limit=50&offset=0`
```json
// 响应体
{
  "items": [
    { "id": "uuid", "role": "user", "content": "地震时怎么办？", "created_at": "..." },
    { "id": "uuid", "role": "assistant", "content": "...", "metadata_json": { "sources": [...], "latency_ms": 3120 }, "created_at": "..." }
  ],
  "total": 10
}
```

## 7.2 问答接口
- `POST /api/chat/query`

请求体：
```json
{
  "session_id": "uuid",
  "query": "地震时在高层办公室怎么办？"
}
```

返回体（SSE 流式，分事件类型）：
```text
event: token
data: {"text": "1. 立即蹲下..."}

event: token
data: {"text": "2. 远离窗户..."}

event: structured
data: {
  "answer": {
    "immediate_actions": ["..."],
    "follow_up_actions": ["..."],
    "clarify_questions": [],
    "sources": [
      {
        "type": "local",
        "title": "第2部分：震时避险",
        "section": "3）高层住宅/高层办公楼"
      }
    ]
  },
  "meta": {
    "latency_ms": 3120,
    "variant": "G2"
  }
}

event: done
data: {}
```

> **说明**：采用两阶段输出方案。`token` 事件逐 token 流式输出文本内容，让用户立即看到回答；`structured` 事件在生成完毕后发送完整结构化 JSON，供前端渲染来源卡片等信息；`done` 事件标记流结束。

## 7.3 评测接口
- `POST /api/eval/run?variant=G1|G2|G3`
- `GET /api/eval/report?variant=G1|G2|G3`

---

## 8. RAG 流程设计（LangChain）

### 8.1 索引构建
1. 读取 `EarthQuakeHandBook.md`
2. 第一层切分：使用 `MarkdownHeaderTextSplitter` 按二级/三级标题分层切分，保留章节语义完整性
3. 第二层切分：对超长小节（>800 字），使用 `RecursiveCharacterTextSplitter` 在列表项断点做二次切分，Chunk 长度 400~800 中文字，重叠 80~120
4. 为每个 chunk 保留**父标题链**元数据（如 `"第2部分：震时避险 > 家中震时避险 > 卧室（夜间）"`）
5. 生成 embedding，写入 FAISS 索引（元信息存储在 LangChain FAISS 的 docstore 中）

### 8.2 在线问答链
1. Query 预处理（去噪、槽位识别）
2. 多轮上下文改写（History-aware Retriever）
3. 本地检索 top_k（默认 5）
4. Reranker 重排序（G2/G3 启用），最终取 top 3-4 送入生成模型
5. 构造提示词（强制结构化输出 + 来源约束）
6. DeepSeek 生成 JSON
7. 后处理（来源去重、引用校验）
8. SSE 流式返回前端（token 事件 + structured 事件）并写入日志

#### 多轮历史加载策略
- 从数据库加载当前会话最近 **5 轮**对话（5 条 user + 5 条 assistant = 10 条消息）
- 历史消息总 token 上限 **2000 tokens**，超限时从最早的一轮开始丢弃
- 历史仅用于 **Query 改写**阶段（生成独立的检索查询），不直接拼入生成 prompt
- 示例：用户先问"地震时怎么办"，再问"如果是在高层呢"，改写后的检索查询为"地震时在高层建筑怎么办"

### 8.3 低置信度追问策略

触发条件（任一满足）：
- 检索分数低于阈值
- 来源不足（<1）
- 关键信息槽位缺失

动作：返回 `clarify_questions`，最多 2 个。

交互规则：
- 追问和回答**互斥**：触发追问时，不返回 `immediate_actions` 和 `follow_up_actions`
- 前端以特殊样式（如带问号图标的卡片）展示追问，与普通回答区分
- 用户回答追问后，系统将追问和回答拼接到查询中重新检索，无需额外状态管理

---

## 9. 错误处理策略

### 9.1 统一错误响应格式

```json
{
  "success": false,
  "error": {
    "code": "LLM_TIMEOUT",
    "message": "生成服务暂时不可用，请稍后重试"
  }
}
```

### 9.2 错误场景与处理

| 错误场景 | 错误码 | HTTP 状态 | 处理方式 |
|---------|--------|----------|---------|
| DeepSeek API 超时 | `LLM_TIMEOUT` | 503 | 提示用户稍后重试 |
| DeepSeek API 限流 | `LLM_RATE_LIMIT` | 429 | 返回限流提示，前端可排队重试 |
| DeepSeek API 余额不足 | `LLM_QUOTA` | 503 | 提示服务暂不可用 |
| 检索无结果 | `RETRIEVAL_EMPTY` | 200 | 触发追问或返回兜底回答 |
| 数据库连接失败 | `DB_ERROR` | 500 | 返回通用错误信息 |
| 向量库查询异常 | `VECTOR_DB_ERROR` | 500 | 返回通用错误信息 |
| 请求参数校验失败 | `VALIDATION_ERROR` | 422 | 返回具体字段错误信息 |

### 9.3 原则
- 错误信息不泄露内部实现细节（如堆栈、API Key 等）
- 所有异常统一通过 FastAPI Exception Handler 捕获
- 关键错误写入日志，便于排查

---

## 10. 提示词与输出约束

模型必须遵循：
- 回答中文。
- 输出结构必须包含：
  - `immediate_actions`（详细步骤，3-10 条，视问题复杂度而定）
  - `follow_up_actions`
  - `sources`
- 禁止编造来源；来源必须来自检索上下文。
- 证据不足时优先追问，不得硬答。

### 10.1 System Prompt 模板

```text
你是一个专业的地震应急指导助手。你的唯一知识来源是下方提供的【参考资料】。

## 回答规则
1. 必须用中文回答
2. 必须严格基于【参考资料】中的内容回答，禁止使用自身知识编造信息
3. 每条行动建议必须具体可执行，避免笼统表述（如"注意安全"应改为"用手臂护住头部和颈部"）
4. 来源引用必须标注参考资料中的实际章节路径

## 输出格式
请严格按以下 JSON 格式输出，不要输出任何其他内容：
{
  "immediate_actions": ["具体步骤1", "具体步骤2", ...],
  "follow_up_actions": ["后续建议1", "后续建议2", ...],
  "sources": [
    {"type": "local", "title": "章节大标题", "section": "具体小节路径"}
  ],
  "clarify_questions": []
}

## 特殊情况处理
- 如果参考资料不足以回答用户问题，将 immediate_actions 和 follow_up_actions 设为空数组，在 clarify_questions 中提出 1-2 个澄清问题
- clarify_questions 示例：["请问您目前所处的具体位置（室内/室外/车内）？", "请问周围有老人或行动不便的人吗？"]

## 参考资料
{context}
```

### 10.2 Query 改写 Prompt 模板（多轮对话用）

```text
根据下面的对话历史和最新提问，生成一个独立的、完整的检索查询。
要求：
- 将代词和省略的主语补全
- 合并历史中的关键约束条件
- 只输出改写后的查询，不要输出其他内容

对话历史：
{chat_history}

最新提问：{question}

改写后的查询：
```

---

## 11. 评估方案（论文与系统共用）

## 11.1 实验分组
- `G1 基线`：本地向量检索 + 生成
- `G2 检索优化`：G1 + 查询改写 + rerank
- `G3 全量优化`：G2 + 输出约束 + 引用校验 + 追问机制

### 实验分组切换机制

通过 `variant` 参数控制 RAG 链路的组装方式，共用同一套代码，无需维护多套 chain：

```python
# app/rag/chain.py 中根据 variant 参数组装不同的链路
def build_chain(variant: str):
    # G1: 基线
    # - 直接用原始 query 检索（不改写）
    # - 不使用 reranker
    # - 使用基础 prompt（不含输出约束和引用校验指令）

    # G2: 检索优化
    # - 启用 query 改写（History-aware Retriever）
    # - 启用 reranker 重排序
    # - 使用基础 prompt

    # G3: 全量优化
    # - 启用 query 改写
    # - 启用 reranker 重排序
    # - 使用增强 prompt（含结构化输出约束 + 引用校验指令 + 追问机制）
```

| 功能模块 | G1 | G2 | G3 |
|---------|----|----|-----|
| 向量检索 | Y | Y | Y |
| Query 改写 | - | Y | Y |
| Reranker | - | Y | Y |
| 结构化输出约束 | - | - | Y |
| 引用校验 | - | - | Y |
| 追问机制 | - | - | Y |

> **说明**：日常使用默认走 G3（最优配置）。评测脚本通过 `POST /api/eval/run?variant=G1` 参数指定分组。每次评测自动记录 variant 到 `eval_runs` 表中。

## 11.2 评测集
- 最低 20 题（已定）
- 建议分布：震前/震时/震后/特殊场景 = 5/5/5/5
- 每题需标注：
  - 标准要点 `gold_points`
  - 期望章节 `gold_sections`

## 11.3 指标定义（重点）
- **检索准确率（主）**
  - `Hit@k = 命中题数 / 总题数`
  - `MRR = 平均倒数排名`
- **幻觉率（主）**
  - `Hallucination Rate = 无证据事实条数 / 事实总条数`
- **引用正确率**
  - `Citation Precision = 正确引用条数 / 引用总条数`
- **响应时延**
  - 平均时延、P95 时延

## 11.4 打分流程
1. 脚本批量跑 20 题，保存答案与检索日志
2. 自动算 Hit@k、MRR、时延
3. 人工双评（2 人）标注幻觉与可执行性
4. 分歧由第三人仲裁
5. 输出对比表与显著案例分析

---

## 12. 开发任务拆解（可直接排期）

### 阶段 A：基础设施
- 项目骨架、配置管理、数据库迁移
- 会话 CRUD API
- 前端基础页面与聊天框
- 统一错误处理机制

### 阶段 B：本地 RAG
- 手册两层切分（Markdown 标题 + 字符长度）、向量索引构建脚本
- Embedding 方案配置（在线/本地可切换）
- LangChain 检索链路与回答结构化
- 来源展示（本地章节 + 父标题链）
- SSE 两阶段输出（token 流式 + structured 结构化）

### 阶段 C：多轮与追问
- 历史会话上下文改写
- 低置信度动态追问机制（互斥交互）

### 阶段 D：评测系统
- 20 题评测集录入
- G1/G2/G3 实验脚本
- 指标统计与报告页

### 阶段 E：测试与验收
- 单元测试/接口测试
- 样例回归测试
- 答辩演示脚本整理

---

## 13. 验收标准（DoD）

- 能完成端到端问答，多轮可用。
- 回答结构固定（immediate_actions / follow_up_actions / sources），步骤数视问题复杂度合理（3-10 条）。
- 能显示本地知识库来源（章节路径）。
- 低置信度时能触发追问，追问交互流程完整。
- 会话管理功能全部可用（增删改查+搜索）。
- 20 题评测可重复运行，并输出 G1/G2/G3 对比结果。
- 论文可直接引用系统日志与指标表。
- 所有错误场景有合理的用户提示，不泄露内部信息。

---

## 14. 环境变量清单

`.env` 示例：

```env
APP_ENV=dev
APP_HOST=0.0.0.0
APP_PORT=8000

DEEPSEEK_API_KEY=your_deepseek_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

EMBEDDING_PROVIDER=online
# online: Silicon Flow API（推荐，免费额度充足，与本地 BGE-M3 向量一致）
# local: 本地 BAAI/bge-m3（需 GPU 或 8GB+ 内存）
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_API_KEY=your_siliconflow_key
EMBEDDING_API_BASE=https://api.siliconflow.cn/v1

RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_API_KEY=your_siliconflow_key
RERANKER_API_BASE=https://api.siliconflow.cn/v1

FAISS_INDEX_DIR=./data/faiss_index

DB_URL=sqlite:///./data/app.db
```

---

## 15. 风险与应对

- **风险**：幻觉率偏高
  **应对**：提高检索阈值、加强引用约束、证据不足时强制追问。

- **风险**：响应慢
  **应对**：控制 top_k（默认 5）、启用 rerank 精选、支持 SSE 流式输出。

- **风险**：Embedding 模型本地部署硬件不足
  **应对**：通过 `EMBEDDING_PROVIDER` 环境变量切换在线/本地方案，开发阶段使用在线 API。

- **风险**：DeepSeek API 不稳定（超时/限流）
  **应对**：统一错误处理，返回友好提示，前端支持重试。

- **风险**：Chunk 切分破坏语义完整性
  **应对**：采用两层切分策略（Markdown 标题 + 字符长度），保留父标题链元数据。

---

## 16. 交付物清单

- 可运行系统（前后端）
- 数据库与向量索引构建脚本
- 评测脚本与 20 题测试集
- 三组实验报告（G1/G2/G3）
- 部署与运行说明（README）


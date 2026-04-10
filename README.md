# DisasterAssistant

一个面向地震应急场景的 RAG 问答系统，基于 FastAPI、SQLite 和 FAISS 构建，支持多轮会话、流式回答、检索策略对比与自动评测。

## 项目特性

- 基于灾害领域知识库进行问答
- 支持 SSE 流式输出
- 支持会话创建、重命名、删除和历史消息查看
- 支持多种检索策略与实验分组 `G1 / G2 / G3`
- 内置评测接口与批量脚本
- 使用 SQLite 持久化会话和评测结果

## 技术栈

- FastAPI
- SQLAlchemy + SQLite
- LangChain
- FAISS
- Jinja2
- Uvicorn

## 目录结构

```text
DisasterAssistant/
├─ app/                  # 应用主代码
│  ├─ api/               # API 路由
│  ├─ core/              # 配置、数据库、日志
│  ├─ models/            # 数据模型
│  ├─ rag/               # 检索、召回、重排、链路
│  ├─ services/          # 业务服务
│  └─ web/               # 前端页面与静态资源
├─ alembic/              # 数据库迁移相关文件
├─ data/                 # 样例数据、原始文档、评测数据
├─ scripts/              # 索引构建、评测、对比脚本
├─ tests/                # 测试目录
├─ main.py               # 启动入口
└─ requirements.txt      # 依赖列表
```

## 环境要求

- Python 3.10+
- 可访问所配置的大模型与向量服务

## 安装依赖

```bash
pip install -r requirements.txt
```

## 环境变量

项目通过根目录下的 `.env` 文件加载配置，常用项如下：

```env
APP_ENV=dev
APP_HOST=0.0.0.0
APP_PORT=8000

DEEPSEEK_API_KEY=your_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

EMBEDDING_PROVIDER=online
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_API_KEY=your_api_key
EMBEDDING_API_BASE=https://api.siliconflow.cn/v1

RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_API_KEY=your_api_key
RERANKER_API_BASE=https://api.siliconflow.cn/v1

FAISS_INDEX_DIR=./data/faiss_index
DB_URL=sqlite+aiosqlite:///./data/app.db
DEFAULT_VARIANT=G3
```

说明：

- `.env` 已被 `.gitignore` 忽略，不会提交到仓库
- 如果没有先构建索引，问答链路通常无法正常召回文档

## 构建索引

首次运行前，建议先构建知识库索引：

```bash
python scripts/build_index.py
```

知识源文件当前位于：

- `data/raw/EarthQuakeHandBook.md`

## 启动项目

```bash
python main.py
```

默认访问地址：

- Web 页面：`http://127.0.0.1:8000/`
- OpenAPI 文档：`http://127.0.0.1:8000/docs`

## 主要接口

### 会话管理

- `POST /api/sessions` 创建会话
- `GET /api/sessions` 获取会话列表
- `PATCH /api/sessions/{session_id}` 重命名会话
- `DELETE /api/sessions/{session_id}` 删除会话
- `GET /api/sessions/{session_id}/messages` 获取消息历史

### 问答接口

- `POST /api/chat/query`
- 返回类型为 `text/event-stream`

### 评测接口

- `POST /api/eval/run?variant=G1|G2|G3`
- `GET /api/eval/report?variant=G1|G2|G3`

## 常用脚本

```bash
python scripts/build_index.py
python scripts/run_eval.py
python scripts/benchmark.py
python scripts/diagnose_dense.py
python scripts/run_g_compare.py
```

## 当前项目定位

这个项目适合用于：

- 灾害应急问答系统原型开发
- RAG 检索策略实验与对比
- 面向课程设计、毕业设计或内部演示的知识问答系统

## License

当前仓库未单独声明许可证。如需开源分发，建议补充 `LICENSE` 文件。

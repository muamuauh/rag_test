# 本地个人知识库 RAG 项目规划

## Context

用户希望从零搭建一个**本地运行的个人知识库 RAG 系统**。当前工作目录 `d:\codes\rag_test` 为空，是全新项目。

经过沟通，已明确以下核心需求：

- **知识库内容**：中英文混合，包含 PDF / Markdown / TXT / Word / PPT / 图片，且 PDF/PPT 中常含图表
- **LLM**：使用云端 API，且需要**多提供商可切换**（OpenAI / Anthropic / 国内 GLM·Qwen·DeepSeek）
- **图像处理策略**：入库时用 VLM（视觉大模型）将图片/图表转为文字描述，再与文本一起 embedding（检索路径统一，效果好且实现简洁）
- **向量库**：Milvus
- **前端**：Streamlit

设计目标：架构清晰、可扩展、对个人电脑友好（Milvus Lite 嵌入式起步），同时保留升级到生产级（Milvus Standalone + Docker）的路径。

---

## 技术选型

| 层 | 选择 | 说明 |
|---|---|---|
| 语言/运行时 | Python 3.11+，**Miniconda 环境** | 环境名固定为 `rag_test`，所有依赖装在该 conda env 中 |
| 文档解析 | **PyMuPDF (fitz)** + **python-docx** + **python-pptx** + **markdown-it-py** | 各格式直接精解析，PyMuPDF 还能直接抽取 PDF 内嵌图片 |
| 复杂版式（可选升级） | **unstructured** 或 **MinerU** | 处理含表格/复杂版式的 PDF；初版可先不用 |
| VLM 图像描述 | 通过 LiteLLM 调云端视觉模型（GPT-4o / Claude / Qwen-VL / GLM-4V） | 入库时一次性生成 caption，结果落盘缓存避免重复花费 |
| 切分 | LlamaIndex `SentenceSplitter`（按 token，带 overlap） | 默认 chunk_size=512, overlap=64 |
| **Embedding** | **BGE-M3**（`BAAI/bge-m3`，本地推理） | **关键推荐**：天然多语言（中英都强），单模型同时支持 dense + sparse + multi-vector，是当前中英混合 + 混合检索场景的最佳开源方案 |
| 向量库 | **Milvus**（开发期用 Milvus Lite，生产升级 Standalone） | 通过 `pymilvus` 客户端；Milvus 2.4+ 原生支持 dense+sparse 混合检索 |
| Reranker | **BAAI/bge-reranker-v2-m3**（本地小模型） | 二阶段重排，对中英混合场景效果显著 |
| LLM 抽象 | **LiteLLM** | 统一 OpenAI/Anthropic/Zhipu/Qwen/DeepSeek 调用接口，切 provider 改一行配置 |
| 前端 | **Streamlit** | 聊天 UI + 文件上传 + 流式输出 + 引用展示 |
| 配置 | `pydantic-settings` + `.env` | API key、模型名、路径等集中管理 |

> **备选评估**：LlamaIndex vs LangChain vs 自建 — 推荐**轻度使用 LlamaIndex 的组件**（`SentenceSplitter`、`SimpleDirectoryReader` 思路），但**核心 pipeline 自己写**，避免被框架抽象绑架，便于调试。

---

## 目录结构

```
d:\codes\rag_test\
├── app\
│   └── streamlit_app.py          # UI 入口（聊天 + 文件上传 + 引用展示）
├── rag\
│   ├── __init__.py
│   ├── config.py                 # Settings (API key / 模型 / 路径)
│   ├── loaders\                  # 各类文档解析器
│   │   ├── pdf_loader.py         # PyMuPDF: 文本块 + 内嵌图片提取
│   │   ├── office_loader.py      # docx + pptx
│   │   ├── markdown_loader.py    # md / txt
│   │   └── image_loader.py       # 独立图片文件
│   ├── parsing\
│   │   ├── vlm_caption.py        # VLM 生成图片描述 + 磁盘缓存
│   │   └── chunker.py            # 包装 SentenceSplitter
│   ├── embedding\
│   │   └── embedder.py           # BGE-M3 封装（dense + sparse 双路输出）
│   ├── vector_store\
│   │   └── milvus_store.py       # Milvus collection 管理 + 混合检索
│   ├── retrieval\
│   │   ├── retriever.py          # query → top-k chunks（dense+sparse 混合）
│   │   └── reranker.py           # bge-reranker-v2-m3 重排
│   ├── llm\
│   │   └── client.py             # LiteLLM 统一封装 + 流式
│   └── pipeline\
│       ├── ingest.py             # CLI：扫描目录 → 解析 → caption → chunk → embed → 入库
│       └── query.py              # 查询 pipeline：retrieve → rerank → prompt → LLM
├── data\
│   ├── raw\                      # 用户放置的原始文档（gitignore）
│   ├── images\                   # 提取/缓存的图片 + caption json
│   └── milvus\                   # Milvus Lite 数据文件
├── tests\
│   └── test_pipeline_smoke.py    # 端到端冒烟测试
├── .env.example                  # 模板（不提交真实 key）
├── .gitignore
├── environment.yml               # conda 环境定义（python=3.11 + 部分依赖）
├── requirements.txt              # pip 依赖（conda 装不到的部分）
└── README.md                     # 使用说明（含 conda env 创建与激活步骤）
```

---

## 关键设计点

### 1. 统一的 Chunk 数据模型

所有 loader 输出统一结构（便于后续 pipeline 处理）：

```python
@dataclass
class Chunk:
    text: str                 # 用于 embed 的文本（图片 caption 也走这里）
    source_path: str          # 原文件路径
    chunk_type: str           # "text" | "image_caption" | "table"
    page: int | None          # PDF/PPT 页码
    image_path: str | None    # 若是图片 caption，原图磁盘路径（前端可展示原图）
    metadata: dict            # 其他扩展
```

### 2. VLM 描述缓存策略（重要：省钱）

- 每张图片用其内容 hash（SHA256）作为缓存 key
- caption 写入 `data/images/captions.jsonl`
- 重新入库 / 跑同一文件不会重复调用 VLM

### 3. 混合检索（Hybrid Search）

利用 BGE-M3 的 dense + sparse 双输出：
- Dense 向量负责**语义召回**
- Sparse（学习版 BM25）负责**关键词召回**
- Milvus 端 RRF 融合（`hybrid_search` API）
- 召回 top-30 → bge-reranker-v2-m3 重排 → 取 top-5 给 LLM

### 4. 多 Provider 切换

`config.py` 中 `LLM_MODEL="anthropic/claude-sonnet-4-6"` 这种 LiteLLM 标准格式，VLM 同理。换 provider 只改 `.env` 一行。

### 5. 引用与可解释性

LLM 回答时，prompt 中给每个 chunk 一个编号 `[1][2]…`，要求模型在结论后标注引用源；Streamlit 侧把 `[1]` 渲染成可点开的卡片，显示原文片段 + 文件名 + 页码 +（如有）原图缩略图。

---

## 实施步骤（分阶段，可独立验证）

### Stage 1：项目骨架 + 配置层
- **创建 conda 环境**：
  ```
  conda create -n rag_test python=3.11 -y
  conda activate rag_test
  ```
- 编写 `environment.yml`（含 python=3.11、pip 依赖入口）+ `requirements.txt`（核心依赖：`pymilvus`, `litellm`, `FlagEmbedding`, `pymupdf`, `python-docx`, `python-pptx`, `markdown-it-py`, `streamlit`, `pydantic-settings`, `torch`(CPU 或 CUDA 视情况)，`transformers` 等）
- `rag/config.py` + `.env.example`
- 验证：在 `rag_test` env 下执行 `python -c "from rag.config import settings; print(settings)"` 不报错

### Stage 2：文档加载器
- 实现 4 个 loader，输出统一 `Chunk` 列表（图片只占位、暂不调 VLM）
- 验证：写一个脚本对 `data/raw/` 跑一遍，打印解析出的 chunk 数量

### Stage 3：VLM 描述 + 缓存
- `parsing/vlm_caption.py`：接受图片路径 → 调 LiteLLM 视觉模型 → 缓存 + 返回 caption
- 验证：手动喂一张图，得到合理的中文描述

### Stage 4：Embedding + Milvus
- `embedding/embedder.py`：`FlagEmbedding.BGEM3FlagModel` 封装，输出 dense + sparse
- `vector_store/milvus_store.py`：建 collection（含 dense + sparse 字段 + metadata）+ 插入 + `hybrid_search`
- 验证：插入 100 条 chunk，hybrid_search 返回合理结果

### Stage 5：Reranker + 查询 pipeline
- `retrieval/reranker.py`：本地加载 bge-reranker-v2-m3
- `pipeline/query.py`：retrieve → rerank → prompt 拼装 → LiteLLM 流式调用
- 验证：CLI 提问，看到带引用的回答

### Stage 6：Streamlit 前端
- 聊天界面 + 流式输出
- 侧边栏：上传文件触发 ingest、查看已入库文档列表
- 引用 chip：点击展开原文 + 原图

### Stage 7：端到端冒烟测试
- 准备 mini 数据集（1 PDF 含图、1 docx、1 md、1 png）
- 一键 `python -m rag.pipeline.ingest data/raw/`
- 用 `streamlit run app/streamlit_app.py` 启动，问几个跨文档问题验证

---

## 关键文件改动清单（实施时聚焦这些）

- `environment.yml` / `requirements.txt` — conda + pip 依赖
- `.env.example` — 配置模板
- `rag/config.py` — 配置入口
- `rag/loaders/pdf_loader.py` — 最复杂，含图像抽取逻辑
- `rag/parsing/vlm_caption.py` — VLM + 缓存（避免重复花钱）
- `rag/embedding/embedder.py` — BGE-M3 dense+sparse 双路
- `rag/vector_store/milvus_store.py` — Milvus 混合检索 schema
- `rag/pipeline/ingest.py` — 全流程串联入口
- `rag/pipeline/query.py` — 查询入口
- `app/streamlit_app.py` — UI

---

## 验证方法（端到端）

1. **环境**：
   ```
   conda create -n rag_test python=3.11 -y
   conda activate rag_test
   pip install -r requirements.txt
   ```
   然后填好 `.env` 中至少一个 provider 的 API key。后续所有命令都默认在已激活的 `rag_test` env 中运行。
2. **入库**：把 3-5 个混合格式样本（含一份带图 PDF）放 `data/raw/`，运行 `python -m rag.pipeline.ingest data/raw/`
3. **CLI 查询**：`python -m rag.pipeline.query "问一个跨文档的问题"` → 期望返回带 `[1][2]` 引用的回答
4. **Web 验证**：`streamlit run app/streamlit_app.py` → 浏览器交互，验证：
   - 流式输出正常
   - 引用可展开看到原文片段
   - 图表问题能命中（说明 VLM caption 生效）
5. **切 provider**：改 `.env` 中 `LLM_MODEL` 从 `openai/gpt-4o` 切到 `anthropic/claude-sonnet-4-6`，重跑同一查询，应仍工作
6. **冒烟测试**：`pytest tests/test_pipeline_smoke.py`

---

## 后续可扩展（不在初版范围）

- 增量更新 / 文件变更监控（watchdog）
- 文档级权限 / 多知识库
- 升级到 Milvus Standalone（Docker）以承载更大规模
- 加入 query rewriting / HyDE / multi-query 提升召回
- 接入 Obsidian / Notion 作为数据源

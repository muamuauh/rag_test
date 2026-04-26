# 架构与技术细节

本文从数据流、模块职责、关键设计决策三个维度介绍系统实现，便于二次开发与调优。

---

## 1. 整体数据流

### 1.1 入库 (Ingest)

```
data/raw/*  ──►  loaders  ──►  Chunk[]  ──►  VLM caption  ──►  chunker
                  (PDF/                        (image_caption     (split text
                   docx/pptx/                   类型填 text)       chunks)
                   md/img)                          │
                                                    ▼
                                           BGE-M3 embed
                                       (dense + sparse 双路)
                                                    │
                                                    ▼
                                          MilvusStore.insert
                                       (collection: rag_chunks)
```

**关键点：**

1. **统一中间结构 `Chunk`**（[rag/schema.py](rag/schema.py)） —— loader 不关心后续怎么用，下游不关心来源是什么格式。
2. **图像也走文本检索路径** —— image_caption 类型的 Chunk 经 VLM 后 `text` 字段被填充，与文本 Chunk 一起 embed、一起入库、一起检索。
3. **VLM 描述按图片 SHA256 缓存** —— 缓存文件 `data/images/captions.jsonl`，重跑同样的图片不会再调 VLM。

### 1.2 查询 (Query)

```
user query  ──►  BGE-M3 encode_query
                 (dense + sparse)
                       │
                       ▼
            MilvusStore.hybrid_search
            (RRF fuse → top-30 candidates)
                       │
                       ▼
              bge-reranker-v2-m3
              (cross-encoder rerank → top-5)
                       │
                       ▼
              LiteLLM chat (stream)
              (system + 编号引用 prompt)
                       │
                       ▼
              user answer (with [1][2] citations)
```

---

## 2. 模块详解

### 2.1 `rag/config.py` — 配置层

基于 `pydantic-settings`，所有可调参数从 `.env` 读取并强类型化。关键字段：

- **模型族**：`llm_model`、`vlm_model`、`embedding_model`、`reranker_model`
- **Milvus**：`milvus_uri`（自动识别本地路径 vs HTTP URL）、`milvus_collection`
- **切分**：`chunk_size=512`，`chunk_overlap=64`
- **检索**：`retrieve_top_k=30`，`rerank_top_k=5`

`milvus_is_remote` 属性用于在 `MilvusStore` 中分支处理 Lite 与 Standalone 两种模式。

### 2.2 `rag/loaders/` — 文档加载器

每个 loader 输入一个文件路径，输出 `list[Chunk]`，**不做切分**（由下游 chunker 负责）。

| 文件 | 处理对象 | 关键实现 |
|---|---|---|
| `pdf_loader.py` | PDF | PyMuPDF（fitz）逐页 `get_text("text")`，`page.get_images(full=True)` 提取内嵌图片字节 |
| `office_loader.py` | DOCX / PPTX | `python-docx` 段落合并；`python-pptx` 按 slide 收集 text frame + picture shapes |
| `markdown_loader.py` | MD / TXT | 简单读 utf-8 |
| `image_loader.py` | 独立图片 | 复制到 `data/images/<sha>.<ext>`，留 image_caption 占位 |
| `__init__.py::load_directory()` | 任意目录 | 按扩展名 dispatch，递归 `rglob` |

**图片落盘策略**：所有 loader 提取出的图片字节先做 SHA256 → 写到 `data/images/<hash>.<ext>` → Chunk 的 `image_path` 指向该路径。这意味着同一张图（即使在不同文档里）只存一份，VLM 也只调一次。

### 2.3 `rag/parsing/` — 后处理

#### `vlm_caption.py`

- **缓存**：进程级 `_cache: dict[str, str]` + 持久化 `data/images/captions.jsonl`（追加写）
- **Prompt**：要求 VLM 用中文详细描述图片，针对图表特别要求说明类型/坐标轴/趋势
- **调用**：`litellm.completion(model=settings.vlm_model, messages=[multimodal])`
  - 图片以 `data:image/<ext>;base64,...` 的 inline data URL 传，避免依赖外部存储

#### `chunker.py`

- 用 `llama_index.core.node_parser.SentenceSplitter`（按 token，含句子边界感知）
- 仅切分 `chunk_type == "text"` 且 `text` 非空的 Chunk；image_caption 不切，保持单条

### 2.4 `rag/embedding/embedder.py` — Embedder 抽象

接口（[rag/embedding/embedder.py](rag/embedding/embedder.py)）：

```python
class Embedder(Protocol):
    dense_dim: int
    has_sparse: bool
    def encode(self, texts) -> EncodeResult           # batch encode
    def encode_query(self, text) -> QueryEncodeResult # 单 query encode
```

两个实现 + 工厂：

| 类 | provider | sparse 来源 |
|---|---|---|
| `LocalBGEM3Embedder` | local | BGE-M3 直接输出 learned sparse（lexical_weights） |
| `CloudEmbedder` | cloud | **不输出**；由 Milvus 端的 BM25 function 从原文计算 |
| `get_embedder()` | — | 按 `settings.embedding_provider` 自动返回正确实例 |

#### LocalBGEM3Embedder

**为什么选 BGE-M3：**

| 需求 | BGE-M3 优势 |
|---|---|
| 中英混合 | 多语言原生支持 |
| 长文档 | max_length 8192 |
| 混合检索 | 单模型同时输出 dense + sparse + multi-vector |

实现要点：

```python
self.model.encode(
    texts,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=False,  # multi-vector 暂未启用，留作扩展点
)
```

返回结构：
- `dense_vecs`: `np.ndarray (N, 1024)`
- `lexical_weights`: `list[dict[int, float]]`（token_id → 权重，可直接喂给 Milvus 的 `SPARSE_FLOAT_VECTOR`）

**单例模式**：`LocalBGEM3Embedder.instance()`，避免每次查询重复加载 ~2GB 模型权重。

#### CloudEmbedder

调 `litellm.embedding(model=settings.cloud_embedding_model, input=texts)`，统一支持 OpenAI / 智谱 / 通义 / Jina / Cohere 等。
- 维度由 `settings.embedding_dim` 指定（必须与模型一致，否则 schema 报错）
- 默认 batch_size=64 自动分批，避免触发单次请求 token 上限
- 不输出 sparse —— 由 Milvus BM25 兜底（见 2.5）

### 2.5 `rag/vector_store/milvus_store.py` — Milvus

`MilvusStore` 在 init 时按两种模式分支建 collection（已有则不覆盖）：

| 模式 | 触发条件 | sparse 怎么来 | dense 维度 |
|---|---|---|---|
| **外部 sparse** | `embedding_provider=local` | 调用方传入（BGE-M3 输出） | 1024 |
| **BM25 function** | `embedding_provider=cloud` | Milvus 内部从 text 字段自动算 | 看 `EMBEDDING_DIM` |

#### Collection schema（共享字段）

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | INT64, primary, auto_id | 主键 |
| `text` | VARCHAR(65535) | 原文片段；BM25 模式下 `enable_analyzer=True` |
| `meta` | JSON | source_path / chunk_type / page / image_path / metadata |
| `dense` | FLOAT_VECTOR(N) | dense 向量，N=embedder.dense_dim |
| `sparse` | SPARSE_FLOAT_VECTOR | learned sparse 或 BM25 sparse |

#### BM25 function 模式特有

```python
schema.add_field(TEXT_FIELD, VARCHAR, max_length=65535,
                 enable_analyzer=True,
                 analyzer_params={"type": settings.milvus_analyzer})
schema.add_function(Function(
    name="text_bm25",
    function_type=FunctionType.BM25,
    input_field_names=[TEXT_FIELD],
    output_field_names=[SPARSE_FIELD],
))
```

- 插入时**不传 sparse 字段**，Milvus 在 server 端按 analyzer 分词、算 BM25 权重写入
- 查询时 sparse 通道传**原始文本字符串**，Milvus 同样在 server 端分词查询
- index 的 `metric_type` 是 `BM25`（不是 `IP`），含 `bm25_k1`、`bm25_b` 调优参数

#### 索引

- `dense`：AUTOINDEX，metric IP（点积；BGE-M3 dense 已归一化）
- `sparse`：SPARSE_INVERTED_INDEX，metric IP

#### 混合检索

```python
dense_req  = AnnSearchRequest(query_dense,  "dense",  IP, limit=30)
sparse_req = AnnSearchRequest(query_sparse, "sparse", IP, limit=30)
hybrid_search(reqs=[dense_req, sparse_req], ranker=RRFRanker(60), limit=30)
```

**RRF（Reciprocal Rank Fusion）**：score = Σ 1/(k + rank_i)，k=60 是经典默认值。dense 与 sparse 各自独立排序后，按倒数排名相加得最终分。

### 2.6 `rag/retrieval/`

#### `retriever.py`

`Retriever` 串联三个组件：embedder → store.hybrid_search → reranker.rerank。query 阶段一次性完成所有检索逻辑，对外只暴露一个 `retrieve(query)` 方法。

#### `reranker.py`

bge-reranker-v2-m3 是一个 cross-encoder：把 (query, passage) 一起喂进去，输出一个相关性得分。比向量相似度更精准但更慢，所以做"一阶段召回 30 → 二阶段精排 5"。

```python
scores = self.model.compute_score([[q, c.text] for c in candidates], normalize=True)
# 按 rerank_score 降序，取 top_k
```

### 2.7 `rag/llm/client.py` — LLM 抽象

#### LiteLLM 模型名约定

LiteLLM 的标准格式 `provider/model_name` 让多 provider 切换零成本：

```python
litellm.completion(model="anthropic/claude-sonnet-4-6", messages=[...])
litellm.completion(model="openai/gpt-4o-mini",          messages=[...])
litellm.completion(model="zhipu/glm-4-plus",            messages=[...])
```

LiteLLM 自动从环境变量读取对应的 API key（`OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `ZHIPUAI_API_KEY` …）。

#### Prompt 模板

System message 强约束：
1. 答案仅依据【参考资料】，不足则明确说明
2. 每个事实/结论后用 `[1][2]` 引用
3. 按用户问题语言回答

User message 把 reranked top-k 拼成编号列表：

```
【参考资料】
[1] 来源: foo.pdf 第 3 页
<chunk text>

[2] 来源: bar.docx
<chunk text>

【问题】
<query>
```

#### 流式输出

`chat_stream()` yield 增量 token，Streamlit 用 `placeholder.markdown(full + "▌")` 实现打字机效果。

### 2.8 `rag/pipeline/`

#### `ingest.py`

CLI 入口：`python -m rag.pipeline.ingest <paths...>`

流程：
1. `load_directory` / `load_file` 拿到原始 Chunk
2. `fill_image_captions` 用 VLM 填 image_caption（失败的丢弃）
3. `split_chunks` 切分文本
4. 批次 embed + insert（带 tqdm 进度条）

#### `query.py`

CLI 入口：`python -m rag.pipeline.query "<question>"`

输出顺序：检索到的引用列表 → 流式回答。

### 2.9 `app/streamlit_app.py`

- **`@st.cache_resource`** 包装 Retriever 与 MilvusStore，避免每次 rerun 重新加载 4GB+ 模型
- **侧边栏**：collection 统计 / 配置展示 / 文件上传入库 / 清空知识库
- **主区**：`st.chat_input` + `st.chat_message`，引用通过 `st.expander` 折叠显示原文与原图

---

## 3. 关键设计决策

### 3.1 为什么图片走 "VLM 转文本" 而不是 multimodal embedding？

| 方案 | 优点 | 缺点 |
|---|---|---|
| **VLM caption（采用）** | 检索路径统一；caption 含语义/数据/趋势，对图表友好；可读性强 | 入库成本（一次性，可缓存） |
| CLIP / jina-clip 多模态 embedding | 入库便宜；以文搜图 | 对复杂图表的语义抽取弱；要维护两套向量空间 |

个人知识库里的图基本都是图表/示意图，VLM caption 是更稳健的选择。

### 3.2 为什么用 Milvus 而不是 Chroma / FAISS？

- **原生混合检索**：dense + sparse + RRF 融合一条 API 搞定，无需手撸
- **后续可平滑升级**：Standalone → Distributed → Zilliz Cloud
- 代价：在 Windows 必须 Docker（已通过 `standalone.bat` 自动化）

### 3.3 为什么用 LiteLLM 而不是各家 SDK？

LiteLLM 把所有 provider 抽象成 OpenAI-style API，且：
- VLM 调用格式统一（OpenAI 的 `image_url` content type 在所有 provider 通用）
- 自动 retry / rate limit / fallback
- 切 provider 改一行配置，不动代码

### 3.4 为什么 chunker 在 caption 之后？

如果先切分再做 caption，可能把 image_caption 的占位 Chunk 当成空文本切丢；先做 caption 让 image_caption 拥有有意义的 `text`，再统一过 chunker（chunker 内部跳过 image_caption 不切）。

### 3.5 为什么要独立的 reranker？

单纯的向量相似度对"语义相近但答非所问"的 case 不敏感。例如查询"capital of China"在小数据集上可能把"Python data science"排在"北京是中国首都"前面（向量空间偶然相近）。reranker 是 cross-encoder，把 query 和 passage 一起编码，能捕获细粒度交互信号，把这类反直觉排序纠正过来。

实测：smoke test 中，"capital of China" 查询经 reranker 后，北京 chunk rerank score 0.99，其他全部 0.00，排序完全正确。

### 3.6 为什么 transformers 锁 `<5.0`？

`transformers 5.x` 移除了 `XLMRobertaTokenizer.prepare_for_model` 等旧 API，FlagEmbedding 的 reranker 还在用。在 FlagEmbedding 适配前，`>=4.40,<5.0` 是兼容窗口。

---

## 4. 性能与成本

### 4.1 资源占用（典型 PC）

| 项目 | 占用 |
|---|---|
| BGE-M3 模型权重 | ~2.3 GB（HF 缓存） |
| bge-reranker-v2-m3 权重 | ~2.4 GB（HF 缓存） |
| 推理时运行内存（CPU）/ 显存（GPU） | CPU 模式 ~3 GB 内存 / CUDA 模式 ~3-4 GB 显存 |
| Milvus Standalone 容器 | ~1 GB RAM idle |
| 单条 chunk Milvus 存储（含双向量 + meta） | ~5-10 KB |

> **"推理时"指什么**：本系统中只有两个本地神经网络模型在跑——**BGE-M3**（每次 embed 文本 / encode query 时）和 **bge-reranker-v2-m3**（每次查询对 30 个候选打分时）。LLM 与 VLM 走云端 API，不占本地资源；Milvus 检索是 C++ 实现的纯向量计算，开销已单列在 "Milvus Standalone 容器"。这一行数字是这两个模型 forward pass 的中间激活（attention 矩阵、token embeddings 等）所需的额外内存/显存。

### 4.2 入库速度（M1 / 普通 i7 CPU）

- 纯文本：~50-100 chunk/s embedding（batch=32）
- 含图片 PDF：受 VLM API 调用速度限制，~1-2 page/s（首次），缓存命中后 100% 复用

### 4.3 查询延迟

- BGE-M3 encode_query：50-100ms（CPU）
- Milvus hybrid_search：5-20ms（数千 chunk 规模）
- Reranker（30 候选）：100-300ms（CPU）
- LLM 流式首 token：取决于 provider，通常 0.5-2s

总体单次查询：**~1-3s 出第一个 token**。

### 4.4 成本（Cloud API）

按 OpenAI gpt-4o-mini 价格估算：
- VLM caption 每张图：~$0.0005-0.002（一次性，缓存复用）
- LLM 单次回答（top-5 chunks ≈ 2k tokens 输入）：~$0.0003

100 个文档 + 200 张图的初始入库 ≈ $0.1-0.4；之后查询基本是 LLM 的钱，单次毫不痛感。

---

## 5. 扩展点

| 想做的事 | 改在哪 |
|---|---|
| 加新文档格式（如 epub） | `rag/loaders/` 加新文件 + `__init__.py::load_file` 加 dispatch |
| 加新 LLM provider | `.env` 改 `LLM_MODEL` 为 LiteLLM 支持的格式即可，无需动代码 |
| 切到 Milvus Standalone 集群 | 改 `MILVUS_URI` 为远端地址；schema 自动建 |
| 引入查询改写 / HyDE / multi-query | 在 `Retriever.retrieve` 之前加变换层 |
| 记忆历史对话 | `chat_stream()` 多传 `messages` 历史；目前每轮独立检索 |
| 多知识库（按用户/项目隔离） | `MILVUS_COLLECTION` 参数化，按 session 切换 |
| 文件变更自动重建索引 | 加 `watchdog` 监听 `data/raw/`，触发增量 ingest |
| 复杂版式 PDF（含表格） | 把 `pdf_loader.py` 的 PyMuPDF 换成 `unstructured` 或 `MinerU` |

---

## 6. 调试技巧

- **看 Milvus 里到底存了什么**：[http://localhost:9091/webui/](http://localhost:9091/webui/) （Milvus 内置 web）
- **embedding 维度对不上**：换 embedding 模型后必须 `standalone.bat delete` 重建（schema 锁定 1024 维）
- **检索结果不对**：先关 reranker 看一阶段命中（`Retriever(reranker=DummyReranker())`），定位问题在召回还是排序
- **VLM 描述不准**：直接看 `data/images/captions.jsonl`，删除对应行重跑会重新生成

---

## 7. 参考资料

- [BGE-M3: Multi-Lingual, Multi-Functionality, Multi-Granularity Embeddings (paper)](https://arxiv.org/abs/2402.03216)
- [Milvus Hybrid Search 官方文档](https://milvus.io/docs/full_text_search_with_milvus.md)
- [LiteLLM 支持的 provider 列表](https://docs.litellm.ai/docs/providers)
- [RRF 论文 (Cormack et al., 2009)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)

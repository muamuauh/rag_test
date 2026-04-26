# 本地个人知识库 RAG

一个**完全本地运行**的个人知识库 RAG 系统：把 PDF / Word / PPT / Markdown / 图片 喂给它，然后用自然语言提问，它会基于你的资料给出**带引用**的回答。

> 中英文混合 · 多模态（VLM 处理图表）· BGE-M3 + Milvus 混合检索 · 多 Provider 可切换

---

## 特性

| 能力 | 说明 |
|---|---|
| 多格式解析 | PDF / DOCX / PPTX / Markdown / TXT / PNG / JPG |
| 多模态 | 入库时 VLM 自动为图片/图表生成中文描述，与文本走统一检索路径 |
| 中英混合 | BGE-M3 多语言 embedding + bge-reranker-v2-m3 重排 |
| 混合检索 | dense（语义）+ sparse（关键词）双路召回，Milvus RRF 融合 |
| Embedding 可云可本地 | `.env` 切 `EMBEDDING_PROVIDER=local` 或 `cloud`；cloud 模式自动用 Milvus BM25 兜 sparse |
| 多 Provider | LiteLLM 统一接口，OpenAI / Anthropic / 智谱 / 通义 / DeepSeek 改一行配置就切 |
| 引用可解释 | 回答附 `[1][2]` 引用，UI 可展开看原文与原图 |
| 缓存省钱 | 图片描述按 SHA256 缓存到 `data/images/captions.jsonl`，重跑不重复花钱 |
| Web UI | Streamlit 聊天界面，支持流式输出与文件上传入库 |

---

## 快速开始（两种方式，任选其一）

### 方式 A：Docker Compose 一键启动（推荐给"只想用"的用户）

只需 **Docker Desktop** + 一个 API key，无需装 Python / Conda。

```bash
# 1. 复制配置模板并填入 API key
cp .env.example .env
# 编辑 .env：至少设置 LLM_MODEL 和对应的 *_API_KEY

# 2. 把要入库的文档放进 data/raw/
mkdir -p data/raw data/images
cp /path/to/your/docs/* data/raw/

# 3. 一键启动（首次构建镜像 + 下载 ~5GB 模型，需要 10-20 分钟）
docker compose up -d

# 4. 浏览器打开
#   Web UI:        http://localhost:8501
#   Milvus 控制台: http://localhost:9091/webui/
```

容器内入库（也可以直接在 Web UI 拖文件入库）：
```bash
docker compose exec app python -m rag.pipeline.ingest data/raw/
```

常用命令：
```bash
docker compose logs -f app    # 看 app 日志
docker compose stop           # 停止（保留数据）
docker compose down           # 停止并删除容器（保留 volumes 数据）
docker compose down -v        # 彻底清理（连知识库/HF 缓存一起删）
```

> **HF 模型缓存挂在 named volume `hf_cache` 里**，第一次下载完之后 `down`+`up` 不会重下；只有 `down -v` 才会清掉。

---

### 方式 B：开发模式（conda 环境，方便改代码）

适合想二次开发、调试的用户。前置条件：
- Miniconda / Anaconda
- Docker Desktop（仅用于跑 Milvus）
- 至少一个 LLM provider 的 API key

#### B.1. 创建 conda 环境

> **CPU 还是 GPU？** 现在有两份依赖文件。没有 NVIDIA 显卡 → CPU；有 NVIDIA 显卡且想加速本地 BGE-M3/Reranker → CUDA。
>
> | 维度 | CPU | CUDA |
> |---|---|---|
> | 依赖文件 | `requirements-cpu.txt` | `requirements-cuda.txt` |
> | torch wheel | `torch` (PyPI 默认) | `torch+cu128` (~2.7GB) |
> | BGE-M3 入库 | ~50-100 chunk/s | ~150-300 chunk/s（实测 RTX 5070 Laptop fp16 ≈ 144） |
> | 显存占用 | 0 | fp16 下 ~1.5GB |
> | `.env` 改动 | `EMBEDDING_DEVICE=cpu` | `EMBEDDING_DEVICE=cuda`，可选 `USE_FP16=true` |

```bash
conda create -n rag_test python=3.11 -y
conda activate rag_test

# 二选一：
pip install -r requirements-cpu.txt    # CPU 默认
# pip install -r requirements-cuda.txt  # NVIDIA GPU 加速 (cu128 wheel ~2.7GB)
```

> **GPU 用户**：装完后跑 `python -c "import torch; print(torch.cuda.is_available())"` 应输出 `True`。如果显示 False 但你确实有 GPU，先 `nvidia-smi` 看驱动版本，再把 `requirements-cuda.txt` 中的 `cu128` 换成你的 CUDA 版本（`cu126` / `cu124` / `cu121` / `cu118`）。

#### B.2. 启动 Milvus

**Windows / macOS / Linux 通用（推荐）：**

```powershell
# 仅首次需要下载脚本
Invoke-WebRequest https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.bat -OutFile standalone.bat

.\standalone.bat start    # 启动
.\standalone.bat stop     # 停止
.\standalone.bat delete   # 清空数据
```

成功后 `docker ps` 能看到 `milvus-standalone (healthy)` 监听 `19530` 端口。

> Linux/macOS 用户也可以直接用 `pip install pymilvus[milvus_lite]`，把 `.env` 中 `MILVUS_URI` 改为本地路径 `./data/milvus/rag_test.db`，免 Docker。

#### B.3. 配置 API key

```bash
cp .env.example .env
# 编辑 .env，填入至少一个 provider 的 key 并设置 LLM_MODEL / VLM_MODEL
```

LiteLLM 的模型名格式如：

| Provider | LLM_MODEL 示例 | VLM_MODEL 示例 |
|---|---|---|
| OpenAI | `openai/gpt-4o-mini` | `openai/gpt-4o-mini` |
| Anthropic | `anthropic/claude-sonnet-4-6` | `anthropic/claude-sonnet-4-6` |
| 智谱 | `zhipu/glm-4-plus` | `zhipu/glm-4v-plus` |
| 通义 | `dashscope/qwen-max` | `dashscope/qwen-vl-max` |
| DeepSeek | `deepseek/deepseek-chat` | *(无视觉模型，混用上面任一)* |

#### B.4. 入库

把文档放入 `data/raw/`，然后：

```bash
python -m rag.pipeline.ingest data/raw/
```

首次运行会从 HuggingFace 下载 BGE-M3 (~2.3GB) 与 reranker (~2.4GB) 模型到 `~/.cache/huggingface/`。

可选参数：
- `--no-vlm` — 跳过图片描述（不消耗 VLM token，但图片不可被检索）
- `--batch-size 32` — embedding 批大小

#### B.5. 提问

**CLI：**

```bash
python -m rag.pipeline.query "你的问题"
```

**Web UI：**

```bash
streamlit run app/streamlit_app.py
```

浏览器打开 http://localhost:8501，左侧栏可以直接拖入新文件入库。

---

## 切换 Provider（三元组模式）

每个云端角色（**LLM** / **VLM** / **Embedding**）独立配置一组 `(MODEL, API_KEY, BASE_URL)`。这套设计能同时覆盖：

- 直连官方（OpenAI / Anthropic / 智谱 / 通义 / DeepSeek...）
- OpenAI-compatible 聚合服务（**AIHubMix / OpenRouter / OneAPI / 自部署 vLLM/FastChat**）
- 不同角色用不同 provider（如 LLM 走聚合、Embedding 走原生）

### 直连官方 vs 聚合服务

```diff
# 直连官方（旧风格也仍可用：留空三元组 → 读 OPENAI_API_KEY/ANTHROPIC_API_KEY）
  LLM_MODEL=openai/gpt-4o-mini
  OPENAI_API_KEY=sk-...

# 切到 AIHubMix（聚合）
- LLM_MODEL=openai/gpt-4o-mini
+ LLM_MODEL=openai/claude-sonnet-4.5         # openai/ 前缀 = OpenAI-style 协议
+ LLM_API_KEY=sk-aihub-xxx
+ LLM_BASE_URL=https://aihubmix.com/v1

# 切到 OpenRouter（聚合）
+ LLM_MODEL=openai/anthropic/claude-sonnet-4.5
+ LLM_API_KEY=sk-or-v1-xxx
+ LLM_BASE_URL=https://openrouter.ai/api/v1

# 切到原生 Anthropic（LiteLLM 直接识别）
+ LLM_MODEL=anthropic/claude-sonnet-4-6
+ ANTHROPIC_API_KEY=sk-ant-...               # 不填 LLM_API_KEY，留空 fallback
```

### 关键规则

1. **`openai/<模型名>` + `BASE_URL`** = 走 OpenAI-style 协议到任何 compatible 端点
2. **`anthropic/...` / `zhipu/...`** 等原生前缀：LiteLLM 自动选对应 SDK，**不用填 BASE_URL**
3. **三元组留空时**自动 fallback 到 LiteLLM 默认行为（读 `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` 等）
4. **三个角色独立**：LLM 走 AIHubMix、VLM 走 OpenRouter、Embedding 直连 OpenAI —— 没问题

### 常见聚合服务的模型 ID 写法

| 服务 | LLM_MODEL 写法 | LLM_BASE_URL |
|---|---|---|
| 直连 OpenAI | `openai/gpt-4o` | （留空） |
| 直连 Claude | `anthropic/claude-sonnet-4-6` | （留空） |
| AIHubMix | `openai/<官方模型名>`（如 `openai/claude-sonnet-4.5`） | `https://aihubmix.com/v1` |
| OpenRouter | `openai/<provider>/<model>`（如 `openai/anthropic/claude-sonnet-4.5`） | `https://openrouter.ai/api/v1` |
| 自部署 OneAPI | `openai/<你定义的名>` | `http://your-host:3000/v1` |
| 本地 Ollama | `openai/<ollama_model>` | `http://localhost:11434/v1` |

代码完全不动；改 `.env` 即可。

---

## 本地 VLM（Ollama）

如果不想吃 OpenAI VLM 的 API 费，可以把 VLM 切到本地 Ollama 跑。LLM 通常仍用云端（本地小模型推理质量不如 GPT/Claude），所以**只切 VLM**。

### 一次性安装

1. 装 [Ollama](https://ollama.com/download)（Windows / macOS / Linux 都有官方安装包）
2. 拉视觉模型：
   ```bash
   ollama pull qwen2.5vl:3b   # 3B (~3GB 显存) — 推荐 8GB 卡
   # 或 ollama pull qwen2.5vl:7b   # 7B (~6GB 显存) — 质量更高
   ```
3. 设系统环境变量让并发能用上：
   ```powershell
   [Environment]::SetEnvironmentVariable("OLLAMA_NUM_PARALLEL", "2", "User")
   ```
   设完重启 Ollama 进程（系统托盘退出再起）。

### `.env` 切换

只改 `VLM_*` 三行，`LLM_*` 不动：

```diff
- VLM_MODEL=openai/gpt-4o-mini
- VLM_API_KEY=
- VLM_BASE_URL=
+ VLM_MODEL=openai/qwen2.5vl:3b
+ VLM_API_KEY=ollama                       # 不校验，留空会被 LiteLLM 拒
+ VLM_BASE_URL=http://localhost:11434/v1
```

### 切回云端备份

把上面三行注释掉，恢复原来的 OpenAI 那三行即可。`.env` 模板里两套配置都有，互相注释切换 10 秒搞定。

### 行为差异

| 维度 | OpenAI VLM | Ollama 本地 VLM (3B) |
|---|---|---|
| 单图耗时 | ~1-2s | ~3-6s（GPU 推理 + cold start 第一次更慢）|
| 并发 | 8（默认） | 2（自动调低，受显存限制）|
| 费用 | ~$0.0005/图（low detail）| 0 |
| 离线 | ❌ | ✅ |
| 描述质量 | 高 | 中（中文识别尚可，复杂图表略弱）|

> ingest 命令的 `--vlm-workers` 参数会自动按 `VLM_BASE_URL` 决定默认值（localhost → 2，云端 → 8）。手动指定可以覆盖。

---

## 切换 Embedding：本地 vs 云端

|  | local（默认） | cloud |
|---|---|---|
| Embedding 来源 | 本地 BGE-M3 推理 | LiteLLM 调云端 API |
| Sparse 通道 | BGE-M3 直接输出 learned sparse | **Milvus 内置 BM25** 从原文自动算 |
| 首次部署 | 下载 ~2.3 GB 模型 | 0 模型，秒启 |
| 推理时机器内存 | ~3 GB | 几乎为 0 |
| 入库速度 | 50-100 chunk/s | 受 API rate limit |
| 隐私 | 内容不出本机 | 内容发到 embedding provider |
| 离线可用 | ✅ | ❌ |
| 入库成本（百万 token） | 0 | $0.02 - $0.13 |

**怎么切：** 改 `.env`：

```diff
- EMBEDDING_PROVIDER=local
+ EMBEDDING_PROVIDER=cloud
+ CLOUD_EMBEDDING_MODEL=openai/text-embedding-3-small
+ EMBEDDING_DIM=1536          # 必须与模型实际维度一致，否则 collection 建错
```

> ⚠️ **切换需要清空 collection**：dense 维度变了，schema 不兼容。先 `standalone.bat delete`（或 `docker compose down -v`），再 `standalone.bat start`、重新入库。

支持的 cloud 模型与维度：

| 模型 | 维度 | 备注 |
|---|---|---|
| `openai/text-embedding-3-small` | 1536 | 性价比好 |
| `openai/text-embedding-3-large` | 3072 | 质量最高 |
| `zhipu/embedding-3` | 2048 | 中文友好，国内访问稳定 |
| `dashscope/text-embedding-v3` | 1024 | 通义，中英都行 |
| `jina_ai/jina-embeddings-v3` | 1024 | 多语言强 |

---

## 测试

```bash
pytest tests/test_pipeline_smoke.py -v
```

冒烟测试不调用 LLM/VLM，只验证 load → chunk → embed → Milvus → retrieve → rerank 主链路。

---

## 项目结构

```
rag_test/
├── app/streamlit_app.py        # Web UI
├── rag/
│   ├── config.py               # 配置（pydantic-settings）
│   ├── schema.py               # Chunk 数据结构
│   ├── loaders/                # 4 个文档加载器
│   ├── parsing/                # VLM caption + chunker
│   ├── embedding/              # BGE-M3 双路向量
│   ├── vector_store/           # Milvus 混合检索
│   ├── retrieval/              # retriever + reranker
│   ├── llm/                    # LiteLLM 包装 + 流式
│   └── pipeline/               # ingest / query 入口
├── data/
│   ├── raw/                    # 你的原始文档
│   ├── images/                 # 提取/缓存的图片 + captions.jsonl
│   └── milvus/                 # （仅 Lite 模式用）
├── tests/test_pipeline_smoke.py
├── environment.yml
├── requirements.txt
├── .env.example
├── standalone.bat              # Milvus 控制脚本（首次自动下载）
├── plan.md                     # 项目规划
├── ARCHITECTURE.md             # 架构与技术细节
└── README.md
```

详细架构请见 [ARCHITECTURE.md](ARCHITECTURE.md)。

---

## 常见问题

**Q: 入库或查询提示 "milvus connection refused"？**
A: Docker Desktop 没启动，或 `standalone.bat start` 没跑成功。`docker ps` 看 milvus-standalone 是否 healthy。

**Q: `standalone.bat start` 报 `bind: An attempt was made to access a socket in a way forbidden by its access permissions`？**
A: 这是 Windows 上的 **Hyper-V 动态端口保留** 问题（不是端口被别的程序占用）。Hyper-V 会在系统启动/更新后随机预留一段端口给虚拟网卡，如果 9091 / 19530 / 2379 落进了预留范围，Docker 就绑不上。**修复**（管理员 PowerShell 执行）：
```powershell
net stop winnat
net start winnat
.\standalone.bat start
```
重启 winnat 后预留端口范围会重新分配，绝大多数情况一次搞定。如果反复出现，可永久排除这三个端口：
```powershell
# 管理员 PowerShell
netsh int ipv4 add excludedportrange protocol=tcp startport=9091 numberofports=1
netsh int ipv4 add excludedportrange protocol=tcp startport=19530 numberofports=1
netsh int ipv4 add excludedportrange protocol=tcp startport=2379 numberofports=1
```
排查命令：`netsh interface ipv4 show excludedportrange protocol=tcp` 看 9091 是否落在某个 Start-End 区间内。

**Q: HuggingFace 模型下载很慢/失败？**
A: 国内可设置 `HF_ENDPOINT=https://hf-mirror.com` 环境变量再启动。

**Q: 提示 `XLMRobertaTokenizer has no attribute prepare_for_model`？**
A: 这是 transformers 5.x 与 FlagEmbedding 的不兼容。`requirements.txt` 已锁 `<5.0`，重装即可。

**Q: Windows 上 conda 报 SSL 证书错误？**
A: conda 默认 `SSL_CERT_FILE` 路径错误。临时方案：
```powershell
$env:SSL_CERT_FILE="C:\Users\<你>\miniconda3\envs\rag_test\Library\ssl\cacert.pem"
```

**Q: 想换更小的 embedding 模型？**
A: 改 `.env` 的 `EMBEDDING_MODEL`（注意：换模型后向量维度可能变，需要 `standalone.bat delete` 清库重建）。

**Q: 关闭电脑后再用要怎么启？**
A: 启动 Docker Desktop → `.\standalone.bat start`（容器已存在会自动复用）→ `streamlit run app/streamlit_app.py`。

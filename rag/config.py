from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- LLM / VLM (LiteLLM 模型名) ----
    # 三元组 (model, api_key, base_url) 模式：
    #   - model 用 LiteLLM 标准格式，如 openai/gpt-4o, anthropic/claude-sonnet-4-6
    #   - 接 OpenAI-compatible 聚合（AIHubMix / OpenRouter / OneAPI / 自部署）时：
    #     model 写成 openai/<聚合服务侧模型ID>，base_url 填聚合服务地址
    #   - 不填 api_key/base_url 则走 LiteLLM 默认（读 OPENAI_API_KEY 等标准 env）
    llm_model: str = "openai/gpt-4o-mini"
    llm_api_key: str | None = None
    llm_base_url: str | None = None

    vlm_model: str = "openai/gpt-4o-mini"
    vlm_api_key: str | None = None
    vlm_base_url: str | None = None

    # 全局 VLM 限速（请求/秒）。0 或留空 = 不限速。
    # OpenAI Tier 1 gpt-4o-mini: 200K TPM + 500 RPM → 安全值 5；
    # AIHubMix / OpenRouter 通常更宽松；本地 Ollama 留空即可。
    vlm_max_rps: float = 0.0

    # ---- 标准 provider key（向后兼容；显式三元组留空时 LiteLLM 自动读这些） ----
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    zhipuai_api_key: str | None = None
    dashscope_api_key: str | None = None
    deepseek_api_key: str | None = None
    openrouter_api_key: str | None = None
    openai_base_url: str | None = None

    # ---- Embedding / Reranker ----
    # provider: local = 本地 BGE-M3 双路; cloud = LiteLLM 调云端 dense + Milvus BM25 sparse
    embedding_provider: Literal["local", "cloud"] = "local"
    embedding_model: str = "BAAI/bge-m3"  # local 模式用
    cloud_embedding_model: str = "openai/text-embedding-3-small"  # cloud 模式用，LiteLLM 名
    embedding_api_key: str | None = None       # cloud 三元组：留空则 fallback 到 OPENAI_API_KEY 等标准 env
    embedding_base_url: str | None = None      # cloud 三元组：聚合服务时填，直连官方留空
    embedding_dim: int = 1024  # local=1024(BGE-M3); cloud 视模型而定（见 .env.example 表）
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    embedding_device: Literal["cpu", "cuda"] = "cpu"
    use_fp16: bool = False
    # Milvus BM25 内置分词器：standard(通用) / chinese(jieba) / english
    milvus_analyzer: Literal["standard", "chinese", "english"] = "standard"

    # ---- Milvus ----
    # Windows 默认走 Docker Standalone (http://localhost:19530)
    # Linux/macOS 可以填本地路径 "./data/milvus/rag_test.db" 走 milvus-lite
    milvus_uri: str = "http://localhost:19530"
    milvus_collection: str = "rag_chunks"

    # ---- 切分 ----
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ---- 检索 ----
    retrieve_top_k: int = 30
    rerank_top_k: int = 5

    # ---- 路径 ----
    data_raw_dir: Path = Field(default=PROJECT_ROOT / "data" / "raw")
    data_images_dir: Path = Field(default=PROJECT_ROOT / "data" / "images")

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def milvus_is_remote(self) -> bool:
        return self.milvus_uri.startswith(("http://", "https://", "tcp://"))

    @property
    def milvus_path(self) -> Path | None:
        """嵌入式 Milvus Lite 的本地文件路径；Docker 模式返回 None。"""
        if self.milvus_is_remote:
            return None
        p = Path(self.milvus_uri)
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()

    def ensure_dirs(self) -> None:
        self.data_raw_dir.mkdir(parents=True, exist_ok=True)
        self.data_images_dir.mkdir(parents=True, exist_ok=True)
        if self.milvus_path is not None:
            self.milvus_path.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()


def _propagate_provider_env() -> None:
    """把 .env 中的标准 provider key 推到 os.environ。

    pydantic-settings 默认只把 .env 读进 Settings 对象，不会推到环境变量。
    LiteLLM 内部读 os.environ.get('OPENAI_API_KEY') 等，看不到 .env 内容会报
    AuthenticationError。这里桥接一下，让 .env 里写的 key 对 LiteLLM 也生效。

    已经存在的真实 env 变量不会被覆盖。
    """
    mapping = {
        "OPENAI_API_KEY": settings.openai_api_key,
        "ANTHROPIC_API_KEY": settings.anthropic_api_key,
        "ZHIPUAI_API_KEY": settings.zhipuai_api_key,
        "DASHSCOPE_API_KEY": settings.dashscope_api_key,
        "DEEPSEEK_API_KEY": settings.deepseek_api_key,
        "OPENROUTER_API_KEY": settings.openrouter_api_key,
        "OPENAI_BASE_URL": settings.openai_base_url,
    }
    for key, val in mapping.items():
        if val and not os.environ.get(key):
            os.environ[key] = val


_propagate_provider_env()

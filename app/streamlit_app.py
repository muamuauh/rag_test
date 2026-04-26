from __future__ import annotations

import shutil
import sys
from pathlib import Path

# 让 streamlit run app/streamlit_app.py 也能 import rag.*
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from rag.config import settings  # noqa: E402
from rag.llm.client import chat_stream  # noqa: E402
from rag.pipeline.ingest import ingest_paths  # noqa: E402
from rag.retrieval.retriever import Retriever  # noqa: E402
from rag.vector_store.milvus_store import MilvusStore  # noqa: E402


st.set_page_config(page_title="本地知识库 RAG", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="加载检索器（首次会下载 BGE-M3 + Reranker 模型）…")
def get_retriever() -> Retriever:
    return Retriever()


@st.cache_resource
def get_store() -> MilvusStore:
    return MilvusStore()


def render_contexts(contexts: list[dict]) -> None:
    for i, c in enumerate(contexts, 1):
        meta = c.get("meta", {})
        meta_inner = meta.get("metadata") or {}
        src = meta_inner.get("file_name") or meta.get("source_path", "?")
        page = meta.get("page")
        chunk_type = meta.get("chunk_type", "text")
        score = c.get("rerank_score", 0.0)
        head = f"**[{i}]** `{src}`"
        if page:
            head += f" · 第 {page} 页"
        head += f" · {chunk_type} · rerank={score:.3f}"
        st.markdown(head)
        if chunk_type == "image_caption" and meta.get("image_path"):
            img_path = Path(meta["image_path"])
            if img_path.exists():
                st.image(str(img_path), width=320)
        text = c.get("text", "")
        st.markdown(f"> {text[:600]}{'…' if len(text) > 600 else ''}")
        st.divider()


# ---------- Sidebar ----------

with st.sidebar:
    st.header("📚 知识库管理")

    store = get_store()
    try:
        cnt = store.count()
        st.metric("已入库 Chunk 数", cnt)
    except Exception as e:
        st.warning(f"无法读取统计：{e}")

    st.markdown(f"**Collection**: `{settings.milvus_collection}`")
    st.markdown(f"**LLM**: `{settings.llm_model}`")
    st.markdown(f"**VLM**: `{settings.vlm_model}`")
    st.markdown(f"**Embedding**: `{settings.embedding_model}`")

    st.divider()
    st.subheader("上传文档")
    use_vlm = st.checkbox("用 VLM 处理图片/图表（更慢但效果好）", value=True)
    uploads = st.file_uploader(
        "支持 PDF / DOCX / PPTX / MD / TXT / 图片",
        type=["pdf", "docx", "pptx", "md", "markdown", "txt", "png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
    )
    if uploads and st.button("入库", type="primary"):
        save_dir = settings.data_raw_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: list[Path] = []
        for f in uploads:
            dst = save_dir / f.name
            with dst.open("wb") as out:
                shutil.copyfileobj(f, out)
            saved_paths.append(dst)
        with st.status("处理中…", expanded=True) as status:
            st.write(f"已保存 {len(saved_paths)} 个文件到 `{save_dir}`")
            n = ingest_paths(saved_paths, use_vlm=use_vlm)
            status.update(label=f"入库完成：插入 {n} 条", state="complete")
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    if st.button("⚠️ 清空知识库", help="删除当前 Milvus collection"):
        try:
            store.drop()
            st.cache_resource.clear()
            st.success("已清空")
            st.rerun()
        except Exception as e:
            st.error(str(e))


# ---------- Chat area ----------

st.title("📚 本地个人知识库")
st.caption("中英文混合 · 多模态（VLM 图像描述）· BGE-M3 + Milvus 混合检索 · 多 Provider 可切换")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and m.get("contexts"):
            with st.expander(f"📎 引用 {len(m['contexts'])} 条参考资料"):
                render_contexts(m["contexts"])


query = st.chat_input("问点什么…（回答仅依据已入库的知识库内容）")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    retriever = get_retriever()
    contexts: list[dict] = []
    with st.chat_message("assistant"):
        with st.spinner("检索中…"):
            contexts = retriever.retrieve(query)
        if not contexts:
            answer = "知识库中未检索到相关内容。请先在左侧上传文档。"
            st.markdown(answer)
        else:
            placeholder = st.empty()
            full = ""
            try:
                for delta in chat_stream(query, contexts):
                    full += delta
                    placeholder.markdown(full + "▌")
                placeholder.markdown(full)
                answer = full
            except Exception as e:
                answer = f"LLM 调用失败：{e}\n\n请检查 `.env` 中对应 provider 的 API key。"
                st.error(answer)
            with st.expander(f"📎 引用 {len(contexts)} 条参考资料"):
                render_contexts(contexts)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "contexts": contexts}
        )

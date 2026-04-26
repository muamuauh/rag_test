from __future__ import annotations

import argparse

from rag.llm.client import chat_stream
from rag.retrieval.retriever import Retriever


def query_once(question: str, stream: bool = True) -> str:
    retriever = Retriever()
    contexts = retriever.retrieve(question)
    if not contexts:
        msg = "知识库中未检索到相关内容。"
        print(msg)
        return msg

    print(f"\n[检索到 {len(contexts)} 条相关片段]")
    for i, c in enumerate(contexts, 1):
        meta = c.get("meta", {})
        src = (meta.get("metadata") or {}).get("file_name") or meta.get("source_path", "?")
        page = meta.get("page")
        page_str = f" p.{page}" if page else ""
        print(f"  [{i}] {src}{page_str} (rerank={c.get('rerank_score', 0):.3f})")

    print("\n[回答]")
    full = []
    if stream:
        for delta in chat_stream(question, contexts):
            print(delta, end="", flush=True)
            full.append(delta)
        print()
    else:
        from rag.llm.client import chat
        text = chat(question, contexts)
        print(text)
        full.append(text)
    return "".join(full)


def main() -> None:
    parser = argparse.ArgumentParser(description="查询 pipeline：retrieve → rerank → LLM")
    parser.add_argument("question", help="用户问题")
    parser.add_argument("--no-stream", action="store_true")
    args = parser.parse_args()
    query_once(args.question, stream=not args.no_stream)


if __name__ == "__main__":
    main()

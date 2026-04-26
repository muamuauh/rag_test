from __future__ import annotations

from typing import Any, Iterator

import litellm

from rag.config import settings


SYSTEM_PROMPT = (
    "你是一个严谨的知识库助手。基于提供的【参考资料】回答用户问题：\n"
    "1. 答案必须仅依据参考资料；若资料不足，明确说明无法从资料中得出。\n"
    "2. 在论述每个事实/结论后，使用方括号编号（如 [1][2]）引用对应资料。\n"
    "3. 中英文均可，按用户问题语言回答。\n"
    "4. 回答简洁，避免编造。"
)


def _llm_extra_kwargs() -> dict[str, Any]:
    """把 LLM 三元组中显式设置的 api_key/api_base 透传给 LiteLLM。
    留空则 LiteLLM 走默认（读 OPENAI_API_KEY / OPENROUTER_API_KEY 等标准 env）。"""
    kw: dict[str, Any] = {}
    if settings.llm_api_key:
        kw["api_key"] = settings.llm_api_key
    if settings.llm_base_url:
        kw["api_base"] = settings.llm_base_url
    return kw


def build_user_prompt(query: str, contexts: list[dict]) -> str:
    parts = ["【参考资料】"]
    for i, c in enumerate(contexts, 1):
        meta = c.get("meta", {})
        src = (meta.get("metadata") or {}).get("file_name") or meta.get("source_path", "?")
        page = meta.get("page")
        head = f"[{i}] 来源: {src}" + (f" 第{page}页" if page else "")
        parts.append(head)
        parts.append(c["text"])
        parts.append("")
    parts.append("【问题】")
    parts.append(query)
    return "\n".join(parts)


def chat(query: str, contexts: list[dict]) -> str:
    user_msg = build_user_prompt(query, contexts)
    resp = litellm.completion(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        **_llm_extra_kwargs(),
    )
    return resp["choices"][0]["message"]["content"]


def chat_stream(query: str, contexts: list[dict]) -> Iterator[str]:
    """流式输出 token 增量。"""
    user_msg = build_user_prompt(query, contexts)
    stream = litellm.completion(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        stream=True,
        **_llm_extra_kwargs(),
    )
    for chunk in stream:
        try:
            delta = chunk["choices"][0]["delta"].get("content")
        except (KeyError, IndexError, AttributeError):
            delta = None
        if delta:
            yield delta

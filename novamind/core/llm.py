"""LLM 模块：Protocol + llama-cpp GGUF 生产实现 + Mock 兜底。作者：晨星

线程纪律：CPU 推理瓶颈在内存带宽，n_threads 锁 2-4（ADR-003）。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Protocol

Message = dict[str, str]


class LLMProvider(Protocol):
    def generate(self, messages: list[Message], max_tokens: int = 512) -> str: ...
    def status(self) -> tuple[str, Optional[str]]: ...


class MockLLM:
    """离线兜底：基于检索上下文的模板化回答，保证 CI 全链路可验证。"""

    name = "mock-template"

    def __init__(self) -> None:
        self._error: Optional[str] = None

    def generate(self, messages: list[Message], max_tokens: int = 512) -> str:
        # 约定：system 消息里带 [CONTEXT] 块时做引用式回答
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        if "[CONTEXT]" in system:
            ctx = system.split("[CONTEXT]", 1)[1].split("[/CONTEXT]")[0].strip()
            snippet = ctx[:300].replace("\n", " ")
            return f"基于知识库内容：{snippet}……（离线模板回答，问题：{user[:50]}）"
        return f"离线模式回答：已收到问题「{user[:80]}」。加载真实模型后此处为生成内容。"

    def status(self) -> tuple[str, Optional[str]]:
        return self.name, self._error


class LlamaCppLLM:
    """llama-cpp-python 加载本地 GGUF。n_threads 锁定（内存带宽瓶颈）。"""

    name = "llama-cpp-gguf"

    def __init__(self, model_path: Path, n_ctx: int = 2048, n_threads: int = 4) -> None:
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"LLM 模型缺失: {model_path}（运行 scripts/download_models.py 下载）"
            )
        from llama_cpp import Llama

        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False,
        )
        self._error: Optional[str] = None

    def generate(self, messages: list[Message], max_tokens: int = 512) -> str:
        out = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return (out["choices"][0]["message"]["content"] or "").strip()

    def status(self) -> tuple[str, Optional[str]]:
        return self.name, self._error


class FallbackLLM:
    def __init__(
        self,
        model_path: Path,
        offline: bool,
        n_ctx: int = 2048,
        n_threads: Optional[int] = None,
    ) -> None:
        self._primary: Optional[LlamaCppLLM] = None
        self._error: Optional[str] = None
        threads = n_threads or min(4, os.cpu_count() or 4)
        if not offline:
            try:
                self._primary = LlamaCppLLM(model_path, n_ctx=n_ctx, n_threads=threads)
            except Exception as exc:  # noqa: BLE001
                self._error = f"{type(exc).__name__}: {exc}"
        else:
            self._error = "offline mode"
        self._fallback = MockLLM()

    @property
    def active(self) -> LLMProvider:
        return self._primary if self._primary is not None else self._fallback

    def generate(self, messages: list[Message], max_tokens: int = 512) -> str:
        return self.active.generate(messages, max_tokens)

    def status(self) -> tuple[str, Optional[str]]:
        name = self._primary.name if self._primary is not None else self._fallback.name
        return name, self._error

"""集中配置：全部经环境变量覆盖，默认值保证离线可运行。作者：晨星"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


@dataclass
class Settings:
    """系统配置。offline=True 时所有后端强制使用兜底实现。"""

    offline: bool = field(
        default_factory=lambda: os.environ.get("NOVAMIND_OFFLINE", "0") == "1"
    )
    # 嵌入
    embed_model_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("NOVAMIND_EMBED_DIR", str(MODELS_DIR / "bge-small-zh-v1.5"))
        )
    )
    embed_dim_offline: int = 256
    # 重排
    rerank_model_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("NOVAMIND_RERANK_DIR", str(MODELS_DIR / "bge-reranker-base"))
        )
    )
    rerank_weight: float = field(
        default_factory=lambda: float(os.environ.get("NOVAMIND_RERANK_WEIGHT", "0.3"))
    )
    # LLM
    llm_model_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "NOVAMIND_LLM_PATH",
                str(MODELS_DIR / "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
            )
        )
    )
    llm_threads: int = field(
        default_factory=lambda: _env_int(
            "NOVAMIND_LLM_THREADS", min(4, os.cpu_count() or 4)
        )
    )
    llm_ctx: int = field(default_factory=lambda: _env_int("NOVAMIND_LLM_CTX", 2048))
    llm_max_tokens: int = field(
        default_factory=lambda: _env_int("NOVAMIND_LLM_MAX_TOKENS", 512)
    )
    # 检索
    dense_top_n: int = 20
    bm25_top_n: int = 20
    rrf_k: int = 60
    default_top_k: int = 5
    # 分块
    chunk_size: int = 400
    chunk_overlap: int = 60
    # 索引
    index_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("NOVAMIND_INDEX_DIR", str(DATA_DIR / "index"))
        )
    )
    # 服务
    host: str = field(default_factory=lambda: os.environ.get("NOVAMIND_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("NOVAMIND_PORT", 8000))


def get_settings() -> Settings:
    return Settings()

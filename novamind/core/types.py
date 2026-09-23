"""共享数据类型。作者：晨星"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredChunk:
    """全链路打分留痕：dense/bm25 粗排分、重排分、最终分。"""

    chunk: Chunk
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: float = 0.0
    rerank_score: Optional[float] = None
    final_score: float = 0.0


@dataclass
class BackendStatus:
    """后端真实状态。error 非 None 表示已静默回退兜底实现——健康检查必须暴露。"""

    name: str
    error: Optional[str] = None

    @property
    def healthy(self) -> bool:
        return self.error is None

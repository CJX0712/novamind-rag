"""重排模块：ONNX cross-encoder 分数**加权融合**，绝不直接覆盖粗排顺序。

背景（实测教训）：把重排顺序直接当最终排序时，一个在查询语言上力不从心的
重排器会无人制衡地压掉正确答案（中文查询 top-1 从 11/12 掉到更低）。
因此 final = (1-w) * rrf_norm + w * rerank_norm，w 默认 0.3。作者：晨星
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol

import numpy as np

from .types import ScoredChunk


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...
    def status(self) -> tuple[str, Optional[str]]: ...


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-9:
        return [0.5] * len(values)
    return ((arr - lo) / (hi - lo)).tolist()


class OnnxReranker:
    """bge-reranker 类 ONNX cross-encoder。输入名运行时探测。"""

    name = "onnx-reranker"

    def __init__(self, model_dir: Path, max_length: int = 512) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_path = model_dir / "model.onnx"
        tok_path = model_dir / "tokenizer.json"
        if not model_path.exists() or not tok_path.exists():
            raise FileNotFoundError(
                f"重排模型缺失: {model_dir}（运行 scripts/download_models.py 下载）"
            )
        self._tokenizer = Tokenizer.from_file(str(tok_path))
        self._tokenizer.enable_truncation(max_length=max_length)
        self._tokenizer.enable_padding()
        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self._session.get_inputs()}
        self._error: Optional[str] = None

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        pairs = [(query, t) for t in texts]
        encodings = self._tokenizer.encode_batch(pairs)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        token_type_ids = np.array([e.type_ids for e in encodings], dtype=np.int64)
        feeds = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        feeds = {k: v for k, v in feeds.items() if k in self._input_names}
        logits = self._session.run(None, feeds)[0].reshape(-1)
        return _sigmoid(logits).astype(float).tolist()

    def status(self) -> tuple[str, Optional[str]]:
        return self.name, self._error


class NoopReranker:
    """离线兜底：不打分，全部返回 None，由融合层退化为纯粗排。"""

    name = "noop"

    def __init__(self) -> None:
        self._error: Optional[str] = None

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [0.5] * len(texts)

    def status(self) -> tuple[str, Optional[str]]:
        return self.name, self._error


class FallbackReranker:
    def __init__(self, model_dir: Path, offline: bool) -> None:
        self._primary: Optional[OnnxReranker] = None
        self._error: Optional[str] = None
        if not offline:
            try:
                self._primary = OnnxReranker(model_dir)
            except Exception as exc:  # noqa: BLE001
                self._error = f"{type(exc).__name__}: {exc}"
        else:
            self._error = "offline mode"
        self._fallback = NoopReranker()

    @property
    def active(self) -> Reranker:
        return self._primary if self._primary is not None else self._fallback

    def score(self, query: str, texts: list[str]) -> list[float]:
        return self.active.score(query, texts)

    def status(self) -> tuple[str, Optional[str]]:
        name = self._primary.name if self._primary is not None else self._fallback.name
        return name, self._error


def fuse_scores(candidates: list[ScoredChunk], weight: float) -> list[ScoredChunk]:
    """加权融合：final = (1-w)*rrf_norm + w*rerank_norm。就地更新 final_score 并排序。"""
    if not candidates:
        return []
    rrf_norm = _minmax([c.rrf_score for c in candidates])
    has_rerank = all(c.rerank_score is not None for c in candidates)
    rerank_norm = _minmax([c.rerank_score or 0.0 for c in candidates])
    for i, c in enumerate(candidates):
        if has_rerank:
            c.final_score = (1 - weight) * rrf_norm[i] + weight * rerank_norm[i]
        else:
            c.final_score = rrf_norm[i]
    candidates.sort(key=lambda c: c.final_score, reverse=True)
    return candidates

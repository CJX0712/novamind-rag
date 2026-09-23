"""嵌入模块：Protocol + ONNX 生产实现 + 哈希 bigram 离线兜底。作者：晨星

关键纪律：
- ONNX 输入名运行时探测（不同导出带/不带 token_type_ids），不硬编码。
- 生产实现初始化失败必须记录 error 并回退，禁止静默。
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

_CJK_RE = re.compile(r"[一-鿿]")
_TOKEN_RE = re.compile(r"[a-z0-9]+|[一-鿿]")


def tokenize(text: str) -> list[str]:
    """中英文混合切分：英文按词、中文按单字+双字 bigram（jieba 无 win 轮子的替代）。"""
    text = text.lower()
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text):
        tok = match.group(0)
        if _CJK_RE.fullmatch(tok):
            tokens.append(tok)
        else:
            tokens.append(tok)
    # 中文 bigram 提升短语匹配
    cjk_runs = re.findall(r"[一-鿿]{2,}", text)
    for run in cjk_runs:
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """返回 (n, dim) L2 归一化向量。"""
        ...

    def status(self) -> tuple[str, Optional[str]]:
        """(实现名, error)。error 非 None = 已回退。"""
        ...


class HashEmbedder:
    """离线兜底：哈希 bigram 词袋向量。零依赖，CI 可用。"""

    name = "hash-bigram"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self._error: Optional[str] = None

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in tokenize(text):
                h = int(hashlib.blake2b(tok.encode("utf-8"), digest_size=8).hexdigest(), 16)
                vecs[row, h % self.dim] += 1.0
            norm = np.linalg.norm(vecs[row])
            if norm > 0:
                vecs[row] /= norm
        return vecs

    def status(self) -> tuple[str, Optional[str]]:
        return self.name, self._error


class OnnxEmbedder:
    """生产实现：onnxruntime + tokenizers 加载 BGE 类 ONNX 嵌入模型。

    目录须含 model.onnx 与 tokenizer.json。模型经 scripts/download_models.py
    从 ModelScope 获取（HuggingFace 不可达）。
    """

    name = "onnx-bge"

    def __init__(self, model_dir: Path, max_length: int = 512) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_path = model_dir / "model.onnx"
        tok_path = model_dir / "tokenizer.json"
        if not model_path.exists() or not tok_path.exists():
            raise FileNotFoundError(
                f"嵌入模型缺失: {model_dir}（运行 scripts/download_models.py 下载）"
            )
        self._tokenizer = Tokenizer.from_file(str(tok_path))
        self._tokenizer.enable_truncation(max_length=max_length)
        self._tokenizer.enable_padding()
        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self._session.get_inputs()}
        output = self._session.get_outputs()[0]
        # 维度从输出 shape 推断（动态维度时回退到一次试跑）
        dim = output.shape[-1] if isinstance(output.shape[-1], int) else None
        self.dim = dim if dim else self._probe_dim()
        self._error: Optional[str] = None

    def _probe_dim(self) -> int:
        return int(self.embed(["维度探测"]).shape[-1])

    def embed(self, texts: list[str]) -> np.ndarray:
        encodings = self._tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        token_type_ids = np.array([e.type_ids for e in encodings], dtype=np.int64)
        feeds = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        feeds = {k: v for k, v in feeds.items() if k in self._input_names}
        hidden = self._session.run(None, feeds)[0]  # (n, seq, dim)
        # CLS pooling（BGE 官方做法）+ L2 归一化
        vecs = hidden[:, 0, :].astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    def status(self) -> tuple[str, Optional[str]]:
        return self.name, self._error


class FallbackEmbedder:
    """优先生产实现，失败时回退兜底并记录 error（健康检查可见）。"""

    def __init__(self, model_dir: Path, offline: bool, offline_dim: int = 256) -> None:
        self._primary: Optional[OnnxEmbedder] = None
        self._error: Optional[str] = None
        if not offline:
            try:
                self._primary = OnnxEmbedder(model_dir)
            except Exception as exc:  # noqa: BLE001 - 任何初始化失败都回退
                self._error = f"{type(exc).__name__}: {exc}"
        else:
            self._error = "offline mode"
        self._fallback = HashEmbedder(dim=offline_dim)

    @property
    def active(self) -> Embedder:
        return self._primary if self._primary is not None else self._fallback

    @property
    def dim(self) -> int:
        return self.active.dim

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.active.embed(texts)

    def status(self) -> tuple[str, Optional[str]]:
        name = self._primary.name if self._primary is not None else self._fallback.name
        return name, self._error

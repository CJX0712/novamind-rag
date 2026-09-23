"""向量库模块：Protocol + FAISS 生产实现 + 内存余弦兜底。作者：晨星"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

from .types import Chunk


class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> int: ...
    def search(self, vector: np.ndarray, top_n: int) -> list[tuple[Chunk, float]]: ...
    def count(self) -> int: ...
    def clear(self) -> int: ...
    def status(self) -> tuple[str, Optional[str]]: ...


class MemoryStore:
    """内存余弦相似度。离线兜底 + 评测独立索引专用。"""

    name = "memory-cosine"

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._chunks: list[Chunk] = []
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._error: Optional[str] = None

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> int:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape != (len(chunks), self.dim):
            raise ValueError(f"向量形状 {vectors.shape} 与 (chunks={len(chunks)}, dim={self.dim}) 不符")
        self._chunks.extend(chunks)
        self._vectors = np.vstack([self._vectors, vectors])
        return len(chunks)

    def search(self, vector: np.ndarray, top_n: int) -> list[tuple[Chunk, float]]:
        if len(self._chunks) == 0:
            return []
        v = np.asarray(vector, dtype=np.float32).ravel()
        scores = self._vectors @ v  # 双方均已归一化 → 余弦
        idx = np.argsort(-scores)[:top_n]
        return [(self._chunks[i], float(scores[i])) for i in idx]

    def count(self) -> int:
        return len(self._chunks)

    def clear(self) -> int:
        n = len(self._chunks)
        self._chunks = []
        self._vectors = np.zeros((0, self.dim), dtype=np.float32)
        return n

    def status(self) -> tuple[str, Optional[str]]:
        return self.name, self._error


class FaissStore:
    """FAISS IndexFlatIP（向量已 L2 归一化 → 内积=余弦），可持久化。"""

    name = "faiss-flatip"

    def __init__(self, dim: int, index_dir: Path) -> None:
        import faiss

        self.dim = dim
        self._dir = Path(index_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._dir / "vectors.faiss"
        self._meta_path = self._dir / "chunks.jsonl"
        self._chunks: list[Chunk] = []
        if self._index_path.exists() and self._meta_path.exists():
            self._index = faiss.read_index(str(self._index_path))
            self._load_meta()
        else:
            self._index = faiss.IndexFlatIP(dim)
        self._error: Optional[str] = None

    def _load_meta(self) -> None:
        with open(self._meta_path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    self._chunks.append(
                        Chunk(
                            chunk_id=row["chunk_id"],
                            doc_id=row["doc_id"],
                            text=row["text"],
                            meta=row.get("meta", {}),
                        )
                    )

    def _persist(self) -> None:
        import faiss

        faiss.write_index(self._index, str(self._index_path))
        with open(self._meta_path, "w", encoding="utf-8") as fh:
            for c in self._chunks:
                fh.write(
                    json.dumps(
                        {
                            "chunk_id": c.chunk_id,
                            "doc_id": c.doc_id,
                            "text": c.text,
                            "meta": c.meta,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> int:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape != (len(chunks), self.dim):
            raise ValueError(f"向量形状 {vectors.shape} 与 (chunks={len(chunks)}, dim={self.dim}) 不符")
        self._index.add(vectors)
        self._chunks.extend(chunks)
        self._persist()
        return len(chunks)

    def search(self, vector: np.ndarray, top_n: int) -> list[tuple[Chunk, float]]:
        if self._index.ntotal == 0:
            return []
        v = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        scores, idx = self._index.search(v, min(top_n, self._index.ntotal))
        return [
            (self._chunks[i], float(s))
            for s, i in zip(scores[0], idx[0])
            if 0 <= i < len(self._chunks)
        ]

    def count(self) -> int:
        return int(self._index.ntotal)

    def clear(self) -> int:
        n = self.count()
        # 沙箱对批量删文件有守卫 → 用 FAISS reset + 逐点清空元数据，不删目录
        self._index.reset()
        self._chunks = []
        self._persist()
        return n

    def status(self) -> tuple[str, Optional[str]]:
        return self.name, self._error


class FallbackStore:
    """优先 FAISS，失败回退内存并记录 error。"""

    def __init__(self, dim: int, index_dir: Path, offline: bool) -> None:
        self._primary: Optional[FaissStore] = None
        self._error: Optional[str] = None
        if not offline:
            try:
                self._primary = FaissStore(dim, index_dir)
            except Exception as exc:  # noqa: BLE001
                self._error = f"{type(exc).__name__}: {exc}"
        else:
            self._error = "offline mode"
        self._fallback = MemoryStore(dim)

    @property
    def active(self) -> VectorStore:
        return self._primary if self._primary is not None else self._fallback

    def upsert(self, chunks, vectors) -> int:
        return self.active.upsert(chunks, vectors)

    def search(self, vector, top_n):
        return self.active.search(vector, top_n)

    def count(self) -> int:
        return self.active.count()

    def clear(self) -> int:
        return self.active.clear()

    def status(self) -> tuple[str, Optional[str]]:
        name = self._primary.name if self._primary is not None else self._fallback.name
        return name, self._error


def new_chunk_id() -> str:
    return uuid.uuid4().hex[:16]

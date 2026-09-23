"""文档摄取：解析（txt/md/pdf）→ 分块 → Chunk 列表。作者：晨星"""
from __future__ import annotations

import re
import uuid
from pathlib import Path

from .types import Chunk


def parse_text(source: str, text: str) -> str:
    return text.strip()


def parse_file(path: Path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    if suffix in {".txt", ".md", ".markdown"}:
        return path.read_text(encoding="utf-8", errors="ignore").strip()
    raise ValueError(f"不支持的文件类型: {suffix}（支持 .txt/.md/.pdf）")


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 60) -> list[str]:
    """按段落优先、长度兜底的滑窗分块。中英文通用（按字符数）。"""
    if not text:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        if len(buf) + len(para) + 1 <= chunk_size:
            buf = f"{buf}\n{para}".strip()
            continue
        if buf:
            chunks.append(buf)
        # 超长段落滑窗硬切
        while len(para) > chunk_size:
            chunks.append(para[:chunk_size])
            para = para[chunk_size - overlap :]
        buf = para
    if buf:
        chunks.append(buf)
    return chunks


def build_chunks(
    doc_id: str, text: str, chunk_size: int = 400, overlap: int = 60
) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=uuid.uuid4().hex[:16],
            doc_id=doc_id,
            text=piece,
            meta={"seq": i, "len": len(piece)},
        )
        for i, piece in enumerate(chunk_text(text, chunk_size, overlap))
    ]

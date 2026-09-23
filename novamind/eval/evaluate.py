"""黄金集评测：永远在全新独立管道上跑，不读不写生产索引。作者：晨星

纪律（实测踩坑）：复用生产索引 + 空库才灌语料 → 跑过几次后黄金语料
一条没入库 → 指标全错但零异常。因此本模块每次调用都新建管道。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..core.pipeline import RAGPipeline

GOLDEN_PATH = Path(__file__).resolve().parent / "golden.json"


@dataclass
class EvalReport:
    recall_at_k: float
    mrr: float
    per_query: list[dict]

    def to_dict(self) -> dict:
        return {
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "per_query": self.per_query,
        }


def load_golden(path: Optional[Path] = None) -> list[dict]:
    path = path or GOLDEN_PATH
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def evaluate(pipeline_factory, k: int = 5, golden_path: Optional[Path] = None) -> EvalReport:
    """pipeline_factory: 无参可调用，返回全新管道（含独立索引）。"""
    golden = load_golden(golden_path)
    pipeline: RAGPipeline = pipeline_factory()
    # 在独立索引上灌入黄金语料
    doc_ids = set()
    for item in golden:
        if item["doc_id"] not in doc_ids:
            pipeline.ingest_text(item["document"], doc_id=item["doc_id"])
            doc_ids.add(item["doc_id"])

    per_query: list[dict] = []
    hits = 0
    rr_sum = 0.0
    for item in golden:
        results = pipeline.retrieve(item["query"], top_k=k)
        rank = None
        for i, hit in enumerate(results):
            if hit.chunk.doc_id == item["doc_id"]:
                rank = i + 1
                break
        hit = rank is not None
        hits += int(hit)
        rr_sum += 1.0 / rank if rank else 0.0
        per_query.append({"query": item["query"], "hit": hit, "rank": rank})

    n = max(len(golden), 1)
    return EvalReport(
        recall_at_k=round(hits / n, 4),
        mrr=round(rr_sum / n, 4),
        per_query=per_query,
    )

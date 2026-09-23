"""从 ModelScope 下载模型（HuggingFace 不可达时的国内通道）。作者：晨星

用法：
    python scripts/download_models.py            # 下载全部（嵌入+重排+LLM）
    python scripts/download_models.py embed      # 仅嵌入模型
    python scripts/download_models.py rerank
    python scripts/download_models.py llm

直链格式（无需 SDK）：
    https://www.modelscope.cn/api/v1/models/{repo}/repo?Revision=master&FilePath={path}
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

MS_BASE = "https://www.modelscope.cn/api/v1/models/{repo}/repo?Revision=master&FilePath={path}"

# (本地目录, ModelScope repo, [(远端路径, 本地文件名)])
# 选用 Xenova 导出的量化 ONNX（int8），体积/精度平衡；LLM 用 Q4_K_M。
TARGETS = {
    "embed": (
        MODELS / "bge-small-zh-v1.5",
        "Xenova/bge-small-zh-v1.5",
        [
            ("onnx/model_quantized.onnx", "model.onnx"),
            ("tokenizer.json", "tokenizer.json"),
            ("config.json", "config.json"),
        ],
    ),
    "rerank": (
        MODELS / "bge-reranker-base",
        "Xenova/bge-reranker-base",
        [
            ("onnx/model_quantized.onnx", "model.onnx"),
            ("tokenizer.json", "tokenizer.json"),
        ],
    ),
    "llm": (
        MODELS,
        "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        [
            ("qwen2.5-0.5b-instruct-q4_k_m.gguf", "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
        ],
    ),
}


def download(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "novamind/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as fh:
            total = int(resp.headers.get("Content-Length") or 0)
            got = 0
            while True:
                block = resp.read(1 << 20)
                if not block:
                    break
                fh.write(block)
                got += len(block)
                if total:
                    pct = got * 100 // total
                    print(f"\r  {dest.name}: {got >> 20}MB / {total >> 20}MB ({pct}%)", end="", flush=True)
            print()
        tmp.replace(dest)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"\n  下载失败 {url}: {type(exc).__name__}: {exc}")
        tmp.unlink(missing_ok=True)
        return False


def pull(kind: str) -> bool:
    dest_dir, repo, files = TARGETS[kind]
    ok = True
    for remote, local in files:
        dest = dest_dir / local
        if dest.exists() and dest.stat().st_size > 0:
            print(f"跳过（已存在）: {dest}")
            continue
        url = MS_BASE.format(repo=repo, path=remote)
        print(f"下载 {repo}/{remote} -> {dest}")
        if not download(url, dest):
            ok = False
    return ok


def main() -> int:
    kinds = sys.argv[1:] or list(TARGETS)
    results = {}
    for kind in kinds:
        if kind not in TARGETS:
            print(f"未知目标: {kind}（可选: {', '.join(TARGETS)}）")
            return 2
        results[kind] = pull(kind)
    print("\n== 下载结果 ==")
    for kind, ok in results.items():
        print(f"  {kind}: {'OK' if ok else 'FAILED'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

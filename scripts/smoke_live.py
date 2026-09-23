"""真实进程冒烟：拉起 uvicorn，跑核心成功流 + 错误流，杀进程。作者：晨星"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
import urllib.error

PORT = 8321
BASE = f"http://127.0.0.1:{PORT}"


def req(method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(r, timeout=120) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


def main() -> int:
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "novamind.api.server:app",
         "--host", "127.0.0.1", "--port", str(PORT)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    checks = []
    try:
        # 等待就绪（在线模式加载 GGUF 较慢）
        ready = False
        for _ in range(120):
            try:
                code, _ = req("GET", "/health")
                if code == 200:
                    ready = True
                    break
            except Exception:
                time.sleep(1)
        checks.append(("服务就绪", ready))

        code, health = req("GET", "/health")
        checks.append(("health 200", code == 200))
        print("backends:", json.dumps(health.get("backends", {}), ensure_ascii=False))

        code, ing = req("POST", "/ingest", {
            "text": "NovaMind 是一个本地知识库问答系统，采用混合检索与融合重排。" * 3,
            "doc_id": "smoke-doc",
        })
        checks.append(("ingest 200 且 chunks>0", code == 200 and ing.get("chunks", 0) > 0))

        code, q = req("POST", "/rag/query", {"query": "NovaMind 采用了什么检索技术", "top_k": 3})
        checks.append(("rag/query 有答案有来源", code == 200 and bool(q.get("answer")) and len(q.get("sources", [])) >= 1))
        print("answer:", (q.get("answer") or "")[:150])

        code, chat = req("POST", "/chat", {"message": "算一下 6*7+8"})
        checks.append(("chat calculator", code == 200 and "50" in chat.get("reply", "")))
        print("chat reply:", chat.get("reply", "")[:100])

        code, ev = req("POST", "/eval", {"k": 5})
        checks.append(("eval recall>=0.75", code == 200 and ev.get("recall_at_k", 0) >= 0.75))
        print("eval:", ev.get("recall_at_k"), ev.get("mrr"))

        code, _ = req("POST", "/ingest", {})
        checks.append(("错误流: 空摄取返回 400", code == 400))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        subprocess.run(["taskkill", "/pid", str(proc.pid), "/t", "/f"],
                       capture_output=True)

    passed = sum(1 for _, ok in checks if ok)
    for name, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"=== 冒烟汇总: passed={passed} failed={len(checks) - passed} ===")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())

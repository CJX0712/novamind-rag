"""P0 门禁：扫描仓库代码中作为功能图标的 emoji。作者：晨星

用法：python tools/scan_emoji.py  → 发现即退出码 1
豁免：.git、models、data、本脚本自身、markdown 文档中的状态图标不在 UI 代码范围
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EMOJI_RE = re.compile(
    "[\U0001F300-\U0001F9FF☀-➿️\U0001F000-\U0001F02F"
    "\U0001F0A0-\U0001F0FF\U0001F100-\U0001F64F\U0001F680-\U0001F6FF"
    "\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF‍⃣\U000E0020-\U000E007F]"
)

SCAN_EXTS = {".py", ".html", ".js", ".css", ".ts", ".tsx", ".jsx", ".vue", ".yaml", ".yml"}
SKIP_DIRS = {".git", "models", "data", "node_modules", "__pycache__", ".venv"}
SKIP_FILES = {"scan_emoji.py"}


def scan() -> list[tuple[Path, int, str]]:
    findings = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SCAN_EXTS:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name in SKIP_FILES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if EMOJI_RE.search(line):
                findings.append((path.relative_to(ROOT), lineno, line.strip()[:100]))
    return findings


def main() -> int:
    findings = scan()
    if findings:
        print(f"P0 违规：发现 {len(findings)} 处 emoji（功能图标必须改用 SVG 图标库）")
        for path, lineno, line in findings:
            print(f"  {path}:{lineno}: {line}")
        return 1
    print("P0 门禁通过：代码中无 emoji")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""交付前统一剥离 UTF-8 BOM（PowerShell Out-File 会加 BOM，破坏 JSON 解析等）。作者：晨星"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".git", "models", "data", "node_modules", "__pycache__", ".venv"}


def main() -> int:
    stripped = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(p in SKIP_DIRS for p in path.parts):
            continue
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        if raw.startswith(b"\xef\xbb\xbf"):
            try:
                path.write_bytes(raw[3:])
            except OSError as exc:
                print(f"skip (locked): {path.relative_to(ROOT)}: {exc}")
                continue
            stripped.append(path.relative_to(ROOT))
    for p in stripped:
        print(f"stripped BOM: {p}")
    print(f"done, {len(stripped)} file(s) stripped")
    return 0


if __name__ == "__main__":
    sys.exit(main())

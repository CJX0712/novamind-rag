"""pytest 配置：保证包可导入。作者：晨星"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

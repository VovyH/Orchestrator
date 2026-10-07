"""让 tests/ 下的脚本在两种目录布局里都能 import orchestrator。

布局一，Orchestrator 是根项目：

    <root>/
    ├── orchestrator/__init__.py
    └── tests/test_probe.py

布局二，Orchestrator 是别人的子项目：

    <repo>/
    └── Orchestrator/
        ├── orchestrator/__init__.py
        └── tests/test_probe.py

直接跑脚本时 sys.path[0] 只有 tests/ 这一层，仓库根目录不在里面，
所以不 pip install -e . 也导入不了。这里从 tests/ 逐级往上找第一个
带 orchestrator/__init__.py 的目录，插到 sys.path 最前面。
插在最前面还有一个用处：跑的一定是当前这份源码，不会撞上装过的另一份。
"""

from __future__ import annotations

import sys
from pathlib import Path

_PACKAGE = "orchestrator"


def find_root(start: Path) -> Path | None:
    """从 start 往上找第一个含 orchestrator 包的目录。找不到返回 None。"""
    for directory in (start, *start.parents):
        if (directory / _PACKAGE / "__init__.py").is_file():
            return directory
    return None


def ensure_importable() -> Path:
    """把仓库根目录放到 sys.path[0]，返回这个根目录。"""
    root = find_root(Path(__file__).resolve().parent)
    if root is None:
        raise RuntimeError(
            "找不到 orchestrator 包：tests/ 的各级上级目录里都没有 "
            "orchestrator/__init__.py。确认 tests/ 和 orchestrator/ 在同一个仓库里。"
        )
    path = str(root)
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)
    return root

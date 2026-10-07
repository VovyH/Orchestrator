"""万能入口：无论当前在哪个目录、无论 Orchestrator 是不是仓库根，都能跑。

    python /mnt/WYH/projects/project-7/Orchestrator/run.py --model ... --data ... --out ...

不带参数时打印用法。这个脚本只做一件事：把 Orchestrator 的根目录放进 sys.path，
然后调用 orchestrator.api 的命令行入口。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from orchestrator.api import main  # noqa: E402

if __name__ == "__main__":
    main()

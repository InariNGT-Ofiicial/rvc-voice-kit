# -*- coding: utf-8 -*-
"""P2 训练启动器：把自身 stdout/stderr 接到文件，再以 __main__ 方式跑 p1_run.py。

为什么要这层：
  WMI 拉起的进程没有控制台，sys.stdout 可能是 None，直接 print 会 AttributeError。
  这里先 dup2 + 换 sys.stdout，保证 p1_run.py 的所有输出都落到 logs/p2_run.log。
用法（由 PowerShell 的 WMI 调用，参数写死在下面 ARGS 里）：
  python p2_launch.py
"""
import sys as _sys
from pathlib import Path as _pth

_sys.path.insert(0, str(_pth(__file__).resolve().parents[2]))
from rvcx import paths as _paths  # noqa: E402

import os
import runpy
import sys
from pathlib import Path

ROOT = _paths.ROOT
LOG = ROOT / "logs" / "p2_run.log"
LOG.parent.mkdir(parents=True, exist_ok=True)

out = open(LOG, "a", encoding="utf-8", buffering=1)
os.dup2(out.fileno(), 1)
os.dup2(out.fileno(), 2)
sys.stdout = out
sys.stderr = out

(ROOT / "logs" / "p2.pid").write_text(str(os.getpid()), encoding="utf-8")
print(f"\n\n===== P2 launcher start {os.getpid()} =====", flush=True)

ARGS = ["p1_run.py", "--exp", "fangno_p2", "--steps", "train",
        "--epochs", "200", "--bs", "4", "--se", "20",
        "--sr", "48k", "--version", "v2"]
sys.argv = ARGS
try:
    runpy.run_path(str(ROOT / "p1_run.py"), run_name="__main__")
except SystemExit as e:
    print(f"===== P2 launcher exit: {e} =====", flush=True)
    raise
finally:
    print("===== P2 launcher done =====", flush=True)

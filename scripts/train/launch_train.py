# -*- coding: utf-8 -*-
"""脱离进程训练启动器：把 stdout/stderr 接到日志文件，再以 __main__ 方式跑 p1_run.py。

为什么要这层（WMI 拉起的进程没有控制台）：
  sys.stdout 可能是 None，p1_run.py 里的 print 会 AttributeError。
  这里先 dup2 + 替换 sys.stdout，保证所有输出都落到 logs/<exp>_launcher.log。

用法：
  # 方式一：直接跑（本进程内）
  python launch_train.py --exp dsxiaoyi_a3_v1 --data E:/AI-RVC/dataset/dsxiaoyi_a3 \
                         --epochs 150 --steps train,index,infer

  # 方式二：用 WMI 拉起（会话结束不被杀，推荐长训练）
  powershell -NoProfile -Command "(Get-WmiObject Win32_Process).Create('...python.exe launch_train.py ...')"
"""
import argparse
import os
import runpy
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parents[2]          # rvc-voice-kit 根
sys.path.insert(0, str(KIT))
from rvcx import paths as _paths                   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="实验名（logs/<exp>）")
    ap.add_argument("--data", default=None, help="训练集目录")
    ap.add_argument("--steps", default="train,index,infer")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--se", type=int, default=20)
    ap.add_argument("--sr", default="48k")
    ap.add_argument("--version", default="v2")
    ap.add_argument("--f0method", default="rmvpe")
    ap.add_argument("--index-rate", default="0.75")
    a = ap.parse_args()

    log_dir = _paths.WORK / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"{a.exp}_launcher.log"
    out = open(log, "a", encoding="utf-8", buffering=1)
    os.dup2(out.fileno(), 1)
    os.dup2(out.fileno(), 2)
    sys.stdout = out
    sys.stderr = out

    (log_dir / f"{a.exp}.pid").write_text(str(os.getpid()), encoding="utf-8")
    print(f"\n\n===== launcher start {os.getpid()}  exp={a.exp}  {log} =====", flush=True)

    argv = [str(KIT / "scripts" / "train" / "p1_run.py"),
            "--exp", a.exp, "--steps", a.steps,
            "--epochs", str(a.epochs), "--bs", str(a.bs), "--se", str(a.se),
            "--sr", a.sr, "--version", a.version, "--f0method", a.f0method,
            "--index-rate", a.index_rate]
    if a.data:
        argv += ["--data", a.data]
    sys.argv = argv
    print("argv:", " ".join(argv), flush=True)

    try:
        runpy.run_path(argv[0], run_name="__main__")
    except SystemExit as e:
        print(f"===== launcher exit: {e} =====", flush=True)
        return int(e.code or 0)
    finally:
        print("===== launcher done =====", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

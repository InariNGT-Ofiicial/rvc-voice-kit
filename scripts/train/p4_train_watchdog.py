# -*- coding: utf-8 -*-
"""DS娘训练看门狗：训练进程死掉就自动拉起，直到 e150 落盘。

★ 为什么需要它（2026-10-26 实测）：
  训练在这个沙箱里**反复死**，而且死法不止一种：
  1. **会话结束被回收**：上一轮训练（00:33→03:03）跑到 e43 时，
     它挂在 WorkBuddy 的受管后台任务上，会话一结束就被回收 ⇒
     日志停在某一轮中间、没有 traceback。
  2. **宿主 RAM 提交量耗尽**（本轮真凶）：日志里出现
     `fatal : Memory allocation failure` + `DataLoader worker exited unexpectedly`
     + 后续 `CUDA error: unknown error`。
     这**不是**显存 OOM（GPU 只用了 1.1/8 GB），
     是 Windows 提交量到了上限（63.5/65.0 GB）——
     训练主进程 9.5 GB + 4 个 dataloader worker 各 2.7 GB ≈ 20 GB，
     叠加常驻换声服务 5.4 GB + Photoshop 6.3 GB + WorkBuddy 3 GB，
     再叠加一轮推理进程的 3 GB 就爆了。

  两种死法都不能靠「再跑一次」解决，所以这里做两件事：
  a. 把 dataloader worker 从 4 降到 2（省 ~5.4 GB，语料只有 279 条，2 个足够）；
  b. 死了自动重拉 —— RVC 自带 checkpoint 续训，重拉后从最近的 e20/e40/e60… 接着跑，
     不会从头再来。

用法（脱离会话，Start-Process 启动）：
    python scripts/train/p4_train_watchdog.py --exp dsxiaoyi_a3_v1 --epochs 150
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

KIT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KIT))
from rvcx import paths as P  # noqa: E402

REPO = P.REPO
WEIGHTS = REPO / "assets" / "weights"
LOGDIR = Path(r"E:/AI-RVC/logs")
EXP = "dsxiaoyi_a3_v1"


def epoch_of(exp: str) -> int:
    """从已落盘的推理权重里读最大轮次（dsxiaoyi_a3_v1_e60_s7800.pth → 60）。"""
    best = 0
    for f in WEIGHTS.glob(f"{exp}_e*_s*.pth"):
        stem = f.stem
        try:
            best = max(best, int(stem.split("_e")[1].split("_s")[0]))
        except (IndexError, ValueError):
            continue
    # 最终模型另存为 <exp>.pth（savee 的 name 分支）
    return best


def trainer_alive() -> bool:
    """用锁文件里的 pid 判活（p3_resume_train.py 写的）。"""
    lock = LOGDIR / ".train.lock"
    if not lock.is_file():
        return False
    try:
        info = lock.read_text(encoding="utf-8").strip().split()
        pid = int(info[0])
    except (OSError, ValueError, IndexError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=EXP)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--se", type=int, default=20)
    ap.add_argument("--sleep", type=int, default=45)
    a = ap.parse_args()

    LOGDIR.mkdir(parents=True, exist_ok=True)
    wd_log = LOGDIR / f"{a.exp}_watchdog.log"

    def log(msg):
        line = f"[watchdog {time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(wd_log, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    log(f"看门狗启动：目标 e{a.epochs}，当前已落盘 e{epoch_of(a.exp)}")
    stalls = 0
    while True:
        cur = epoch_of(a.exp)
        if cur >= a.epochs:
            log(f"已达 e{cur} >= e{a.epochs}，看门狗退出")
            return 0
        if trainer_alive():
            stalls = 0
            time.sleep(a.sleep)
            continue

        stalls += 1
        log(f"训练进程不在（当前 e{cur}），第 {stalls} 次拉起")
        # 死过一次就退避，避免「秒崩秒拉」把机器打满
        time.sleep(min(30 * stalls, 120))
        lock = LOGDIR / ".train.lock"
        if lock.is_file():
            try:
                lock.unlink()          # 僵尸锁：p3 会接管
            except OSError:
                pass
        cmd = [str(P.PY), str(KIT / "scripts" / "train" / "p3_resume_train.py"),
               "--exp", a.exp, "--epochs", str(a.epochs),
               "--bs", str(a.bs), "--se", str(a.se)]
        log("拉起：" + " ".join(cmd))
        try:
            subprocess.Popen(cmd, cwd=str(KIT),
                             stdout=open(LOGDIR / f"{a.exp}_wd_child.log", "a",
                                         encoding="utf-8", buffering=1),
                             stderr=subprocess.STDOUT)
        except Exception as e:
            log(f"拉起失败：{e}")
        time.sleep(a.sleep)


if __name__ == "__main__":
    sys.exit(main())

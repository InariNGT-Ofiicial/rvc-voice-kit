# -*- coding: utf-8 -*-
"""断点续训：直接从 RVC 的 train.train 恢复，跳过 p1_run 的「已有模型就跳过」逻辑。

★ 为什么需要它：p1_run.step_train() 见到 assets/weights/<exp>_e*.pth 就跳过，
  所以崩溃后无法用它续训。这里直接调 train.train，RVC 自己会
  load_checkpoint(latest_checkpoint_path(logs/<exp>, "D_*.pth")) 恢复。

★ 为什么 OOM（dsxiaoyi_a3_v1 在 e22 崩）：**两个训练进程并存**。
  launcher 日志里 00:27 与 00:33 各启动一次，两个进程同时抢同一块 8 GB 显卡，
  判别器前向时 conv2d 分配失败。切片长度与芳乃一致（max 4.0 s）⇒ 不是数据问题。
  所以这里加一把**进程锁**：同一实验只允许一个训练进程。

用法：
  python p3_resume_train.py --exp dsxiaoyi_a3_v1 --epochs 150 --bs 4 --se 20
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rvcx import paths as P  # noqa: E402

LOCK = Path(r"E:/AI-RVC/logs/.train.lock")


def acquire_lock(exp: str) -> "tuple[int, bool]":
    """返回 (pid, 是否新拿到的锁)。拿不到就退出，绝不并发跑同一实验。"""
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()} {exp} {time.time()}\n".encode())
        os.close(fd)
        return os.getpid(), True
    except FileExistsError:
        info = ""
        try:
            info = LOCK.read_text(encoding="utf-8").strip()
        except Exception:
            pass
        pid = int(info.split()[0]) if info.split()[:1] and info.split()[0].isdigit() else -1
        alive = False
        if pid > 0:
            try:
                os.kill(pid, 0)
                alive = True
            except OSError:
                alive = False
        if alive:
            raise SystemExit(
                f"[resume] 训练锁被占用：pid={pid} exp={exp}\n"
                f"          拒绝并发启动（并发正是上次 OOM 的原因）。\n"
                f"          若确认那是僵尸锁，删除后重试：del {LOCK}"
            )
        # 僵尸锁：接管
        LOCK.write_text(f"{os.getpid()} {exp} {time.time()}\n", encoding="utf-8")
        return os.getpid(), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--se", type=int, default=20)
    ap.add_argument("--sr", default="48k")
    ap.add_argument("--version", default="v2")
    ap.add_argument("--steps", default="train,index,infer")
    a = ap.parse_args()

    pid, _ = acquire_lock(a.exp)
    print(f"[resume] 拿到训练锁 pid={pid} exp={a.exp}")

    repo = P.REPO
    env = P.child_env()
    # 训练日志单独落盘（RVC 的 loguru 输出 GBK 且跨轮追加）
    logdir = Path(r"E:/AI-RVC/logs")
    logdir.mkdir(parents=True, exist_ok=True)
    log = logdir / f"{a.exp}_resume.log"
    out = open(log, "a", encoding="utf-8", buffering=1)

    cmd = [str(P.PY), "-m", "train.train",
           "-e", a.exp, "-sr", a.sr, "-f0", "1", "-bs", str(a.bs), "-g", "0",
           "-te", str(a.epochs), "-se", str(a.se),
           "-pg", f"assets/pretrained_v2/f0G{a.sr}.pth",
           "-pd", f"assets/pretrained_v2/f0D{a.sr}.pth",
           "-l", "1", "-c", "0", "-sw", "1", "-v", a.version]
    print("[resume] " + " ".join(cmd))
    print(f"[resume] 日志 -> {log}")
    env["PYTHONUNBUFFERED"] = "1"
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=str(repo), stdout=out, stderr=subprocess.STDOUT, env=env)
        rc = p.returncode
    finally:
        out.close()
    print(f"[resume] 退出 rc={rc} 用时 {time.time()-t0:.0f}s")
    print(f"[resume] 日志尾部：")
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        for l in lines[-25:]:
            print("  " + l[:200])
    except Exception as e:
        print("  读日志失败", e)
    try:
        LOCK.unlink()
    except Exception:
        pass
    return rc


if __name__ == "__main__":
    sys.exit(main())

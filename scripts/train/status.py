# -*- coding: utf-8 -*-
"""训练状态速查：从 04_train.log 的本轮片段里算 epoch / 速度 / ETA。

日志跨轮追加且是 GBK，所以先按 ===== 时间头切块，取最后一个含该 exp 的块；
epoch 用 loss_disc 行数 + checkpoint 文件名双重印证（不依赖中文，避开编码坑）。
"""
import sys as _sys
from pathlib import Path as _pth

_sys.path.insert(0, str(_pth(__file__).resolve().parents[2]))
from rvcx import paths as _paths  # noqa: E402

import re
import sys
import time
from pathlib import Path

ROOT = _paths.ROOT
LOG = ROOT / "logs" / "04_train.log"
EXP = sys.argv[1] if len(sys.argv) > 1 else "fangno_p2"
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
SPE = 111          # 445 切片 / bs4 ≈ 111 step/epoch
LOGDIR = ROOT / "repo" / "logs" / EXP

raw = LOG.read_bytes().decode("gbk", errors="replace").replace("\r\n", "\n")
blocks = re.split(r"\n\n===== (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) =====\n", raw)
# blocks = [前导, t1, b1, t2, b2, ...] 取最后一个含本 exp 的块
cur_t, cur_b = None, None
for i in range(1, len(blocks) - 1, 2):
    if f"-e {EXP}" in blocks[i + 1]:
        cur_t, cur_b = blocks[i], blocks[i + 1]
if cur_b is None:
    print("未找到本轮日志块")
    raise SystemExit(0)

t0 = time.mktime(time.strptime(cur_t, "%Y-%m-%d %H:%M:%S"))
elapsed = (time.time() - t0) / 60

losses = re.findall(r"loss_disc=([\d.]+)", cur_b)
# log_interval=200 step → 约 1.8 epoch 一条
est_ep = min(EPOCHS, round(len(losses) * 200 / SPE))
cks = sorted(int(p.stem.split("_")[1]) for p in LOGDIR.glob("G_*.pth")) if LOGDIR.is_dir() else []
small = sorted((ROOT / "repo" / "assets" / "weights").glob(f"{EXP}_e*.pth"))

print(f"exp={EXP}  开跑 {cur_t}  已运行 {elapsed:.0f} 分钟")
print(f"loss_disc {len(losses)} 条 | 最近 {losses[-1] if losses else '—'}"
      f" | 初值 {losses[0] if losses else '—'}")
if est_ep:
    per = elapsed / est_ep
    print(f"≈ epoch {est_ep}/{EPOCHS} | {per:.1f} 分钟/轮 | 剩余 ≈ {per * (EPOCHS - est_ep):.0f} 分钟")
print(f"日志检查点 {cks[-5:] if cks else '—（每 20 轮存一次）'}")
print(f"小模型 {[p.name for p in small] if small else '—（每 20 轮导出一次）'}")

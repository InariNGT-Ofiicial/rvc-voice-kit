# -*- coding: utf-8 -*-
"""从 logs/<exp>/G_*.pth 提取推理用小模型（等价于 WebUI 的 ckpt 提取页，但不碰 train.log 的 eval）。"""
import sys as _sys
from pathlib import Path as _pth

_sys.path.insert(0, str(_pth(__file__).resolve().parents[2]))
from rvcx import paths as _paths  # noqa: E402

import os
import sys
import glob

REPO = str(_paths.REPO)
os.chdir(REPO)
sys.path.insert(0, REPO)

import torch  # noqa: E402
from train.utils import get_hparams_from_dir  # noqa: E402
from train.process_ckpt import savee  # noqa: E402

exp = sys.argv[1] if len(sys.argv) > 1 else "fangno_p1"
sr = sys.argv[2] if len(sys.argv) > 2 else "48k"

cands = sorted(glob.glob(f"logs/{exp}/G_*.pth"), key=os.path.getmtime)
if not cands:
    raise SystemExit(f"没有找到 logs/{exp}/G_*.pth")
cp = cands[-1]

d = torch.load(cp, map_location="cpu", weights_only=False)
hps = get_hparams_from_dir(f"logs/{exp}")
step = int(d.get("iteration", 0))
name = f"{exp}_step{step}"
msg = savee(d["model"], sr, 1, name, step, "v2", hps)
print("ckpt:", cp, "| iteration:", step)
print("result:", msg)
print("->", f"assets/weights/{name}.pth")

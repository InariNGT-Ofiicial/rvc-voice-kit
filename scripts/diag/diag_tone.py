# -*- coding: utf-8 -*-
"""诊断：声调指标口径是否可信（pyin vs rmvpe / DTW vs 线性对齐）。

背景：zh03/zh04 的 DTW tone_r 掉到 0.44~0.77，但 ASR 的**声调 CER = 0.0000**
（一个字都没听错调）。两个判据不能同时为真，先查是哪个坏了。
"""
import sys as _sys
from pathlib import Path as _pth

_sys.path.insert(0, str(_pth(__file__).resolve().parents[2]))
from rvcx import paths as _paths  # noqa: E402

import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(_paths.REPO))
from infer.rmvpe import RMVPE  # noqa: E402

DLG = _paths.WORK / "dialogue_zh"


def load(p, sr):
    y, s = sf.read(str(p), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(1)
    if s != sr:
        y = librosa.resample(y, orig_sr=s, target_sr=sr)
    return y


def pyin_st(y):
    f0 = librosa.pyin(y, fmin=65, fmax=1100, sr=44100, frame_length=2048,
                      hop_length=512)[0]
    return f0


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rmv = RMVPE(str(_paths.REPO / "assets" / "rmvpe" / "rmvpe.pt"), False, device=dev)

    for lid in ("zh01", "zh03", "zh04"):
        for sid in ("src", "e120"):
            p = (DLG / "src_tts" / f"{lid}.wav") if sid == "src" \
                else (DLG / "out" / f"{lid}_{sid}.wav")
            y16 = load(p, 16000)
            f0r = rmv.infer_from_audio(y16, thred=0.03)
            v = f0r[f0r > 0]
            # pyin 对照
            y44 = load(p, 44100)
            f0p = pyin_st(y44)
            vp = f0p[~np.isnan(f0p)]
            print(f"{lid} {sid:4s} rmvpe: voiced={v.size}/{f0r.size} "
                  f"中位={np.median(v):.1f}Hz 半音std={12*np.log2(v/np.median(v)).std():.2f} | "
                  f"pyin: voiced={vp.size} 中位={np.median(vp):.1f}Hz "
                  f"半音std={12*np.log2(vp/np.median(vp)).std():.2f}", flush=True)


if __name__ == "__main__":
    main()

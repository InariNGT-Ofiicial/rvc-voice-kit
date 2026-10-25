# -*- coding: utf-8 -*-
"""比较两种声调对齐口径：DTW(subseq) vs 线性时间对齐（只比共有的浊音帧）。

为什么试线性：出/入时长比只有 0.988~0.997（差 ≤1.2%），DTW 的自由弯曲
反而是噪声来源（短句 zh02 上 DTW 给出 0.64，而 ASR 声调 CER = 0）。
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
CASES = [("zh01", "e120"), ("zh02", "e120"), ("zh02", "e200"),
         ("zh03", "e120_ir0"), ("zh03", "e120"), ("zh04", "e120")]


def load(p, sr=16000):
    y, s = sf.read(str(p), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(1)
    if s != sr:
        y = librosa.resample(y, orig_sr=s, target_sr=sr)
    return y


def f0_series(y, rmv):
    f0 = np.asarray(rmv.infer_from_audio(y, thred=0.03), dtype=np.float64)
    st = np.full(f0.shape, np.nan)
    ok = f0 > 0
    st[ok] = 12 * np.log2(f0[ok] / 55.0)
    med = np.full(st.shape, np.nan)
    for i in range(st.size):
        seg = st[max(0, i - 2):i + 3]
        seg = seg[~np.isnan(seg)]
        if seg.size:
            med[i] = np.median(seg)
    st[(~np.isnan(st)) & (~np.isnan(med)) & (np.abs(st - med) > 6)] = np.nan
    return st


def linear_cmp(a, b):
    n = min(a.size, b.size)
    xa = np.linspace(0, 1, a.size)
    xb = np.linspace(0, 1, b.size)
    g = np.linspace(0, 1, n)
    # 只在两端的「浊音连续段」内插值：先把 nan 用前向填充避免差太多
    fa = np.interp(g, xa[~np.isnan(a)], a[~np.isnan(a)])
    fb = np.interp(g, xb[~np.isnan(b)], b[~np.isnan(b)])
    # 用最近邻判断共有浊音区
    va = np.isnan(np.interp(g, xa, np.where(np.isnan(a), np.nan, 0))) if False else None
    na = np.interp(g, xa, np.where(np.isnan(a), 1.0, 0.0))
    nb = np.interp(g, xb, np.where(np.isnan(b), 1.0, 0.0))
    m = (na < 0.5) & (nb < 0.5)
    if m.sum() < 20:
        return None, None
    x, y = fa[m], fb[m]
    r = float(np.corrcoef(x, y)[0, 1])
    return round(r, 3), round(float(np.sqrt(np.mean((x - y) ** 2))), 2)


def dtw_cmp(a, b):
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    C = np.abs(a[:, None] - b[None, :])
    _, wp = librosa.sequence.dtw(C=C, subseq=True)
    p = wp[::-1] - 1
    p[:, 0] = np.clip(p[:, 0], 0, a.size - 1)
    p[:, 1] = np.clip(p[:, 1], 0, b.size - 1)
    x, y = a[p[:, 0]], b[p[:, 1]]
    return round(float(np.corrcoef(x, y)[0, 1]), 3), \
        round(float(np.sqrt(np.mean((x - y) ** 2))), 2)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rmv = RMVPE(str(_paths.REPO / "assets" / "rmvpe" / "rmvpe.pt"), False, device=dev)
    for lid, sid in CASES:
        s = f0_series(load(DLG / "src_tts" / f"{lid}.wav"), rmv)
        o = f0_series(load(DLG / "out" / f"{lid}_{sid}.wav"), rmv)
        lr, lrm = linear_cmp(s, o)
        dr, drm = dtw_cmp(s, o)
        print(f"{lid} {sid:10s} 线性 r={lr} rmse={lrm}st | DTW r={dr} rmse={drm}st")


if __name__ == "__main__":
    main()

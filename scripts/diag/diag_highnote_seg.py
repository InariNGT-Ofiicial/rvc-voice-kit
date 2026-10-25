# -*- coding: utf-8 -*-
"""专项诊断：单个高音缺陷段的真伪判定。

要回答三个问题（逐段）：
  Q1 源在缺陷处**到底有没有那个基频**？
     ⇒ 若 src 在 f0_src 附近的窄带能量极低，rmvpe 给的源 f0 就是幻觉，
        判据本身失效 ⇒ 不是缺陷。
  Q2 换声后该处**实际输出**是什么？
     ⇒ 谐波能量峰值在哪、f0 落在哪。
  Q3 差异是「模型画错了」还是「源本来就没有可比的东西」？

用法：
  python diag_highnote_seg.py --src <源.wav> --conv <换声.wav> \
      --t0 115.69 --t1 116.07 --f0-src 759.4 [--json out.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rvcx import paths as P  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import f0tool  # noqa: E402

sys.path.insert(0, str(P.REPO))

RMVPE_PATH = None


def load_mono(path, t0=None, t1=None):
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(1)
    if t0 is not None:
        y = y[int(t0 * sr): int(t1 * sr)]
    return y, sr


def f0_rmvpe(y, sr, rmv):
    return f0tool.f0_rmvpe(y, sr, rmv)


def f0_pm(y, sr):
    return f0tool.f0_pm(y, sr)


def stft_mag(y, sr, n=2048, hop=512):
    """返回 (freqs, S)，S 形状 (freq, frames)。"""
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    win = np.hanning(n)
    nfr = 1 + (len(y) - n) // hop
    frames = np.lib.stride_tricks.as_strided(
        y, shape=(nfr, n), strides=(y.strides[0] * hop, y.strides[0]))
    S = np.abs(np.fft.rfft(frames * win, axis=1)).T
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    return freqs, S


def band_pct(S, freqs, f0, bw=10.0):
    m = (freqs >= f0 - bw) & (freqs <= f0 + bw)
    tot = float((S ** 2).sum()) + 1e-12
    return float((S[m] ** 2).sum() / tot * 100.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--conv", required=True)
    ap.add_argument("--t0", type=float, required=True)
    ap.add_argument("--t1", type=float, required=True)
    ap.add_argument("--f0-src", type=float, default=0.0)
    ap.add_argument("--pad", type=float, default=0.25)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    from infer.rmvpe import RMVPE
    rmv = RMVPE(str(P.REPO / "assets" / "rmvpe" / "rmvpe.pt"), False,
                device="cuda" if torch.cuda.is_available() else "cpu")

    t0 = max(0.0, a.t0 - a.pad)
    t1 = a.t1 + a.pad
    ys, sr = load_mono(a.src, t0, t1)
    yc, _ = load_mono(a.conv, t0, t1)

    rep = {"seg": [round(t0, 3), round(t1, 3)], "src": a.src, "conv": a.conv}

    # ---- 逐 20 ms 的 RMS 包络 ----
    hop = int(sr * 0.02)
    n = min(len(ys), len(yc)) // hop

    def env(y):
        e = np.sqrt((y[:n * hop].reshape(n, hop) ** 2).mean(1))
        return 20 * np.log10(np.maximum(e, 1e-12))

    es, ec = env(ys), env(yc)
    rep["rms_env_db"] = [[round(t0 + i * 0.02, 3), round(es[i], 1), round(ec[i], 1)]
                         for i in range(n)]

    # ---- f0 轨迹（两口径同为 10 ms 步长）----
    fs, ts = f0_rmvpe(ys, sr, rmv)
    fc, tc = f0_rmvpe(yc, sr, rmv)
    rep["f0_frame_sec"] = f0tool.FRAME_SEC
    rep["f0_src_rmvpe"] = [round(float(v), 1) for v in fs]
    rep["f0_conv_rmvpe"] = [round(float(v), 1) for v in fc]
    try:
        ps, ps_t = f0_pm(ys, sr)
        pc, pc_t = f0_pm(yc, sr)
        rep["f0_src_pm"] = [round(float(v), 1) for v in ps]
        rep["f0_conv_pm"] = [round(float(v), 1) for v in pc]
    except Exception as e:
        rep["f0_pm_error"] = str(e)

    # ---- Q1：源在该处有没有 f0_src 的基频？----
    fs_s, Ss = stft_mag(ys, sr)
    fs_c, Sc = stft_mag(yc, sr)
    probe_f0s = [a.f0_src] if a.f0_src > 0 else []
    # 再扫几个候选：源的 rmvpe 中位
    v = fs[fs > 0]
    if v.size:
        probe_f0s.append(round(float(np.median(v)), 1))
    # conv 的 rmvpe 中位
    v2 = fc[fc > 0]
    if v2.size:
        probe_f0s.append(round(float(np.median(v2)), 1))
    rep["narrow_band_probe"] = []
    for f0 in probe_f0s:
        rep["narrow_band_probe"].append({
            "f0_hz": f0,
            "src_narrow_pct": round(band_pct(Ss, fs_s, f0, 10.0), 4),
            "conv_narrow_pct": round(band_pct(Sc, fs_c, f0, 10.0), 4),
        })

    # ---- Q2：各自的频谱峰值（top5）----
    def peaks(S, freqs):
        k = min(5, S.shape[0])
        idx = np.argsort(S.sum(axis=1))[-k:][::-1]
        mx = float(S.sum(axis=1).max()) + 1e-12
        return [[round(float(freqs[i]), 1), round(float(S[i].sum() / mx * 100), 2)]
                for i in idx]

    rep["src_spectrum_top5"] = peaks(Ss, fs_s)
    rep["conv_spectrum_top5"] = peaks(Sc, fs_c)

    # ---- 判决 ----
    src_rms = float(np.sqrt((ys ** 2).mean()))
    conv_rms = float(np.sqrt((yc ** 2).mean()))
    rep["src_rms_db"] = round(20 * np.log10(max(src_rms, 1e-12)), 2)
    rep["conv_rms_db"] = round(20 * np.log10(max(conv_rms, 1e-12)), 2)

    out = json.dumps(rep, ensure_ascii=False, indent=2, default=float)
    if a.json:
        Path(a.json).write_text(out, encoding="utf-8")
        print("报告 ->", a.json)
    else:
        print(out)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    main()

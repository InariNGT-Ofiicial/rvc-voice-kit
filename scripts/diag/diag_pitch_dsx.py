# -*- coding: utf-8 -*-
"""诊断：《下等马》DS娘(e20) 换声 vs 泠鸢原唱 —— 音准偏差 + 高音区表现。

★ 为什么先量再修（skill ai-song-cover-pipeline §5.1）：
  「修音/调音九成情况不该动手」—— 弱音上的音准偏差是 F0 提取伪影，
  硬修会把本来准确的部分改坏。所以必须按能量分桶看偏差，
  并且单独检查高音区（换声最容易崩的地方）。

本脚本产出（stdout，JSON 可选）：
  1. F0 分布对比（中位 / 半音 std / 八度跳变数）
  2. 按帧能量分桶的平均绝对偏差（音分）
  3. 线性对齐后的逐帧偏差曲线统计
  4. 高音区专项：原唱高音帧上，DS娘版的偏差与八度错误率
  5. 高音区的谐波结构（HNR 代理）对比
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

sys.path.insert(0, str(P.REPO))
from infer.rmvpe import RMVPE  # noqa: E402

SR = 16000  # rmvpe 内部按 16k 工作
HOP = 160  # 10 ms


def load_mono16(p):
    y, s = sf.read(str(p), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if s != SR:
        import librosa
        y = librosa.resample(y, orig_sr=s, target_sr=SR)
    return y


def f0_series(y, rmv):
    f0 = np.asarray(rmv.infer_from_audio(y, thred=0.03), dtype=np.float64)
    return f0


def frame_energy(y, n):
    """与 f0 帧对齐的 RMS 包络（16k 域）。"""
    hop = 160
    m = (len(y) // hop) * hop
    a = y[:m].reshape(-1, hop)
    rms = np.sqrt((a.astype(np.float64) ** 2).mean(axis=1))
    if len(rms) < n:
        rms = np.pad(rms, (0, n - len(rms)))
    return rms[:n]


def cents_ratio(f_ref, f_test):
    """逐帧音分偏差（只在两边都有声时有效）。"""
    ok = (f_ref > 0) & (f_test > 0)
    out = np.full(f_ref.shape, np.nan)
    out[ok] = 1200.0 * np.log2(f_test[ok] / f_ref[ok])
    return out, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="原唱干声（泠鸢）")
    ap.add_argument("--test", required=True, help="换声结果（DS娘）")
    ap.add_argument("--json", default=None)
    ap.add_argument("--hi-thresh", type=float, default=0.80,
                    help="高音区阈值：原唱 F0 分位（0.8 ≈ 最高的 20%%）")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rmv = RMVPE(str(P.REPO / "assets" / "rmvpe" / "rmvpe.pt"), False, device=dev)

    yr = load_mono16(a.ref)
    yt = load_mono16(a.test)
    n = min(len(yr), len(yt))
    yr, yt = yr[:n], yt[:n]

    fr = f0_series(yr, rmv)
    ft = f0_series(yt, rmv)
    L = min(len(fr), len(ft))
    fr, ft = fr[:L], ft[:L]
    er = frame_energy(yr, L)

    rep = {}

    # ---- 1. F0 分布 ----
    vr, vt = fr[fr > 0], ft[ft > 0]
    med_r = float(np.median(vr))
    med_t = float(np.median(vt))
    rep["f0"] = {
        "ref_voiced": int(vr.size), "test_voiced": int(vt.size),
        "ref_median_hz": round(med_r, 2), "test_median_hz": round(med_t, 2),
        "median_diff_hz": round(med_t - med_r, 2),
        "ref_semitone_std": round(float(12 * np.log2(vr / med_r).std()), 3),
        "test_semitone_std": round(float(12 * np.log2(vt / med_t).std()), 3),
        "ref_p95_hz": round(float(np.percentile(vr, 95)), 2),
        "test_p95_hz": round(float(np.percentile(vt, 95)), 2),
    }

    # ---- 2. 逐帧偏差 + 八度错误 ----
    dev_c, ok = cents_ratio(fr, ft)
    absd = np.abs(dev_c[ok])
    # 八度错误：|偏差| > 500 音分（半八度以上）
    oct_err = np.abs(dev_c) > 500
    rep["accuracy"] = {
        "frames_compared": int(ok.sum()),
        "median_abs_cents": round(float(np.median(absd)), 2),
        "mean_abs_cents": round(float(absd.mean()), 2),
        "p90_abs_cents": round(float(np.percentile(absd, 90)), 2),
        "pct_over_100c": round(float((absd > 100).mean() * 100), 2),
        "pct_over_50c": round(float((absd > 50).mean() * 100), 2),
        "octave_like_frames": int(oct_err.sum()),
        "octave_like_pct": round(float(oct_err.sum() / max(ok.sum(), 1) * 100), 3),
    }

    # ---- 3. 按能量分桶（关键：区分「真跑调」与「弱音抓错基频」） ----
    e = er.copy()
    e[~ok] = np.nan
    qs = np.nanpercentile(e, [20, 40, 60, 80])
    buckets = []
    prev = -np.inf
    labels = ["最弱20%", "20-40%", "40-60%", "60-80%", "最强20%"]
    edges = [-np.inf, qs[0], qs[1], qs[2], qs[3], np.inf]
    for i, lab in enumerate(labels):
        m = (e > edges[i]) & (e <= edges[i + 1]) & ok
        if m.sum() < 20:
            buckets.append({"bucket": lab, "n": int(m.sum()), "mean_abs_cents": None})
            continue
        buckets.append({
            "bucket": lab,
            "n": int(m.sum()),
            "mean_abs_cents": round(float(np.nanmean(abs(dev_c[m]))), 2),
            "median_abs_cents": round(float(np.nanmedian(abs(dev_c[m]))), 2),
            "signed_mean_cents": round(float(np.nanmean(dev_c[m])), 2),
        })
    rep["by_energy_bucket"] = buckets

    # ---- 4. 高音区专项 ----
    hi_thr = float(np.percentile(vr, a.hi_thresh * 100))
    hi_mask = (fr >= hi_thr) & ok
    hi_dev = dev_c[hi_mask]
    rep["high_notes"] = {
        "threshold_hz": round(hi_thr, 2),
        "n_frames": int(hi_mask.sum()),
        "mean_abs_cents": round(float(np.nanmean(abs(hi_dev))), 2),
        "median_abs_cents": round(float(np.nanmedian(abs(hi_dev))), 2),
        "signed_mean_cents": round(float(np.nanmean(hi_dev)), 2),
        "pct_over_50c": round(float((np.abs(hi_dev) > 50).mean() * 100), 2),
        "pct_over_100c": round(float((np.abs(hi_dev) > 100).mean() * 100), 2),
        "octave_like_pct": round(float((np.abs(hi_dev) > 500).mean() * 100), 2),
    }

    # 高音区：原唱在高音、但换声后掉出音高（丢失/被压扁）
    hi_only = (fr >= hi_thr) & (ft <= 0)
    rep["high_notes"]["frames_lost_voicing"] = int(hi_only.sum())
    # 换声后最高音 vs 原唱最高音
    rep["high_notes"]["test_max_hz"] = round(float(vt.max()), 2)
    rep["high_notes"]["ref_max_hz"] = round(float(vr.max()), 2)

    # ---- 5. 高音区谐波结构（HNR 代理）对比 ----
    # 高音区帧的谱平坦度：越高说明噪声/混叠越多，音质发糊
    def hf_flatness(y, frames, hop=160, nfft=1024):
        vals = []
        for i in frames:
            s = i * hop
            if s + nfft > len(y):
                continue
            seg = y[s:s + nfft] * np.hanning(nfft)
            S = np.abs(np.fft.rfft(seg)) ** 2 + 1e-12
            psd = S / S.max()
            gm = float(np.exp(np.mean(np.log(psd))))
            am = float(np.mean(psd))
            vals.append(gm / max(am, 1e-12))
        return float(np.mean(vals)) if vals else float("nan")

    hi_idx = np.where(hi_mask)[0]
    if hi_idx.size > 20:
        step = max(1, hi_idx.size // 300)
        sel = hi_idx[::step]
        rep["high_notes"]["ref_spectral_flatness"] = round(hf_flatness(yr, sel), 5)
        rep["high_notes"]["test_spectral_flatness"] = round(hf_flatness(yt, sel), 5)

    # ---- 6. 静音/气口 ----
    rep["voicing"] = {
        "ref_zero_ratio": round(float(np.mean(fr <= 0)), 4),
        "test_zero_ratio": round(float(np.mean(ft <= 0)), 4),
    }

    print(json.dumps(rep, ensure_ascii=False, indent=2))
    if a.json:
        Path(a.json).write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                                encoding="utf-8")


if __name__ == "__main__":
    main()

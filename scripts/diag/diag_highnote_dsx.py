# -*- coding: utf-8 -*-
"""高音区根因定位：把「高音掉 2 个半音 + 谱脏 9 倍」拆成可判定的子问题。

要回答三个问题（决定是重推理还是修 F0）：
  Q1 输入侧还是输出侧的锅？
     → 分别用 rmvpe 提取原唱(Vocals) 与换声结果的 F0，看原唱本身高音是否也"掉"。
     → 若原唱 F0 最高能到 843 Hz 而输出只到 744 Hz ⇒ 是**模型/推理**侧丢失，不是提取。
  Q2 rmvpe 在高音区是否抓错（八度错）？
     → 用 pm(parselmouth) 做独立提取，对比 rmvpe，两者不一致处即疑似抓错。
  Q3 高音段落在时间轴的哪里？
     → 输出「需要修」的时间段清单，供分段重推理用。

输出 JSON + 时间段清单（stdout）。
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

SR = 16000
HOP = 160  # 10ms


def load_mono16(p):
    y, s = sf.read(str(p), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if s != SR:
        import librosa
        y = librosa.resample(y, orig_sr=s, target_sr=SR)
    return y


def f0_rmvpe(y, rmv):
    return np.asarray(rmv.infer_from_audio(y, thred=0.03), dtype=np.float64)


def f0_pm(y):
    """parselmouth（Praat）独立提取，作为交叉校验的第二意见。"""
    import parselmouth
    t = parselmouth.Sound(y, SR)
    pitch = t.to_pitch_ac(time_step=0.01, voicing_threshold=0.6,
                          pitch_floor=50.0, pitch_ceiling=1100.0)
    return np.asarray(pitch.selected_array["frequency"], dtype=np.float64)


def rms_frames(y, n):
    m = (len(y) // HOP) * HOP
    a = y[:m].reshape(-1, HOP)
    r = np.sqrt((a.astype(np.float64) ** 2).mean(axis=1))
    if len(r) < n:
        r = np.pad(r, (0, n - len(r)))
    return r[:n]


def align_len(a, b):
    n = min(len(a), len(b))
    return a[:n], b[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="原唱人声（分离后的 Vocals / in_lead）")
    ap.add_argument("--conv", required=True, help="换声结果")
    ap.add_argument("--json", default=None)
    ap.add_argument("--hi-q", type=float, default=0.80,
                    help="高音区定义：原唱 F0 的分位（0.8 = 最高 20%）")
    ap.add_argument("--min-run", type=float, default=0.35,
                    help="成段的最短时长（秒）")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rmv = RMVPE(str(P.REPO / "assets" / "rmvpe" / "rmvpe.pt"), False, device=dev)

    ys = load_mono16(a.src)
    yc = load_mono16(a.conv)
    n = min(len(ys), len(yc))
    ys, yc = ys[:n], yc[:n]

    fr = f0_rmvpe(ys, rmv)
    fc = f0_rmvpe(yc, rmv)
    fr, fc = align_len(fr, fc)
    e = rms_frames(ys, len(fr))

    vr = fr[fr > 0]
    hi_thr = float(np.percentile(vr, a.hi_q * 100))

    # ---- Q1: 音域上界对比 ----
    # 用「连续稳定的高音」而非单帧最大值，避免被孤立毛刺拉高
    def stable_max(f0, win=15):
        v = np.where(f0 > 0, f0, np.nan)
        best = 0.0
        for i in range(0, len(v) - win, 3):
            seg = v[i:i + win]
            if np.isnan(seg).any():
                continue
            best = max(best, float(np.median(seg)))
        return best

    rep = {}
    rep["Q1_range"] = {
        "hi_threshold_hz": round(hi_thr, 2),
        "src_stable_max_hz": round(stable_max(fr), 2),
        "conv_stable_max_hz": round(stable_max(fc), 2),
        "src_p99_hz": round(float(np.percentile(vr, 99)), 2),
        "conv_p99_hz": round(float(np.percentile(fc[fc > 0], 99)), 2),
        "src_frames_above_600hz": int((fr > 600).sum()),
        "conv_frames_above_600hz": int((fc > 600).sum()),
        "src_frames_above_700hz": int((fr > 700).sum()),
        "conv_frames_above_700hz": int((fc > 700).sum()),
    }
    lost_range = rep["Q1_range"]["src_stable_max_hz"] - rep["Q1_range"]["conv_stable_max_hz"]
    rep["Q1_range"]["stable_max_drop_hz"] = round(lost_range, 2)
    rep["Q1_range"]["stable_max_drop_semitones"] = round(
        12 * np.log2(max(rep["Q1_range"]["src_stable_max_hz"], 1) /
                     max(rep["Q1_range"]["conv_stable_max_hz"], 1)), 2)

    # ---- Q2: rmvpe vs pm 交叉校验（原唱侧）----
    ps = f0_pm(ys)
    ps, fr2 = align_len(ps, fr)
    ok = (ps > 0) & (fr2 > 0)
    if ok.sum() > 50:
        diff = 1200 * np.log2(ps[ok] / fr2[ok])
        hi = (fr2[ok] > hi_thr)
        rep["Q2_crosscheck_src"] = {
            "n": int(ok.sum()),
            "median_abs_cents": round(float(np.median(np.abs(diff))), 2),
            "pct_octave_disagree": round(float((np.abs(diff) > 500).mean() * 100), 3),
            "hi_n": int(hi.sum()),
            "hi_median_abs_cents": round(float(np.median(np.abs(diff[hi]))), 2)
            if hi.sum() else None,
            "hi_pct_octave_disagree": round(float((np.abs(diff[hi]) > 500).mean() * 100), 3)
            if hi.sum() else None,
        }

    # ---- Q3: 时间轴 —— 逐帧算偏差，圈出「需要修」的区段 ----
    both = (fr > 0) & (fc > 0)
    cents = np.full(fr.shape, np.nan)
    cents[both] = 1200 * np.log2(fc[both] / fr[both])
    hi = (fr > hi_thr) & both
    # 需要修：高音区 且（|偏差|>60 音分 或 换声掉出有声）
    lost = (fr > hi_thr) & (fc <= 0)
    need = hi & (np.abs(cents) > 60)
    need = need | lost

    # 成段
    segs = []
    i = 0
    while i < len(need):
        if need[i]:
            j = i
            while j + 1 < len(need) and need[j + 1]:
                j += 1
            dur = (j - i + 1) * HOP / SR
            if dur >= a.min_run:
                c = cents[i:j + 1]
                segs.append({
                    "start_sec": round(i * HOP / SR, 2),
                    "end_sec": round((j + 1) * HOP / SR, 2),
                    "dur_sec": round(dur, 2),
                    "mean_cents": round(float(np.nanmean(c)), 1),
                    "mean_abs_cents": round(float(np.nanmean(np.abs(c))), 1),
                    "max_abs_cents": round(float(np.nanmax(np.abs(c))), 1),
                    "src_f0_med": round(float(np.median(fr[i:j + 1])), 1),
                    "conv_f0_med": round(float(np.median(fc[i:j + 1])), 1),
                    "lost_voicing_frames": int(lost[i:j + 1].sum()),
                })
            i = j + 1
        else:
            i += 1
    rep["Q3_segments"] = {
        "count": len(segs),
        "total_dur_sec": round(sum(s["dur_sec"] for s in segs), 2),
        "segments": segs[:60],
    }

    # 高音区整体偏差方向（是否系统性偏低 → 若是，则修「模型音域」而非逐句）
    if hi.sum() > 30:
        rep["high_bias"] = {
            "signed_mean_cents": round(float(np.nanmean(cents[hi])), 2),
            "signed_median_cents": round(float(np.nanmedian(cents[hi])), 2),
            "pct_below": round(float((cents[hi] < -60).mean() * 100), 2),
            "pct_above": round(float((cents[hi] > 60).mean() * 100), 2),
        }

    print(json.dumps(rep, ensure_ascii=False, indent=2))
    if a.json:
        Path(a.json).write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                                encoding="utf-8")


if __name__ == "__main__":
    main()

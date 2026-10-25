# -*- coding: utf-8 -*-
"""精细缺陷扫描：把「需要修的高音/掉音/八度错」逐帧定位成可施工的时间段清单。

★ 为什么不用 60 音分阈值：60 只找到 2 段 0.82 秒，但训练集音域天花板
  （语料 >600Hz 占比 0.00%）意味着缺陷可能散落多处。本脚本用三类硬缺陷判据：

  D1 八度错   : |cents| > 300（超半八度 ⇒ 听感上是另一个音）
  D2 掉音     : 原唱有声，换声无声（模型没唱出来）
  D3 高音跑调 : 原唱 f0 > 高音阈值 且 |cents| > 30（明显可听的音准偏差）

输出：每段含 起止秒 / 时长 / 原唱中位 f0 / 换声中位 f0 / 偏差 / 原唱电平 / 换声电平。
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
HOP = 160  # 10 ms


def load_mono16(p):
    y, s = sf.read(str(p), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if s != SR:
        import librosa
        y = librosa.resample(y, orig_sr=s, target_sr=SR)
    return y


def frames_rms(y, n):
    m = (len(y) // HOP) * HOP
    a = y[:m].reshape(-1, HOP)
    r = np.sqrt((a.astype(np.float64) ** 2).mean(axis=1))
    if len(r) < n:
        r = np.pad(r, (0, n - len(r)))
    return r[:n]


def db(x):
    return 20 * np.log10(max(float(x), 1e-12))


def group(mask, min_run_sec, hop=0.01):
    """把布尔掩码切成连续段，返回 (start_sec, end_sec) 列表。"""
    segs = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            if (j - i + 1) * hop >= min_run_sec:
                segs.append((i * hop, (j + 1) * hop))
            i = j + 1
        else:
            i += 1
    return segs


def merge(segs, gap=0.25):
    """把间隔小于 gap 的段合并（同一处缺陷常被一两个静音帧切开）。"""
    if not segs:
        return []
    out = [list(segs[0])]
    for s, e in segs[1:]:
        if s - out[-1][1] <= gap:
            out[-1][1] = e
        else:
            out.append([s, e])
    return [tuple(x) for x in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="原唱人声")
    ap.add_argument("--conv", required=True, help="换声结果")
    ap.add_argument("--json", default=None)
    ap.add_argument("--hi-q", type=float, default=0.80)
    ap.add_argument("--dev-tol", type=float, default=30.0, help="D3 偏差阈值（音分）")
    ap.add_argument("--oct-tol", type=float, default=300.0, help="D1 八度错阈值（音分）")
    ap.add_argument("--min-run", type=float, default=0.15, help="成段最短秒数")
    ap.add_argument("--gap", type=float, default=0.25, help="合并间隔秒数")
    ap.add_argument("--pad", type=float, default=0.0, help="每段前后扩展秒数")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rmv = RMVPE(str(P.REPO / "assets" / "rmvpe" / "rmvpe.pt"), False, device=dev)

    ys = load_mono16(a.src)
    yc = load_mono16(a.conv)
    n = min(len(ys), len(yc))
    ys, yc = ys[:n], yc[:n]
    fr = np.asarray(rmv.infer_from_audio(ys, thred=0.03), dtype=np.float64)
    fc = np.asarray(rmv.infer_from_audio(yc, thred=0.03), dtype=np.float64)
    L = min(len(fr), len(fc))
    fr, fc = fr[:L], fc[:L]
    er = frames_rms(ys, L)
    ec = frames_rms(yc, L)

    both = (fr > 0) & (fc > 0)
    cents = np.full(L, np.nan)
    cents[both] = 1200 * np.log2(fc[both] / fr[both])

    vr = fr[fr > 0]
    hi_thr = float(np.percentile(vr, a.hi_q * 100))

    d1 = both & (np.abs(cents) > a.oct_tol)
    d2 = (fr > 0) & (fc <= 0)
    d3 = (fr > hi_thr) & both & (np.abs(cents) > a.dev_tol)
    # 高音区整体偏低（系统性）：原唱在高音，换声更低
    d4 = (fr > hi_thr) & both & (cents < -60)

    report = {
        "meta": {
            "src": a.src, "conv": a.conv,
            "hi_threshold_hz": round(hi_thr, 2),
            "dev_tol_cents": a.dev_tol, "oct_tol_cents": a.oct_tol,
            "frames": int(L),
        },
        "counts": {
            "D1_octave_error_frames": int(d1.sum()),
            "D2_dropped_voicing_frames": int(d2.sum()),
            "D3_high_dev_frames": int(d3.sum()),
            "D4_high_flat_frames": int(d4.sum()),
            "D2_pct_of_src_voiced": round(float(d2.sum() / max((fr > 0).sum(), 1) * 100), 3),
        },
    }

    def build(mask, kind):
        segs = merge(group(mask, a.min_run), a.gap)
        out = []
        for s, e in segs:
            i0 = max(0, int(s / 0.01))
            i1 = min(L, int(e / 0.01))
            cm = cents[i0:i1]
            cm_ok = cm[~np.isnan(cm)]
            out.append({
                "kind": kind,
                "start_sec": round(max(0.0, s - a.pad), 2),
                "end_sec": round(min(L * 0.01, e + a.pad), 2),
                "dur_sec": round(e - s, 2),
                "src_f0_med": round(float(np.median(fr[i0:i1][fr[i0:i1] > 0])), 1)
                if (fr[i0:i1] > 0).any() else None,
                "conv_f0_med": round(float(np.median(fc[i0:i1][fc[i0:i1] > 0])), 1)
                if (fc[i0:i1] > 0).any() else None,
                "mean_cents": round(float(np.mean(cm_ok)), 1) if cm_ok.size else None,
                "mean_abs_cents": round(float(np.mean(np.abs(cm_ok))), 1) if cm_ok.size else None,
                "max_abs_cents": round(float(np.max(np.abs(cm_ok))), 1) if cm_ok.size else None,
                "src_rms_db": round(db(np.sqrt((er[i0:i1] ** 2).mean())), 1),
                "conv_rms_db": round(db(np.sqrt((ec[i0:i1] ** 2).mean())), 1),
            })
        return out

    segs_all = []
    segs_all += build(d1, "D1_octave")
    segs_all += build(d2, "D2_drop")
    segs_all += build(d3, "D3_high_dev")
    segs_all += build(d4, "D4_high_flat")
    segs_all.sort(key=lambda x: x["start_sec"])
    report["segments"] = segs_all
    report["total_defect_dur_sec"] = round(sum(s["dur_sec"] for s in segs_all), 2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if a.json:
        Path(a.json).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                encoding="utf-8")


if __name__ == "__main__":
    main()

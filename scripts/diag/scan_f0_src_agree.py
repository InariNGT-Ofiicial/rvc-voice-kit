# -*- coding: utf-8 -*-
"""源侧 f0 双口径分歧扫描（rmvpe vs parselmouth/Praat）。

★ 为什么要单独扫源（实测踩到的坑）：
  RVC 内部**就是用 rmvpe 提取 f0** 的。
  若 rmvpe 在**源**信号上就八度下错（实测：115.7 s 的 700→842 Hz 假声滑音
  被 rmvpe 判成 253 Hz，Praat 给的是 697→842 Hz），
  那么 RVC 会照着 253 Hz 合成 ⇒ 高音整个画不出来。
  而常规的「conv vs src」缺陷扫描，src 侧也用 rmvpe ⇒
  **两边一致地错 ⇒ 漏报**。
  ⇒ 缺陷扫描必须引入一个与 rmvpe 独立的第三口径（Praat）。

判据：
  逐点 |1200*log2(f0_rmvpe / f0_pm)| >= 分歧阈值（默认 100 音分 ≈ 半八度）
  且两者都有值 ⇒ 该点「两口径不一致」。
  连续成段 ⇒ 输出可疑段列表（含该处 src 的谱峰与能量占比）。

用法：
  python scan_f0_src_agree.py --src <源.wav> --json out.json [--tol 100]
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


def f0_rmvpe(y, sr, rmv):
    return f0tool.f0_rmvpe(y, sr, rmv)


def f0_pm(y, sr):
    return f0tool.f0_pm(y, sr)


def group(mask, hop_sec):
    """把布尔 mask 成段，返回 [start_idx, end_idx) 列表。"""
    segs = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            segs.append((i, j))
            i = j
        else:
            i += 1
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--tol", type=float, default=100.0, help="分歧阈值（音分）")
    ap.add_argument("--min-sec", type=float, default=0.10, help="成段最短秒数")
    ap.add_argument("--gap", type=float, default=0.15, help="合并间隔秒数")
    a = ap.parse_args()

    from infer.rmvpe import RMVPE
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rmv = RMVPE(str(P.REPO / "assets" / "rmvpe" / "rmvpe.pt"), False, device=dev)

    y, sr = sf.read(a.src, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(1)
    total = len(y)

    fr, tr = f0_rmvpe(y, sr, rmv)
    fp, tp = f0_pm(y, sr)

    # ★ 两口径现在同为 10 ms 步长（rmvpe 降到 16 kHz 后帧间隔 160/16000 = 10 ms）
    #   ⇒ 可逐点直接对齐（此前按 44.1 kHz 域的 3.628 ms 帧率对齐全是错位）
    n = min(len(fr), len(fp))
    fr, fp = fr[:n], fp[:n]
    tr, tp = tr[:n], tp[:n]

    both = (fr > 0) & (fp > 0)
    cents = np.zeros(n)
    cents[both] = 1200.0 * np.log2(fr[both] / fp[both])
    disagree = both & (np.abs(cents) >= a.tol)

    hop_sec = 0.01
    segs = group(disagree, hop_sec)
    gap = int(a.gap / hop_sec)
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], s[1])
        else:
            merged.append(s)

    out = []
    for (i0, i1) in merged:
        if (i1 - i0) * hop_sec < a.min_sec:
            continue
        t0, t1 = i0 * hop_sec, i1 * hop_sec
        m = slice(i0, i1)
        r_med = float(np.median(fr[m][fr[m] > 0])) if (fr[m] > 0).any() else None
        p_med = float(np.median(fp[m][fp[m] > 0])) if (fp[m] > 0).any() else None
        out.append({
            "start_sec": round(t0, 2), "end_sec": round(t1, 2),
            "dur_sec": round(t1 - t0, 2),
            "f0_rmvpe_med": round(r_med, 1) if r_med else None,
            "f0_pm_med": round(p_med, 1) if p_med else None,
            "cents_med": round(float(np.median(cents[m])), 1),
            "ratio": round(float(r_med / p_med), 3) if (r_med and p_med) else None,
            "near_octave": bool(abs(round(float(np.median(cents[m])) / 100.0)) % 12 in (0, 6)),
        })

    rep = {
        "src": a.src, "sr": sr, "total_sec": round(total / sr, 2),
        "tol_cents": a.tol,
        "frames": int(n),
        "both_voiced_frames": int(both.sum()),
        "disagree_frames": int(disagree.sum()),
        "disagree_pct": round(float(disagree.sum()) / max(int(both.sum()), 1) * 100, 3),
        "segments": out,
    }
    txt = json.dumps(rep, ensure_ascii=False, indent=2)
    if a.json:
        Path(a.json).write_text(txt, encoding="utf-8")
        print(f"报告 -> {a.json}")
    print(json.dumps({k: v for k, v in rep.items() if k != "segments"},
                     ensure_ascii=False, indent=2))
    for s in out:
        print("  %7.2f–%7.2fs (%4.2fs) rmvpe %8s  pm %8s  %+7.1f音分  ratio %s%s"
              % (s["start_sec"], s["end_sec"], s["dur_sec"],
                 s["f0_rmvpe_med"], s["f0_pm_med"], s["cents_med"], s["ratio"],
                 "  ←近八度" if s["near_octave"] else ""))


if __name__ == "__main__":
    main()

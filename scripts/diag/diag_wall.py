# -*- coding: utf-8 -*-
"""验证「模型音域天花板」假设，并决定补正策略。

假设：DS娘语料 F0 上限 ~657 Hz（实测 p99.9=416 Hz，>600Hz 占 0.00%），
      所以模型在 >600 Hz 区域没有训练信号 ⇒ 换声时该区域必然画不出。

★ 这个脚本要回答的：
  ① 模型的"音域墙"在哪？（把语料 F0 直方图与缺陷段的 F0 画在同一条轴上）
  ② 缺陷段的"正确音高"落在墙内还是墙外？
     - 墙内 ⇒ 是偶发（该重推理，可能参数问题）
     - 墙外 ⇒ 是天花板（只能靠升 key 或换模型，不该指望重推理）
  ③ 试算：把缺陷段整体升 N 个半音（用原唱 f0 作为目标），
     新音高是否落回墙内 ⇒ 给出可行的 key 补偿量。
"""
import argparse
import glob
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


def load_mono16(p):
    y, s = sf.read(str(p), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if s != SR:
        import librosa
        y = librosa.resample(y, orig_sr=s, target_sr=SR)
    return y


def corpus_f0(rmv, exp_dir):
    fs = sorted(glob.glob(exp_dir + "/0_gt_wavs/*.wav"))
    vals = []
    for f in fs:
        y, s = sf.read(f, dtype="float32")
        if y.ndim > 1:
            y = y.mean(1)
        if s != SR:
            import librosa
            y = librosa.resample(y, orig_sr=s, target_sr=SR)
        f0 = np.asarray(rmv.infer_from_audio(y, thred=0.03))
        vals.append(f0[f0 > 0])
    return np.concatenate(vals) if vals else np.array([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="dsxiaoyi_a3_v1")
    ap.add_argument("--src", required=True)
    ap.add_argument("--xcheck", required=True, help="xcheck_defects.py 的输出 json")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rmv = RMVPE(str(P.REPO / "assets" / "rmvpe" / "rmvpe.pt"), False, device=dev)

    cv = corpus_f0(rmv, str(P.REPO / "logs" / a.exp))
    wall = float(np.percentile(cv, 99.9))       # 语料音域墙
    wall_max = float(cv.max())

    xs = json.loads(Path(a.xcheck).read_text(encoding="utf-8"))
    real = [s for s in xs if s["verdict"] == "REAL_DEFECT"]

    rep = {
        "corpus_f0": {
            "frames": int(cv.size),
            "median_hz": round(float(np.median(cv)), 1),
            "p99_hz": round(float(np.percentile(cv, 99)), 1),
            "p999_hz": round(wall, 1),
            "max_hz": round(wall_max, 1),
            "wall_definition": "语料 p99.9（模型音域墙）",
            "pct_over_600hz": round(float((cv > 600).mean() * 100), 4),
            "pct_over_500hz": round(float((cv > 500).mean() * 100), 4),
            "pct_over_400hz": round(float((cv > 400).mean() * 100), 4),
        },
        "wall_hz": round(wall, 1),
        "wall_max_hz": round(wall_max, 1),
        "defects": [],
    }

    for s in real:
        src_f0 = s["src_f0_pm"] or s["src_f0_rmvpe"]
        conv_f0 = s["conv_f0_pm"] or s["conv_f0_rmvpe"]
        ratio = conv_f0 / src_f0 if src_f0 else None
        # 试算：把源音高降半音数，直到落回墙内
        semis_needed = None
        if src_f0 and wall:
            k = 0
            while src_f0 / (2 ** (k / 12)) > wall and k < 24:
                k += 1
            semis_needed = k
        rep["defects"].append({
            "seg": s["seg"],
            "src_f0_hz": src_f0,
            "conv_f0_hz": conv_f0,
            "ratio": round(ratio, 3) if ratio else None,
            "semitones_off": round(12 * np.log2(ratio), 1) if ratio else None,
            "src_inside_wall": bool(src_f0 <= wall),
            "conv_inside_wall": bool(conv_f0 <= wall),
            "shift_semitones_to_fit_wall": semis_needed,
            "diagnosis": ("源音高在墙外⇒天花板" if src_f0 > wall else
                          "源音高在墙内⇒偶发，可重推理"),
        })

    # 决策建议
    n_out = sum(1 for d in rep["defects"] if d["src_f0_hz"] > wall)
    rep["decision"] = {
        "defects_outside_wall": n_out,
        "defects_total": len(rep["defects"]),
        "verdict": ("语料音域天花板是主因" if n_out >= max(1, len(rep["defects"]) // 2)
                    else "主要是偶发，参数可救"),
        "actions": [
            ("换声全程加 key 补偿" if n_out else "保持 key=0"),
            "重训语料必须补 600-900Hz 样本（当前 >600Hz 占 0.00%）",
            "对墙外段单独重推理 + 微调 key 是唯一可行的人工修法",
        ],
    }

    print(json.dumps(rep, ensure_ascii=False, indent=2))
    if a.json:
        Path(a.json).write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                                encoding="utf-8")


if __name__ == "__main__":
    main()

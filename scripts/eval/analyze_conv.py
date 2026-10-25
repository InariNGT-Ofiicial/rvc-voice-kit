# -*- coding: utf-8 -*-
"""
换声客观诊断：原干声 vs 换声输出
判据沿用 ai-song-cover-pipeline §6：MFCC 距离 / F0 中位差 / 谱心变化 / RMS / silent_ratio
用法：python analyze_conv.py <原始.wav> <换声.wav> [out.json]
"""
import json
import sys
import warnings

import numpy as np
import soundfile as sf
import librosa

warnings.filterwarnings("ignore")


def load(p):
    y, sr = sf.read(p, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(1)
    return y, sr


def feats(y, sr):
    # MFCC（音色包络）
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=512)
    # 谱质心
    c = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)
    # F0（pyin，歌声域放宽）
    f0 = librosa.pyin(y, fmin=65, fmax=1100, sr=sr, frame_length=2048, hop_length=512)[0]
    voiced = f0[~np.isnan(f0)]
    return {
        "mfcc": m.mean(1),
        "centroid": float(np.mean(c)),
        "f0_median": float(np.median(voiced)) if voiced.size else None,
        "f0_voiced_ratio": float(voiced.size / max(len(f0), 1)),
    }


def main():
    a_in, a_out = sys.argv[1], sys.argv[2]
    out_json = sys.argv[3] if len(sys.argv) > 3 else None

    y1, sr1 = load(a_in)
    y2, sr2 = load(a_out)
    if sr1 != sr2:
        y2 = librosa.resample(y2, orig_sr=sr2, target_sr=sr1)
        sr2 = sr1

    n = min(len(y1), len(y2))
    y1, y2 = y1[:n], y2[:n]

    f1, f2 = feats(y1, sr1), feats(y2, sr1)

    mfcc_dist = float(np.linalg.norm(f1["mfcc"] - f2["mfcc"]))
    cent_chg = (f2["centroid"] - f1["centroid"]) / f1["centroid"] * 100

    def db(x):
        return 20 * np.log10(max(float(np.sqrt(np.mean(x ** 2))), 1e-12))

    def silent_ratio(x, thr=0.0):
        return float(np.mean(np.abs(x) <= thr))

    rep = {
        "dur_in_s": round(len(y1) / sr1, 3),
        "dur_out_s": round(len(y2) / sr2, 3),
        "sr": sr1,
        "mfcc_l2": round(mfcc_dist, 3),
        "centroid_hz_in": round(f1["centroid"], 1),
        "centroid_hz_out": round(f2["centroid"], 1),
        "centroid_change_pct": round(float(cent_chg), 2),
        "f0_median_in_hz": None if f1["f0_median"] is None else round(f1["f0_median"], 1),
        "f0_median_out_hz": None if f2["f0_median"] is None else round(f2["f0_median"], 1),
        "rms_in_dbfs": round(db(y1), 2),
        "rms_out_dbfs": round(db(y2), 2),
        "peak_out": round(float(np.max(np.abs(y2))), 4),
        "silent_ratio_in": round(silent_ratio(y1, 1e-6), 5),
        "silent_ratio_out": round(silent_ratio(y2, 1e-6), 5),
        "exact_zero_ratio_out": round(silent_ratio(y2, 0.0), 5),
    }
    if rep["f0_median_in_hz"] and rep["f0_median_out_hz"]:
        rep["f0_median_diff_hz"] = round(rep["f0_median_out_hz"] - rep["f0_median_in_hz"], 1)

    # 判定（★ 不能写 x or 999：差值恰为 0.0 是 falsy，会被误判成不保持）
    d = rep.get("f0_median_diff_hz")
    rep["judge"] = {
        "换声生效": rep["mfcc_l2"] > 5,
        "音高保持": (abs(d) < 30) if d is not None else False,
        "无削波": rep["peak_out"] < 1.0,
        "电平方差_dB": round(rep["rms_out_dbfs"] - rep["rms_in_dbfs"], 2),
    }

    text = json.dumps(rep, ensure_ascii=False, indent=2)
    print(text)
    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\n-> {out_json}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""多系统 A/B 串联试听：原声 + 各换声版本，响度对齐后首尾相接。

纪律（来自 ai-song-cover-pipeline 验收规程）：
  1) 响度必须先对齐再拼接，否则听到的差异全是「谁更响」；
  2) 对齐后整体回退防削波（峰值 >0.95 才退，不改变相对关系）；
  3) 段间留 0.4 s 静音，方便人工切点。

用法：
  python ab_make.py --out out.wav --pair "原声:a.wav" "RVC:RVC路径/lead15s.wav" "DDSP:..."
"""
import argparse

import numpy as np
import soundfile as sf
import librosa

SR = 48000
GAP = 0.4


def load(p):
    y, sr = sf.read(p, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y


def rms_db(x):
    return 20 * np.log10(max(float(np.sqrt(np.mean(x ** 2))), 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pair", nargs="+", required=True, help='格式 "标签:路径"')
    a = ap.parse_args()

    items = []
    for s in a.pair:
        label, path = s.split(":", 1)
        items.append((label, load(path)))

    ref = rms_db(items[0][1])          # 以第一段（原声）为基准
    print(f"基准（{items[0][0]}）RMS {ref:.2f} dBFS")
    segs, gap = [], np.zeros(int(GAP * SR), dtype=np.float32)
    for label, y in items:
        g = ref - rms_db(y)
        y2 = y * (10 ** (g / 20))
        segs += [y2, gap]
        print(f"{label:12s} 补偿 {g:+.2f} dB -> RMS {rms_db(y2):.2f} dBFS | 峰值 {np.abs(y2).max():.3f}")

    out = np.concatenate(segs)
    pk = np.abs(out).max()
    if pk > 0.95:
        out = out * (0.95 / pk)
        print(f"整体回退 {-20 * np.log10(0.95 / pk):.2f} dB 防削波")
    sf.write(a.out, out, SR)
    print(f"\n-> {a.out}  {len(out) / SR:.1f}s  峰值 {np.abs(out).max():.3f}")


if __name__ == "__main__":
    main()

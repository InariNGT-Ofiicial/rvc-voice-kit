# -*- coding: utf-8 -*-
"""LTAS 相关性：判断换声后音色是否朝目标角色移动。
用法：python ltas_check.py <源干声.wav> <换声.wav> <目标参考目录> [样本数]
判据来自 ai-song-cover-pipeline §8：单句频谱质心会被音素内容主导，必须用长时平均频谱 + 参考集相关。
"""
import sys
import glob
import random

import numpy as np
import librosa
import soundfile as sf

SR = 44100


def load(p):
    y, sr = sf.read(p, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y


def ltas(y):
    # 只取有声段，避免静音把平均谱拉平
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    e = S.sum(0)
    thr = np.percentile(e, 40)
    S = S[:, e >= thr]
    m = S.mean(1)
    return librosa.power_to_db(m + 1e-10)


def main():
    src, conv, refdir = sys.argv[1], sys.argv[2], sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 25
    random.seed(0)

    files = sorted(glob.glob(f"{refdir}/*.wav"))
    random.shuffle(files)
    refs = [ltas(load(f)) for f in files[:n] if True]
    L = min(len(r) for r in refs)
    ref = np.mean([r[:L] for r in refs], axis=0)

    def corr(x):
        x = x[:L]
        return float(np.corrcoef(x, ref)[0, 1])

    a = ltas(load(src))
    b = ltas(load(conv))
    ra, rb = corr(a), corr(b)

    print(f"参考集：{n} 条目标角色片段（{refdir}）")
    print(f"换声前（源音色）与目标 LTAS 相关 r = {ra:+.4f}")
    print(f"换声后（RVC 输出）与目标 LTAS 相关 r = {rb:+.4f}")
    print(f"音色位移 Δr = {rb - ra:+.4f}  ->  {'朝目标移动 ✓' if rb > ra else '未越过源 ✗'}")
    # 自测上限参考
    self_r = [corr(ltas(load(f))) for f in files[n:n + min(8, len(files) - n)]]
    if self_r:
        print(f"目标本人其余片段自测 r（上限参考）= {np.mean(self_r):+.4f}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""f0 提取共享工具。

★★★ 本项目踩过的最贵的一个坑（整套缺陷判定曾因此作废）★★
  RMVPE 的 mel 滤波器组是按 **16 kHz** 训练/设计的。
  若把 44.1 kHz 音频直接喂给它 ⇒ 频率轴被压缩 44100/16000 = **2.756 倍**
  ⇒ **所有 f0 读数偏低 2.756 倍**。

  实测（源《下等马》 vocals.wav）：

      段            pm(Praat)   rmvpe@44.1k   rmvpe@16k(正确)
      130–180 s       499.8        181.0        499.1
      115 s           815.0        290.6        814.6
      30 s            295.7        107.1        294.6

  连带后果：一切以**音分**为单位的判据都被放大了 2.756 倍
  （八度错 |cents|>300 实际等价于真值 |cents|>108.8；音准偏差同理），
  ⇒ 之前统计出来的「缺陷帧数」「音分偏差」全部是伪影。

  ★ RVC 本身没有这个毛病：infer/vc/modules.py 的 `load_audio(path, 16000)`
    先降到 16 kHz 再喂 rmvpe ⇒ 模型侧一直是对的，受影响的只有我们的测量脚本。

  ★ 降采样到 16 kHz 的额外好处：帧间隔 = 160/16000 = **10 ms**，
    与 parselmouth 的 `to_pitch_ac(time_step=0.01)` **完全同频**
    ⇒ 两口径逐点可直接对齐（此前按 44.1 k 域的 3.628 ms 帧率去配对，纯属错位）。

用法：
    from f0tool import f0_rmvpe, f0_pm, align_pair
    fr = f0_rmvpe(y, sr, rmv)      # 返回 (f0 序列, 时间轴秒)
    fp = f0_pm(y, sr)              # 同为 10 ms 步长
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RMV_SR = 16000          # RMVPE 的工作采样率
FRAME_SEC = 0.01        # 两口径统一后的帧间隔（秒）


def to_rmv16(y, sr):
    """降到 16 kHz（RMVPE 的工作域）。已是 16 kHz 则原样返回。"""
    if sr == RMV_SR:
        return np.ascontiguousarray(y, dtype=np.float32)
    import librosa
    return np.ascontiguousarray(
        librosa.resample(np.asarray(y, dtype=np.float32), orig_sr=sr,
                         target_sr=RMV_SR).astype(np.float32))


def f0_rmvpe(y, sr, rmv, thred=0.03):
    """★ 正确用法：自动降到 16 kHz 再喂 RMVPE。
    返回 (f0 序列 float64, 时间轴 float64 秒)，帧间隔恒为 10 ms。"""
    y16 = to_rmv16(y, sr)
    f0 = np.asarray(rmv.infer_from_audio(y16, thred=thred), dtype=np.float64)
    t = np.arange(len(f0)) * FRAME_SEC
    return f0, t


def f0_pm(y, sr, time_step=0.01, floor=50.0, ceiling=1100.0):
    """parselmouth/Praat 独立口径。返回 (f0 序列, 时间轴 秒)，10 ms 步长。"""
    import parselmouth
    snd = parselmouth.Sound(np.ascontiguousarray(np.asarray(y, dtype=np.float64)), sr)
    pit = snd.to_pitch_ac(time_step=time_step, voicing_threshold=0.6,
                           pitch_floor=floor, pitch_ceiling=ceiling)
    f0 = np.asarray(pit.selected_array["frequency"], dtype=np.float64)
    t = np.arange(len(f0)) * time_step
    return f0, t


def align_pair(fr, tr, fp, tp, win=0.005):
    """把两条不同帧率的 f0 轨迹对齐到 fp 的时间轴。
    返回 (fr_aligned, mask)：fr_aligned 为 NaN 表示该点无对应 voiced 帧。"""
    fr_al = np.full(len(fp), np.nan)
    for j, t in enumerate(tp):
        lo = np.searchsorted(tr, t - win)
        hi = np.searchsorted(tr, t + win)
        if hi > lo:
            w = fr[lo:hi]
            w = w[w > 0]
            if w.size:
                fr_al[j] = float(np.median(w))
    return fr_al, np.isfinite(fr_al) & (fp > 0)


def cents_ratio(a, b):
    """a 相对 b 的音分差（正 = a 更高）。"""
    m = (a > 0) & (b > 0)
    out = np.zeros(len(a))
    out[m] = 1200.0 * np.log2(a[m] / b[m])
    return out


def stft_mag(y, sr, n=2048, hop=512):
    """返回 (freqs, S)，S 形状 (freq, frames)。"""
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    win = np.hanning(n)
    nfr = 1 + (len(y) - n) // hop
    fr = np.lib.stride_tricks.as_strided(
        y, shape=(nfr, n), strides=(y.strides[0] * hop, y.strides[0]))
    S = np.abs(np.fft.rfft(fr * win, axis=1)).T
    return np.fft.rfftfreq(n, 1.0 / sr), S


def band_pct(S, freqs, f0, bw=15.0):
    """窄带能量占总能量百分比。"""
    m = (freqs >= f0 - bw) & (freqs <= f0 + bw)
    tot = float((S ** 2).sum()) + 1e-12
    return float((S[m] ** 2).sum() / tot * 100.0)

# -*- coding: utf-8 -*-
"""交叉验证：换声输出的高音缺陷，是【模型真的画错】还是【rmvpe 误判】？

★ 为什么要交叉验证（项目纪律：两个判据打架先怀疑测量工具）：
  rmvpe 在换声输出上曾出现八度跳变（pyin 更糟），单一口径可能把
  "测量伪影"当成 "模型缺陷" 去修 —— 修完反而更糟。
  ⇒ 用 parselmouth(Praat) 独立提取同一段，两家都说是低音才是真缺陷。

另外做频谱侧证：若输出真的在 784 Hz，
  则 784 / 1568 / 2352 Hz 附近应有能量峰；若真实基频是 412 Hz，
  则 412 / 824 / 1236 Hz 有峰、784 Hz 反而弱。
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
HOP = 160


def load_mono16(p, t0=None, t1=None):
    y, s = sf.read(str(p), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if s != SR:
        import librosa
        y = librosa.resample(y, orig_sr=s, target_sr=SR)
    if t0 is not None:
        y = y[int(t0 * SR): int(t1 * SR)]
    return y


def f0_rmvpe(y, rmv):
    return np.asarray(rmv.infer_from_audio(y, thred=0.03), dtype=np.float64)


def f0_pm(y):
    import parselmouth
    t = parselmouth.Sound(y, SR)
    pit = t.to_pitch_ac(time_step=0.01, voicing_threshold=0.6,
                        pitch_floor=50.0, pitch_ceiling=1100.0)
    return np.asarray(pit.selected_array["frequency"], dtype=np.float64)


def harmonic_energy(y, f0, sr=44100):
    """在 f0 及其 2/3/4 次谐波处测能量（相对全带），判断真实基频。"""
    import numpy as np
    if f0 <= 0:
        return {}
    n = min(len(y), sr * 3)
    seg = y[:n]
    if len(seg) < 2048:
        return {}
    w = np.hanning(len(seg))
    S = np.abs(np.fft.rfft(seg * w, n=8192)) ** 2
    freqs = np.fft.rfftfreq(8192, 1.0 / sr)
    tot = S.sum() + 1e-20
    out = {}
    for k in (1, 2, 3, 4):
        f = f0 * k
        if f >= sr / 2:
            continue
        m = (freqs > f - 25) & (freqs < f + 25)
        out[f"h{k}_{f:.0f}Hz"] = round(float(S[m].sum() / tot * 100), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--conv", required=True)
    ap.add_argument("--segments", required=True,
                    help="JSON 文件：[{start_sec,end_sec,...}, ...]")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rmv = RMVPE(str(P.REPO / "assets" / "rmvpe" / "rmvpe.pt"), False, device=dev)

    segs = json.loads(Path(a.segments).read_text(encoding="utf-8"))
    out = []
    for s in segs:
        t0 = max(0.0, s["start_sec"] - 0.25)
        t1 = s["end_sec"] + 0.25
        ys = load_mono16(a.src, t0, t1)
        yc = load_mono16(a.conv, t0, t1)

        fr_r = f0_rmvpe(ys, rmv)
        fr_p = f0_pm(ys)
        fc_r = f0_rmvpe(yc, rmv)
        fc_p = f0_pm(yc)

        def med(v):
            v = v[v > 0]
            return round(float(np.median(v)), 1) if v.size else None

        # 频谱侧证（用 44.1k 原始文件）
        ys44, sr44 = sf.read(a.src, dtype="float32", always_2d=False)
        if ys44.ndim > 1:
            ys44 = ys44.mean(1)
        yc44, _ = sf.read(a.conv, dtype="float32", always_2d=False)
        if yc44.ndim > 1:
            yc44 = yc44.mean(1)
        i0, i1 = int(t0 * sr44), int(t1 * sr44)
        src_h = harmonic_energy(ys44[i0:i1], med(fr_r) or 0, sr44)
        conv_h = harmonic_energy(yc44[i0:i1], med(fc_r) or 0, sr44)

        # 判决：两家口径是否一致认为偏低
        src_m = med(fr_r)
        conv_m_rmv = med(fc_r)
        conv_m_pm = med(fc_p)
        both_low = (conv_m_rmv is not None and conv_m_pm is not None
                    and src_m is not None
                    and conv_m_rmv < src_m * 0.75 and conv_m_pm < src_m * 0.75)
        # pm 在换声输出上的可靠性：pm 是否也检测到有声
        pm_voiced = int((fc_p > 0).sum())

        out.append({
            "seg": f'{s["start_sec"]:.2f}-{s["end_sec"]:.2f}',
            "src_f0_rmvpe": med(fr_r),
            "src_f0_pm": med(fr_p),
            "conv_f0_rmvpe": conv_m_rmv,
            "conv_f0_pm": conv_m_pm,
            "conv_pm_voiced_frames": pm_voiced,
            "verdict": "REAL_DEFECT" if both_low else
                       ("MEASUREMENT_ARTIFACT" if conv_m_pm is not None
                        and src_m and conv_m_pm >= src_m * 0.75 else "INCONCLUSIVE"),
            "src_harmonic_energy_pct": src_h,
            "conv_harmonic_energy_pct": conv_h,
        })

    print(json.dumps(out, ensure_ascii=False, indent=2))
    if a.json:
        Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                encoding="utf-8")


if __name__ == "__main__":
    main()

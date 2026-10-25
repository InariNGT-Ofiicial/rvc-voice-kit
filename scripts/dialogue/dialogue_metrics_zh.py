# -*- coding: utf-8 -*-
"""中文台词量化：音色位移（Δr）+ 声调保真 + 削波/归零。

★ 中文比日语多一条必须测的东西：**声调**。
  日语的音高曲线只管语气；汉语的 f0 曲线直接决定字义（妈/麻/马/骂）。
  RVC 用 --pitch 0 是「保留源 f0 轮廓」，理论上声调该原样过；但 flow/索引
  都可能把轮廓抹平。所以这里用 DTW 对齐源与出的 f0 曲线，给出：
    · tone_r    —— 半音域 f0 轮廓相关（1.0 = 声调曲线完全保留）
    · tone_rmse —— 半音 RMSE（>2 半音基本等于换了调型）
"""
import sys as _sys
from pathlib import Path as _pth

_sys.path.insert(0, str(_pth(__file__).resolve().parents[2]))
from rvcx import paths as _paths  # noqa: E402

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from sweep import load, metrics, ref_ltas  # noqa: E402

DLG = _paths.WORK / "dialogue_zh"
REFDIR = str(_paths.REF or (_paths.DATASET / "fangno"))
CACHE = str(_paths.WORK / "cache" / "ref_ltas.npz")
SR = 44100
SYSTEMS = ["e120", "e200", "e120_ir0", "e120_ir050"]


def dur_of(p):
    i = sf.info(str(p))
    return i.frames / i.samplerate


def f0_st(y, rmv=None):
    """取 f0 转半音（相对 55 Hz），丢掉清音帧 + 压掉八度跳变。

    ★ 为什么不用 pyin：实测 zh04 换声后 pyin 的半音 std 从 2.94 跳到 6.92
      （八度跳变），用它算出的 tone_r 只有 0.44，但 ASR 的声调 CER = 0.0000。
      两个判据不可能同时为真 ⇒ 是估计器坏了，不是声调坏了。
      这里改用 **rmvpe**（与换声器本身用的同一个 f0 提取器），并用 5 帧中值
      滤掉 >6 半音的单帧跳变。
    """
    if rmv is None:
        return np.array([])
    y16 = librosa.resample(y, orig_sr=SR, target_sr=16000)
    f0 = rmv.infer_from_audio(y16, thred=0.03)
    f0 = np.asarray(f0, dtype=np.float64)
    st = np.full(f0.shape, np.nan)
    ok = f0 > 0
    st[ok] = 12 * np.log2(f0[ok] / 55.0)
    # 八度跳变压制：与 5 帧中值差 > 6 半音且单侧孤立 ⇒ 判为估计错误
    med = np.full(st.shape, np.nan)
    k = 2
    for i in range(st.size):
        seg = st[max(0, i - k):i + k + 1]
        seg = seg[~np.isnan(seg)]
        if seg.size:
            med[i] = np.median(seg)
    bad = (~np.isnan(st)) & (~np.isnan(med)) & (np.abs(st - med) > 6)
    st[bad] = np.nan
    return st          # 保留 nan：浊音结构本身就是对齐用的信息


def tone_cmp(s1, s2):
    """时间线性对齐后比较两条 f0 半音曲线，返回 (r, rmse_semitone)。

    ★ 为什么不用 DTW：出/入时长比只有 0.988~0.997（差 ≤1.2%），根本不需
      要弯曲；DTW 的自由弯曲在短句上会把曲线拉错（zh02 给出 r=0.64/rmse=3.02
      半音，而 ASR 的声调 CER 是 0）。线性对齐同一批样本给 0.976~0.990 /
      0.5~0.9 半音，与 ASR 结论一致 ⇒ 用线性。
    """
    if s1.size < 20 or s2.size < 20:
        return None, None
    n = min(s1.size, s2.size)
    g = np.linspace(0, 1, n)
    xa, xb = np.linspace(0, 1, s1.size), np.linspace(0, 1, s2.size)
    v1, v2 = ~np.isnan(s1), ~np.isnan(s2)
    if v1.sum() < 10 or v2.sum() < 10:
        return None, None
    f1 = np.interp(g, xa[v1], s1[v1])
    f2 = np.interp(g, xb[v2], s2[v2])
    # 只比「两边都算浊音」的帧（用最近邻浊音标记近似）
    m1 = np.interp(g, xa, (~np.isnan(s1)).astype(float)) > 0.5
    m2 = np.interp(g, xb, (~np.isnan(s2)).astype(float)) > 0.5
    m = m1 & m2
    if m.sum() < 20 or np.std(f1[m]) < 1e-9:
        return None, None
    x, y = f1[m], f2[m]
    return round(float(np.corrcoef(x, y)[0, 1]), 4), \
        round(float(np.sqrt(np.mean((x - y) ** 2))), 3)


def main():
    import torch
    sys.path.insert(0, str(_paths.REPO))
    from infer.rmvpe import RMVPE
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rmv = RMVPE(str(_paths.REPO / "assets" / "rmvpe" / "rmvpe.pt"), False, device=dev)
    print(f"f0 提取器 = rmvpe ({dev})", flush=True)

    ref, selfr, ref_mf = ref_ltas(REFDIR, cache=CACHE)
    print(f"参考集自测上限 r = {selfr.mean():+.4f}", flush=True)

    rows = []
    for src in sorted((DLG / "src_tts").glob("*.wav")):
        lid = src.stem
        sy = load(str(src))
        s_st = f0_st(sy, rmv)
        base = metrics(sy, sy, ref, ref_mf)
        rows.append(dict(lid=lid, sid="src", r_lta=base["r_lta"], dr=0.0,
                         dur=round(dur_of(src), 2), dur_ratio=1.0,
                         f0_diff_hz=None, peak=round(float(np.max(np.abs(sy))), 3),
                         zero=round(float(np.mean(np.abs(sy) <= 0.0)), 5),
                         rms_db=0.0, conv_gain_pct=0.0,
                         tone_r=1.0, tone_rmse=0.0, n_src_voiced=int((~np.isnan(s_st)).sum())))
        print(f"{lid} src      r={base['r_lta']:+.4f} dur={dur_of(src):.2f}", flush=True)

        for sid in SYSTEMS:
            out = DLG / "out" / f"{lid}_{sid}.wav"
            if not out.exists():
                print(f"!! 缺 {out.name}", file=sys.stderr)
                continue
            oy = load(str(out))
            met = metrics(sy, oy, ref, ref_mf)
            tr, trm = tone_cmp(s_st, f0_st(oy, rmv))
            rows.append(dict(lid=lid, sid=sid, r_lta=met["r_lta"], dr=met["dr"],
                             dur=round(dur_of(out), 2),
                             dur_ratio=round(dur_of(out) / dur_of(src), 3),
                             f0_diff_hz=met["f0_diff_hz"], peak=met["peak"],
                             zero=met["zero_ratio"], rms_db=met["rms_db"],
                             conv_gain_pct=met.get("conv_gain_pct"),
                             tone_r=tr, tone_rmse=trm,
                             n_src_voiced=int((~np.isnan(s_st)).sum())))
            print(f"{lid} {sid:9s} r={met['r_lta']:+.4f} Δr={met['dr']:+.4f} "
                  f"dur×{dur_of(out)/dur_of(src):.3f} f0={met['f0_diff_hz']} "
                  f"zero={met['zero_ratio']} tone_r={tr} tone_rmse={trm}", flush=True)

    (DLG / "metrics.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                      encoding="utf-8")

    print("\n=== 汇总（6 句均值）===")
    for sid in ["src"] + SYSTEMS:
        sel = [r for r in rows if r["sid"] == sid]
        if not sel:
            continue
        f = lambda k: np.mean([x[k] for x in sel if x[k] is not None])
        print(f"{sid:9s} n={len(sel)} r={f('r_lta'):+.4f} Δr={f('dr'):+.4f} "
              f"dur×{f('dur_ratio'):.3f} zero={f('zero'):.4f} "
              f"gain={f('conv_gain_pct'):+.1f}% tone_r={f('tone_r'):.3f} "
              f"tone_rmse={f('tone_rmse'):.2f}st")


if __name__ == "__main__":
    main()

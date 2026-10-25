# -*- coding: utf-8 -*-
"""DS娘（zh-CN-XiaoyiNeural）RVC 模型验收评估 —— 指标计算。

★ 判据纪律（全部来自项目 MEMORY.md，不是新发明）：
  1. Δr 只能在**同信道内**比；每次对拍必须报「参考集自测上限」（留一法 LOO）。
  2. 声调用 **rmvpe + 线性对齐**，不用 pyin（八度跳变假报警）、不用 DTW（短句拉错）。
  3. 中文三层 CER：汉字 / 拼音（带调，同音字不罚）/ 声调（只比调号）。
  4. 带宽指纹：24 kHz 源在 15 kHz 以上是数字静音，输出「闷」是物理宿命。
  5. 真人通道 f0 半音 std 的 输出/源 比值 < 0.85 ⇒ 判定「R1 只会说一种腔调」命中。

参考集 = **留出的 edge 语料**（语料本身的留一法自测上限），不是训练集全体：
  训练集已被模型拟合，拿它做参考集会虚高。留出集用 a3 语料里没进训练的文件
  （holdout_x0000.wav 是语料外另合的一句同引擎样本，作为「同引擎上限」的锚）。

输出：JSON（逐条明细 + 各检查点汇总），供 build_dsx_accept_page.py 消费。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rvcx import paths as P  # noqa: E402

SR = 44100
WORK = Path(r"E:/AI-RVC/work/dsx_accept")
DATASET = Path(r"E:/AI-RVC/dataset")
XVAL = DATASET / "dsxiaoyi_xval"
CORPUS = DATASET / "dsxiaoyi_a3"
PROBE_IN = Path(r"E:/AI-RVC/probe/in")

# 换声输入分两组，两组都测，但**分开统计**（不同信道不能混算 Δr）
GROUPS = {
    "xval": dict(
        label="交叉集（CosyVoice2 zero-shot，异源同音色）",
        src_dir=XVAL,
        channel="cosyvoice2",
        note="异源引擎合成，但音色是同一个（零样本克隆 DS娘 参考音）。"
             "★ 与真人通道不同信道，Δr 不与真人组互相搬运。",
    ),
    "real": dict(
        label="真人干声通道（芳乃真人干声）",
        src_dir=PROBE_IN,
        channel="real_human",
        note="真人录音（44.1 kHz），带宽与韵律都不同于 TTS。"
             "★ 这是 R1（只会说一种腔调）的唯一判据通道。",
    ),
}


# ---------------------------------------------------------------- 音频工具
def load(p) -> tuple[np.ndarray, int]:
    y, sr = sf.read(str(p), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(1)
    return y, sr


def resample_to(y, sr, target=SR):
    if sr == target:
        return y
    import librosa
    return librosa.resample(y, orig_sr=sr, target_sr=target)


def ltas(y, sr=SR):
    """长时平均频谱（对数功率）。只取有声段，避免静音拉偏。"""
    import librosa
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    e = S.sum(0)
    S = S[:, e >= np.percentile(e, 40)]
    return librosa.power_to_db(S.mean(1) + 1e-10)


def band_ratio_db(y, sr=SR, lo=12000, hi=24000, ref_lo=1000, ref_hi=4000):
    """12–24 kHz 相对 1–4 kHz 的能量比（dB）——带宽指纹。

    ★ 24 kHz 源在 15 kHz 以上是数字静音 ⇒ 训练集本身这个值就在 −40 dB 上下。
      输出「闷」是**物理宿命**，不是训练失败；要看的是「输出与训练集是否一致」。
    """
    import librosa
    S = np.abs(librosa.stft(y, n_fft=4096, hop_length=1024)) ** 2
    fr = librosa.fft_frequencies(sr=sr, n_fft=4096)
    e = S.sum(0)
    S = S[:, e >= np.percentile(e, 40)]

    def band(a, b):
        m = (fr >= a) & (fr < b)
        return float(S[m].sum()) if m.any() else 0.0

    hi_e = band(lo, min(hi, sr / 2))
    ref_e = band(ref_lo, ref_hi)
    if hi_e <= 0 or ref_e <= 0:
        return None
    return round(10.0 * np.log10(hi_e / ref_e), 2)


# ---------------------------------------------------------------- 参考集 & Δr
def build_reference(n_ref=24, n_loo=8):
    """参考集 = edge 语料留出的一部分；自测上限 = 留一法。

    ★ 为什么不用训练集全体：模型拟合过训练集，拿它当参考集会自欺（MEMORY 判据 8）。
    ★ 为什么 LOO 上限是「同一信道内 Δr 的天花板」：r 的上限由录音信道锁死。
    """
    import librosa
    files = sorted(CORPUS.glob("*.wav"))
    # 语料文件名 xNNNN 与 xval 同号但内容不同；这里用 corpus 全体的随机子集做参考，
    # 并用**留一法**算上限：每条与「其余条的平均」比。
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(files))
    sel = [files[i] for i in idx[:n_ref]]
    loo = [files[i] for i in idx[n_ref:n_ref + n_loo]]

    ys = [resample_to(*load(f)) for f in sel]
    refs = [ltas(y) for y in ys]
    L = min(len(r) for r in refs)
    ref = np.mean([r[:L] for r in refs], axis=0)

    selfs = []
    for f in loo:
        y = resample_to(*load(f))
        selfs.append(float(np.corrcoef(ltas(y)[:L], ref)[0, 1]))
    return ref, np.array(selfs), [f.name for f in sel], [f.name for f in loo]


def delta_r(src_y, out_y, ref):
    L = min(len(ref), len(ltas(out_y)))
    r_out = float(np.corrcoef(ltas(out_y)[:L], ref[:L])[0, 1])
    r_src = float(np.corrcoef(ltas(src_y)[:L], ref[:L])[0, 1])
    return round(r_out, 4), round(r_src, 4), round(r_out - r_src, 4)


# ---------------------------------------------------------------- f0 / 声调
def f0_semitone(y, sr, rmv):
    """f0 → 半音（相对 55 Hz），丢清音帧，压八度跳变。

    ★ 用 rmvpe（与换声器同一提取器），不用 pyin：pyin 在换声输出上八度跳变，
      给出 tone_r=0.44 的假报警，而同批声调 CER=0。三个判据不可能同时为真。
    """
    import librosa
    y16 = librosa.resample(y, orig_sr=sr, target_sr=16000)
    f0 = np.asarray(rmv.infer_from_audio(y16, thred=0.03), dtype=np.float64)
    st = np.full(f0.shape, np.nan)
    ok = f0 > 0
    st[ok] = 12 * np.log2(f0[ok] / 55.0)
    med = np.full(st.shape, np.nan)
    k = 2
    for i in range(st.size):
        seg = st[max(0, i - k):i + k + 1]
        seg = seg[~np.isnan(seg)]
        if seg.size:
            med[i] = np.median(seg)
    bad = (~np.isnan(st)) & (~np.isnan(med)) & (np.abs(st - med) > 6)
    st[bad] = np.nan
    return st


def tone_cmp(s1, s2):
    """线性对齐后比较两条 f0 半音曲线 → (r, rmse 半音)。

    ★ 不用 DTW：出/入时长比差 ≤1.2%，根本不需要弯曲；
      DTW 在 <1.5 s 短句上会把曲线拉错，给出 r=0.64 的假警报。
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
    m1 = np.interp(g, xa, (~np.isnan(s1)).astype(float)) > 0.5
    m2 = np.interp(g, xb, (~np.isnan(s2)).astype(float)) > 0.5
    m = m1 & m2
    if m.sum() < 20 or np.std(f1[m]) < 1e-9:
        return None, None
    x, y = f1[m], f2[m]
    return (round(float(np.corrcoef(x, y)[0, 1]), 4),
            round(float(np.sqrt(np.mean((x - y) ** 2))), 3))


def f0_std_semitone(st):
    """f0 半音标准差（韵律起伏幅度）——R1 判据用。"""
    v = st[~np.isnan(st)]
    if v.size < 5:
        return None
    return round(float(np.std(v)), 3)


# ---------------------------------------------------------------- CER
def norm_zh(s: str) -> str:
    s = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", s)
    return s


def edit_distance(a, b):
    if not a:
        return 0.0
    dp = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(b) + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[len(b)] / len(a)


def py_seq(s: str):
    from pypinyin import lazy_pinyin, Style
    t = norm_zh(s)
    parts = lazy_pinyin(t, style=Style.TONE3, neutral_tone_with_five=True,
                        errors=lambda x: list(x))
    out = []
    for p in parts:
        p = re.sub(r"[^a-z0-9]", "", p.lower())
        if p:
            out.append(p)
    return out


def tone_seq(pys):
    return [(re.search(r"\d$", p).group(0) if re.search(r"\d$", p) else "0")
            for p in pys]


def cer_layers(ref_text, hyp_text):
    r, h = norm_zh(ref_text), norm_zh(hyp_text)
    c_han = edit_distance(r, h) if r else 0.0
    rp, hp = py_seq(ref_text), py_seq(hyp_text)
    c_py = edit_distance(rp, hp)
    c_tone = edit_distance(tone_seq(rp), tone_seq(hp))
    return (round(c_han, 4), round(c_py, 4), round(c_tone, 4),
            " ".join(hp), " ".join(rp))


# ---------------------------------------------------------------- 主流程
def collect_inputs(group):
    g = GROUPS[group]
    d = g["src_dir"]
    if group == "xval":
        return sorted(d.glob("*.wav"))
    return sorted(d.glob("real_fangno_*.wav"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", default="e20,e40,e100,e150",
                    help="逗号分隔的检查点名（对应 out/<ckpt>/ 下已有的换声输出）")
    ap.add_argument("--out", default=str(WORK / "metrics.json"))
    ap.add_argument("--asr", action="store_true", help="跑 ASR 三层 CER（需 CosyVoice env）")
    a = ap.parse_args()
    ckpts = [c.strip() for c in a.ckpts.split(",") if c.strip()]

    import torch
    sys.path.insert(0, str(P.REPO))
    from infer.rmvpe import RMVPE
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rmv = RMVPE(str(P.REPO / "assets" / "rmvpe" / "rmvpe.pt"), False, device=dev)
    print(f"f0 提取器 = rmvpe ({dev})", flush=True)

    print("构建参考集（edge 语料留出）…", flush=True)
    ref, selfr, ref_files, loo_files = build_reference()
    print(f"参考集 {len(ref_files)} 条，LOO 自测上限 r = {selfr.mean():+.4f} "
          f"(min {selfr.min():+.4f} / max {selfr.max():+.4f})", flush=True)

    # 语料本身的带宽指纹（作为「输出应该长什么样」的锚）
    corpus_band = []
    for f in sorted(CORPUS.glob("*.wav"))[:12]:
        y, sr = load(f)
        b = band_ratio_db(resample_to(y, sr))
        if b is not None:
            corpus_band.append(b)
    corpus_band_mean = round(float(np.mean(corpus_band)), 2) if corpus_band else None
    print(f"语料带宽指纹 12–24k/1–4k = {corpus_band_mean} dB（24 kHz 源的物理宿命）", flush=True)

    rows = []
    for group in GROUPS:
        srcs = collect_inputs(group)
        print(f"\n=== {group}: {GROUPS[group]['label']}（{len(srcs)} 条）===", flush=True)
        for src in srcs:
            sy, ssr = load(src)
            sy = resample_to(sy, ssr)
            s_st = f0_semitone(sy, ssr, rmv)
            base = dict(
                group=group, src=src.name,
                src_dur=round(len(sy) / SR, 3),
                src_peak=round(float(np.max(np.abs(sy))), 4),
                src_band_db=band_ratio_db(sy),
                src_f0_std_st=f0_std_semitone(s_st),
                src_n_voiced=int((~np.isnan(s_st)).sum()),
            )
            for ck in ckpts:
                outp = WORK / "out" / ck / group / src.stem + ".wav"
                outp = Path(str(outp))
                if not outp.is_file():
                    print(f"  !! 缺 {ck}/{group}/{src.name}", file=sys.stderr)
                    continue
                oy, osr = load(outp)
                oy = resample_to(oy, osr)
                r_out, r_src, dr = delta_r(sy, oy, ref)
                tr, trm = tone_cmp(s_st, f0_semitone(oy, osr, rmv))
                o_st = f0_semitone(oy, osr, rmv)
                rows.append(dict(
                    **base, ckpt=ck,
                    out_dur=round(len(oy) / SR, 3),
                    dur_ratio=round((len(oy) / SR) / (len(sy) / SR), 4),
                    r_out=r_out, r_src=r_src, dr=dr,
                    tone_r=tr, tone_rmse=trm,
                    out_f0_std_st=f0_std_semitone(o_st),
                    out_peak=round(float(np.max(np.abs(oy))), 4),
                    out_zero_ratio=round(float(np.mean(np.abs(oy) <= 0.0)), 6),
                    out_rms_db=round(20 * np.log10(max(float(np.sqrt(np.mean(oy ** 2))), 1e-12)), 2),
                    out_band_db=band_ratio_db(oy),
                    out_f0_std_ratio=(round(f0_std_semitone(o_st) / f0_std_semitone(s_st), 4)
                                      if (f0_std_semitone(o_st) and f0_std_semitone(s_st)) else None),
                ))
                print(f"  {ck:5s} {src.name:22s} Δr={dr:+.4f} tone_r={tr} "
                      f"dur×{rows[-1]['dur_ratio']:.3f} band={rows[-1]['out_band_db']}dB", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(
        rows=rows, ref_files=ref_files, loo_files=loo_files,
        loo_upper_r=round(float(selfr.mean()), 4),
        loo_min=round(float(selfr.min()), 4), loo_max=round(float(selfr.max()), 4),
        corpus_band_db=corpus_band_mean,
        groups={k: dict(label=v["label"], channel=v["channel"], note=v["note"])
                for k, v in GROUPS.items()},
    ), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""RVC 参数扫描：checkpoint × index_rate 的客观指标矩阵。

一次调用可扫多个 checkpoint 与多个 index_rate，输出：
  · r_lta   —— 换声后与目标角色参考集的 LTAS 相关（越高越像）
  · Δr      —— 相对源音色的位移（>0 = 朝目标移动）
  · mfcc_l2 —— 音色包络距离（太小 = 没换，太大 = 换过头）
  · f0diff  —— 音高中位差（Hz，越接近 0 越好）
  · peak / zero —— 削波与「精确归零」比例（DDSP 的病）

用法：
  python sweep.py --model a.pth,b.pth --index x.index --tag p2 --rates 0.0,0.5,0.75,1.0
"""
import sys as _sys
from pathlib import Path as _pth

_sys.path.insert(0, str(_pth(__file__).resolve().parents[2]))
from rvcx import paths as _paths  # noqa: E402

import argparse
import glob
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = _paths.ROOT
REPO = _paths.REPO
PY = _paths.PY
FFMPEG_DIR = _paths.FFMPEG_DIR
SR = 44100


# ---------- 指标 ----------
def load(p):
    y, sr = sf.read(p, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y


def ltas(y):
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    e = S.sum(0)
    S = S[:, e >= np.percentile(e, 40)]      # 只取有声段
    return librosa.power_to_db(S.mean(1) + 1e-10)


def ref_ltas(refdir, n=25, cache=None):
    """返回 (参考 LTAS, 本人自测相关, 参考集 MFCC 均值)。

    ★ 只靠 mfcc_l2（源 vs 出）分不清「更像源」还是「更像目标」，
      必须再算一个「输出到目标 MFCC 均值」的距离，才能判断换声强度。
    """
    if cache and Path(cache).exists():
        z = np.load(cache)
        if "mfcc" in z:
            return z["ref"], z["self"], z["mfcc"]
    files = sorted(glob.glob(f"{refdir}/*.wav"))
    random.seed(0)
    random.shuffle(files)
    sel = files[:n]
    ys = [load(f) for f in sel]
    refs = [ltas(y) for y in ys]
    L = min(len(r) for r in refs)
    ref = np.mean([r[:L] for r in refs], axis=0)
    mf = np.mean([librosa.feature.mfcc(y=y, sr=SR, n_mfcc=20, hop_length=512).mean(1)
                  for y in ys], axis=0)
    selfs = [float(np.corrcoef(ltas(load(f))[:L], ref)[0, 1]) for f in files[n:n + 8]]
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, ref=ref, self=np.array(selfs), mfcc=mf)
    return ref, np.array(selfs), mf


def metrics(srcy, outy, ref, ref_mfcc=None):
    L = min(len(ref), len(ltas(outy)))
    r_out = float(np.corrcoef(ltas(outy)[:L], ref[:L])[0, 1])
    r_src = float(np.corrcoef(ltas(srcy)[:L], ref[:L])[0, 1])

    mf1 = librosa.feature.mfcc(y=srcy, sr=SR, n_mfcc=20, hop_length=512).mean(1)
    mf2 = librosa.feature.mfcc(y=outy, sr=SR, n_mfcc=20, hop_length=512).mean(1)

    f1 = librosa.pyin(srcy, fmin=65, fmax=1100, sr=SR, frame_length=2048, hop_length=512)[0]
    f2 = librosa.pyin(outy, fmin=65, fmax=1100, sr=SR, frame_length=2048, hop_length=512)[0]
    v1, v2 = f1[~np.isnan(f1)], f2[~np.isnan(f2)]
    f0d = float(np.median(v2) - np.median(v1)) if v1.size and v2.size else None

    rep = {
        "r_lta": round(r_out, 4),
        "r_src": round(r_src, 4),
        "dr": round(r_out - r_src, 4),
        "mfcc_l2": round(float(np.linalg.norm(mf1 - mf2)), 3),
        "f0_diff_hz": None if f0d is None else round(f0d, 1),
        "peak": round(float(np.max(np.abs(outy))), 4),
        "zero_ratio": round(float(np.mean(np.abs(outy) <= 0.0)), 5),
        "rms_db": round(20 * np.log10(max(float(np.sqrt(np.mean(outy ** 2))), 1e-12))
                        - 20 * np.log10(max(float(np.sqrt(np.mean(srcy ** 2))), 1e-12)), 2),
    }
    if ref_mfcc is not None:
        d_src = float(np.linalg.norm(mf1 - ref_mfcc))
        d_out = float(np.linalg.norm(mf2 - ref_mfcc))
        rep["mfcc_to_tgt"] = round(d_out, 2)     # 输出到目标的距离
        rep["mfcc_src_to_tgt"] = round(d_src, 2)  # 源到目标的距离（基准）
        rep["conv_gain_pct"] = round((1 - d_out / max(d_src, 1e-9)) * 100, 1)  # >0 = 朝目标靠了
    return rep


def run_infer(model, index, rate, src, out, extra=None):
    """★ input 是单个文件时，CLI 把 --output 解释为「输出文件路径」并校验其后缀，
      所以这里必须传完整文件路径，不能传目录（否则目录名里的 r0.75 会被当成格式）。"""
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    if (FFMPEG_DIR / "ffmpeg.exe").exists():
        env["PATH"] = str(FFMPEG_DIR) + os.pathsep + env.get("PATH", "")
    cmd = [str(PY), "-m", "infer.cli", "--model", str(model), "--index", str(index),
           "--input", str(src), "--output", str(out), "--pitch", "0", "--f0-method", "rmvpe",
           "--index-rate", str(rate), "--protect", "0.33", "--format", "wav", "--overwrite"]
    if extra:
        cmd += extra
    r = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return r.returncode == 0, (r.stdout or "") + (r.stderr or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="逗号分隔的 pth 列表")
    ap.add_argument("--index", required=True)
    ap.add_argument("--tag", default="sweep")
    ap.add_argument("--rates", default="0.0,0.5,0.75,1.0")
    ap.add_argument("--src", default=str(ROOT / "probe" / "in"))
    ap.add_argument("--refdir", default=str(_paths.REF or (_paths.DATASET / "fangno")),
                    help="参考集目录（目标角色的训练集）；默认取 RVCX_REF")
    ap.add_argument("--outdir", default=str(ROOT / "probe" / "sweep"))
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    srcs = sorted(Path(a.src).glob("*.wav"))
    if not srcs:
        raise SystemExit(f"没有输入 wav: {a.src}")

    ref, selfr, ref_mf = ref_ltas(a.refdir, cache=str(ROOT / "cache" / "ref_ltas.npz"))
    print(f"参考集自测上限（芳乃本人）r = {selfr.mean():+.4f}")
    print(f"测试片段 {len(srcs)} 条: {[p.name for p in srcs]}\n")

    models = [Path(m.strip()) for m in a.model.split(",") if m.strip()]
    rates = [float(r) for r in a.rates.split(",") if r.strip()]
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for m in models:
        if not m.exists():
            print(f"!! 缺模型 {m}", file=sys.stderr)
            continue
        for r in rates:
            od = outdir / f"{a.tag}_{m.stem}_r{r}"
            od.mkdir(parents=True, exist_ok=True)
            per = []
            for src in srcs:
                outf = od / f"{src.stem}.wav"
                ok, log = run_infer(m, a.index, r, str(src), str(outf))
                if not ok or not outf.exists():
                    print(f"!! 失败 {m.stem} r={r} {src.name}\n{log[-400:]}", file=sys.stderr)
                    continue
                per.append(metrics(load(str(src)), load(str(outf)), ref, ref_mf))
            if not per:
                continue
            # 多片段取均值；f0 差值单独处理 None
            met = {}
            for k in per[0]:
                vals = [p[k] for p in per if p.get(k) is not None]
                met[k] = round(float(np.mean(vals)), 4) if vals else None
            met.update({"model": m.stem, "rate": r, "n_clips": len(per)})
            rows.append(met)
            print(f"{m.stem:28s} r={r:<5} r_lta={met['r_lta']:+.4f} "
                  f"Δr={met['dr']:+.4f} mfcc={met['mfcc_l2']:>7.2f} "
                  f"f0={met['f0_diff_hz']} peak={met['peak']} zero={met['zero_ratio']}"
                  f" conv={met.get('conv_gain_pct')}%  (n={len(per)})", flush=True)

    # 排序：先看是否越过源（dr>0），再按 r_lta 排
    rows.sort(key=lambda x: (-x["dr"], -x["r_lta"]))
    print("\n=== 按 Δr 排序（前 8）===")
    for m in rows[:8]:
        print(f"{m['model']:28s} r={m['rate']:<5} r_lta={m['r_lta']:+.4f} Δr={m['dr']:+.4f}")

    if a.json:
        Path(a.json).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n-> {a.json}")


if __name__ == "__main__":
    main()

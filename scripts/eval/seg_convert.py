# -*- coding: utf-8 -*-
"""分段换声渲染：把长音频切成带重叠的段，逐段独立进程推理再拼接。

★ 为什么需要它（实测）：
  一次性渲染 187 s 全曲时，RVC 的 CUDA Graph 私有池会累积到 8.9 GB，
  在 8 GB 卡上与训练进程并存时必然 CUDA OOM：
      "8.90 GiB allocated in private pools (e.g., CUDA Graphs)"
  而训练要跑 1.5 小时，不能打断 ⇒ 改用分段渲染：
  每段一个独立进程 ⇒ 显存恒定、进程退出即释放。

★ 采样率必须显式对齐（实测，别想当然）：
  - RVC 模型输出采样率 = checkpoint 的 config[-1] = **48000**
    （输入会被降到 16 kHz 喂 HuBERT，输出按 config.data.sampling_rate 写盘）
  - 但 `data.sampling_rate` 不在 config.json 顶层，只在 train/data 里
  - 如果输入是 44.1 kHz 而输出按样本索引直接写入 ⇒ 时间轴被压缩/拉伸 8.84 %
  ⇒ 统一传 `--resample-sr <输入sr>`，让 RVC 自己重采样，长度天然对齐。

★ 每段末尾固定少 20 ms（实测 3.000 s → 2.980 s，882 帧 @44.1 k）：
  这是 RVC pipeline 的固有截断，**缺口在末尾**（有声段包络互相关 lag=-4.4 ms）
  ⇒ 拼接用「前向推进」：下一段起点 = 上一段实际输出终点 − overlap，
    缺口由下一段的头部覆盖，内容不丢、时间轴连续。

★ 拼接不会听起来是「拼接的」：
  · 相邻段真实重叠 ov 秒，重叠区做**权重归一**淡化（权重和恒为 1 ⇒ 幅度不损失）
  · 段边界选在低能量处（RMS 谷底），避免切在音节中间
  · 权重由「上一段实际长度」推出，不假设等长

用法：
  python seg_convert.py --model <权重> --input <源.wav> --output <输出.wav> \
      --seg 30 --overlap 2 --index-rate 0
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rvcx import paths as P  # noqa: E402

FFMPEG_DIR = Path(r"D:/AI-Song/01_人声分离_MSST-GUI/env")


def env_with_ffmpeg():
    env = dict(os.environ)
    env["PATH"] = str(FFMPEG_DIR) + os.pathsep + env.get("PATH", "")
    return env


def find_cuts(y, sr, seg_sec, overlap_sec):
    """在 RMS 谷底处找段边界：候选点附近能量最低处。返回样本索引列表。"""
    hop = int(sr * 0.05)
    n = len(y) // hop
    if n < 2:
        return []
    rms = np.sqrt((y[:n * hop].reshape(n, hop).astype(np.float64) ** 2).mean(axis=1))
    step = int(seg_sec / 0.05)
    win = int(overlap_sec / 0.05)
    cuts = []
    i = step
    while i < n - win:
        lo, hi = max(0, i - win), min(n, i + win)
        best = lo + int(np.argmin(rms[lo:hi]))
        cuts.append(best * hop)
        i = best + step
    return cuts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--index", default="")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seg", type=float, default=30.0, help="目标段长（秒）")
    ap.add_argument("--overlap", type=float, default=2.0, help="重叠（秒）")
    ap.add_argument("--index-rate", type=float, default=0.0)
    ap.add_argument("--protect", type=float, default=0.33)
    ap.add_argument("--pitch", type=int, default=0)
    ap.add_argument("--f0-method", default="rmvpe")
    ap.add_argument("--workdir", default="")
    ap.add_argument("--keep-seg", action="store_true")
    a = ap.parse_args()

    y, sr = sf.read(a.input, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(1)
    total = len(y)
    ov = int(a.overlap * sr)

    work = Path(a.workdir) if a.workdir else Path(a.output).parent / "_segwork"
    work.mkdir(parents=True, exist_ok=True)

    cuts = find_cuts(y, sr, a.seg, a.overlap)
    # 段起点：第 i 段的起点 = 第 i-1 个切点 − overlap（首段为 0）
    # 段终点：前向推进，由上一段的实际输出长度决定
    starts = [0]
    for c in cuts:
        starts.append(max(0, c - ov))
    print(f"总长 {total/sr:.2f}s → {len(starts)} 段（切点 {len(cuts)} 个）", flush=True)

    env = env_with_ffmpeg()
    out = np.zeros(total, dtype=np.float64)
    wsum = np.zeros(total, dtype=np.float64)
    seg_log = []
    t0 = time.time()

    for i, s in enumerate(starts):
        # 本段输入区间：从 s 到「下一切点」附近（末段到曲末）
        e = starts[i + 1] + ov if i + 1 < len(starts) else total
        e = min(e, total)
        if e - s < int(0.2 * sr):
            continue
        seg_in = work / f"seg{i:03d}_in.wav"
        seg_out = work / f"seg{i:03d}_out.wav"
        sf.write(seg_in, y[s:e], sr, subtype="PCM_16")
        cmd = [str(P.PY), "-m", "infer.cli", "--model", a.model,
               "--input", str(seg_in), "--output", str(seg_out),
               "--f0-method", a.f0_method,
               "--index-rate", str(a.index_rate),
               "--protect", str(a.protect), "--pitch", str(a.pitch),
               "--resample-sr", str(sr),          # ★ 让 RVC 直接输出源采样率
               "--format", "wav", "--overwrite"]
        if a.index and a.index_rate > 0:
            cmd += ["--index", a.index]
        r = subprocess.run(cmd, cwd=str(P.REPO), env=env,
                           capture_output=True, text=True)
        if r.returncode != 0 or not seg_out.exists():
            print(f"  段 {i} 失败:\n{(r.stderr or '')[-800:]}", flush=True)
            raise SystemExit(f"分段渲染失败于段 {i}")
        zo, so = sf.read(seg_out, dtype="float32", always_2d=False)
        if zo.ndim > 1:
            zo = zo.mean(1)
        if so != sr:                       # 兜底：万一 --resample-sr 没生效
            import librosa
            zo = librosa.resample(zo, orig_sr=so, target_sr=sr)
        want = e - s
        raw_len = len(zo)
        if raw_len > want:
            zo = zo[:want]
        elif raw_len < want:
            zo = np.concatenate([zo, np.zeros(want - raw_len, dtype=np.float32)])

        # ---- 权重：头部 overlap 升、尾部 overlap 降（线性，权重和恒为 1）----
        w = np.ones(want, dtype=np.float64)
        if i > 0 and ov > 0:
            head = min(ov, want)
            w[:head] = np.linspace(0.0, 1.0, head, dtype=np.float64)
        if i + 1 < len(starts) and ov > 0:
            tail = min(ov, want)
            w[-tail:] = np.linspace(1.0, 0.0, tail, dtype=np.float64)

        out[s:s + want] += w * zo
        wsum[s:s + want] += w
        seg_log.append({"i": i, "start": round(s / sr, 2), "end": round(e / sr, 2),
                        "raw_len_sec": round(raw_len / sr, 3),
                        "trunc_ms": round((raw_len - want) / sr * 1000, 1)})
        print(f"  段 {i:02d} [{s/sr:7.2f}–{e/sr:7.2f}s] ok "
              f"(截 {seg_log[-1]['trunc_ms']:.0f}ms, {time.time()-t0:.0f}s 累计)", flush=True)
        if not a.keep_seg:
            for f in (seg_in, seg_out):
                try:
                    f.unlink()
                except OSError:
                    pass

    # ---- 权重归一：重叠区权重和 ≈ 1 ⇒ 幅度不损失 ----
    cover = wsum > 1e-6
    out[cover] /= wsum[cover]
    miss = (~cover).sum()
    if miss:
        print(f"  ⚠ {miss} 个样本未被覆盖（保持静音）", flush=True)

    out = out.astype(np.float32)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    sf.write(a.output, out, sr, subtype="PCM_16")
    rep = {"input": a.input, "output": a.output, "sr": sr,
           "total_sec": round(total / sr, 2), "segments": len(seg_log),
           "seg_sec": a.seg, "overlap_sec": a.overlap,
           "index_rate": a.index_rate, "model": a.model,
           "elapsed_sec": round(time.time() - t0, 1),
           "peak": round(float(np.abs(out).max()), 4),
           "rms_dbfs": round(float(20 * np.log10(max(float(np.sqrt((out ** 2).mean())), 1e-12))), 2),
           "uncovered_samples": int(miss),
           "seg_log": seg_log}
    rp = Path(a.output).with_suffix(".segreport.json")
    rp.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "seg_log"},
                     ensure_ascii=False, indent=2))
    print(f"分段报告 -> {rp}")
    if not a.keep_seg:
        try:
            work.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    main()

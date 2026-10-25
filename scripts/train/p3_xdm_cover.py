# -*- coding: utf-8 -*-
"""P3《下等马》DS娘换声成曲管线。

背景：DS娘（晓伊A3）的 RVC 权重 ``dsxiaoyi_a3_v1_e20`` 已训到 e20 崩于显存，
但 e20 检查点可用。本脚本把「泠鸢原唱《下等马》」整轨换声成 DS娘音色并混伴奏。

★ 纪律 D1：**不拼接**多系统输出。每一步的产物都独立成文件、独立可听，
  并在 manifest 里标注「它是什么 / 哪个模型 / 什么参数」。

用法：
  python p3_xdm_cover.py --model <weights.pth> --index <index> [--outdir DIR]
                         [--tag e20] [--index-rate 0.75] [--protect 0.33]
                         [--pitch 0] [--f0-method rmvpe]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ★ 本机唯一完整版 ffmpeg 在 MSST-GUI 的 env 里（JiJiDown 那个阉割版读不了 wav）
FFMPEG_DIR = Path(r"D:\AI-Song\01_人声分离_MSST-GUI\env")
FFMPEG = FFMPEG_DIR / "ffmpeg.exe"
FFPROBE = FFMPEG_DIR / "ffprobe.exe"

# 泠鸢原唱《下等马》的既有管线产物（MSST 分离结果）
XDM_ROOT = Path(r"D:\AI-Song\_pipeline_xiadengma")
KARA = XDM_ROOT / "01_sep" / "step2_kara" / "vocals"
SRC_VOCAL = KARA / "Vocals.wav"        # 泠鸢原唱主唱（= 02_conv/in_lead/lead.wav，corr=1.0）
ACCOMP = KARA / "instrumental.wav"      # 伴奏


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("  $", " ".join(str(c) for c in cmd[:6]), "..." if len(cmd) > 6 else "")
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", **kw)
    if p.returncode != 0:
        print(p.stdout[-2000:])
        print(p.stderr[-2000:])
        raise SystemExit(f"命令失败 rc={p.returncode}")
    return p


def ensure_env():
    import os
    os.environ["PATH"] = str(FFMPEG_DIR) + os.pathsep + os.environ.get("PATH", "")


def probe(path: Path) -> dict:
    p = run([str(FFPROBE), "-v", "error", "-show_entries",
             "stream=sample_rate,channels,duration:format=duration", "-of", "json", str(path)])
    d = json.loads(p.stdout)
    st = d["streams"][0]
    return {
        "sr": int(st["sample_rate"]),
        "channels": int(st["channels"]),
        "duration": float(st.get("duration") or d["format"]["duration"]),
    }


def peak_dbfs(path: Path) -> tuple[float, float]:
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(1)
    pk = float(np.abs(y).max()) if y.size else 0.0
    rms = float(np.sqrt(np.mean(y ** 2))) if y.size else 0.0
    to_db = lambda x: 20 * np.log10(max(x, 1e-12))
    return to_db(pk), to_db(rms)


def rvc_convert(model: Path, index: Path, src: Path, dst: Path, args) -> dict:
    """跑 RVC 离线推理。返回耗时与产物信息。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    py = r"E:\AI-RVC\venv\Scripts\python.exe"
    repo = Path(r"E:\AI-RVC\repo")
    cmd = [py, "-m", "infer.cli", "--model", str(model), "--index", str(index),
           "--input", str(src), "--output", str(dst),
           "--f0-method", args.f0_method, "--index-rate", str(args.index_rate),
           "--protect", str(args.protect), "--pitch", str(args.pitch),
           "--format", "wav", "--overwrite"]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    dt = time.time() - t0
    if p.returncode != 0:
        print(p.stdout[-3000:]); print(p.stderr[-3000:])
        raise SystemExit(f"换声失败 rc={p.returncode}")
    return {"seconds": round(dt, 1), "stdout_tail": p.stdout.strip().splitlines()[-3:] if p.stdout.strip() else []}


def mix(vocal: Path, accomp: Path, dst: Path, vocal_gain_db: float = 0.0,
        accomp_gain_db: float = 0.0) -> dict:
    """人声 + 伴奏混合（ffmpeg amix，带响度对齐与限峰）。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    vg = 10 ** (vocal_gain_db / 20.0)
    ag = 10 ** (accomp_gain_db / 20.0)
    # amix normalize=0（不做自动衰减，否则人声被压），加 alimiter 限峰
    cmd = [str(FFMPEG), "-y", "-v", "error",
           "-i", str(vocal), "-i", str(accomp),
           "-filter_complex",
           f"[0:a]volume={vg:.4f}[v];[1:a]volume={ag:.4f}[a];"
           f"[v][a]amix=inputs=2:duration=longest:normalize=0[m];"
           f"[m]alimiter=limit=0.95:level=disabled[out]",
           "-map", "[out]", "-ar", "44100", "-ac", "2", str(dst)]
    run(cmd)
    return {"vocal_gain_db": vocal_gain_db, "accomp_gain_db": accomp_gain_db}


def make_delivery(dir_in: Path, dir_out: Path, title: str, notes: str) -> None:
    """把 wav 转成 mp3（320k）供普通播放器使用。"""
    dir_out.mkdir(parents=True, exist_ok=True)
    for w in sorted(dir_in.glob("*.wav")):
        out = dir_out / (w.stem + ".mp3")
        run([str(FFMPEG), "-y", "-v", "error", "-i", str(w),
             "-c:a", "libmp3lame", "-b:a", "320k", str(out)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"E:\AI-RVC\repo\assets\weights\dsxiaoyi_a3_v1_e20_s2640.pth")
    ap.add_argument("--index", default=r"E:\AI-RVC\repo\assets\indices\dsxiaoyi_a3_v1_added_IVF1929_Flat_nprobe_1_dsxiaoyi_a3_v1_v2.index")
    ap.add_argument("--outdir", default=r"E:\AI-RVC\work\dsx_xdm")
    ap.add_argument("--tag", default="e20")
    ap.add_argument("--index-rate", type=float, default=0.75)
    ap.add_argument("--protect", type=float, default=0.33)
    ap.add_argument("--pitch", type=int, default=0)
    ap.add_argument("--f0-method", default="rmvpe")
    ap.add_argument("--vocal-gain-db", type=float, default=0.0)
    ap.add_argument("--accomp-gain-db", type=float, default=-6.0)
    ap.add_argument("--skip-mix", action="store_true")
    args = ap.parse_args()

    ensure_env()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "title": "《下等马》DS娘（晓伊A3）换声成曲",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": {
            "file": str(args.model),
            "weight_tag": args.tag,
            "engine": "RVC-WebUI 2.3.260718 (infer.cli)",
            "format": "v2",
            "sample_rate": "48k",
            "if_f0": 1,
        },
        "params": {
            "f0_method": args.f0_method,
            "index_rate": args.index_rate,
            "protect": args.protect,
            "pitch": args.pitch,
            "resample_sr": 0,
            "rms_mix_rate": 1.0,
        },
        "source": {
            "what": "泠鸢原唱《下等马》分离主唱（MSST→kara vocals）",
            "file": str(SRC_VOCAL),
            "note": "同一文件与 02_conv/in_lead/lead.wav 相关性 corr=1.0（同一轨）",
        },
        "steps": [],
    }

    print("[1/5] 检查输入")
    for p in (SRC_VOCAL, ACCOMP, Path(args.model), Path(args.index)):
        if not p.exists():
            raise SystemExit(f"缺文件: {p}")
    src_info = probe(SRC_VOCAL)
    acc_info = probe(ACCOMP)
    print(f"  原唱 {src_info}")
    print(f"  伴奏 {acc_info}")

    print("[2/5] RVC 换声（DS娘 e20 权重）")
    conv = out / f"lead_dsx_{args.tag}.wav"
    r = rvc_convert(Path(args.model), Path(args.index), SRC_VOCAL, conv, args)
    conv_info = probe(conv)
    pk, rms = peak_dbfs(conv)
    print(f"  换声 {conv_info} 用时 {r['seconds']}s peak={pk:.2f}dBFS rms={rms:.2f}dBFS")
    manifest["steps"].append({
        "step": "rvc_convert",
        "output": str(conv),
        "what": "泠鸢原唱 → DS娘音色 的整轨换声（未混伴奏）",
        "model": args.model,
        "params": manifest["params"],
        "probe": conv_info,
        "peak_dbfs": round(pk, 2),
        "rms_dbfs": round(rms, 2),
        "seconds": r["seconds"],
    })

    if not args.skip_mix:
        print("[3/5] 与伴奏混合")
        mixed = out / f"下等马_DS娘_{args.tag}_mix.wav"
        mix(conv, ACCOMP, mixed, args.vocal_gain_db, args.accomp_gain_db)
        minfo = probe(mixed)
        mpk, mrms = peak_dbfs(mixed)
        print(f"  成品 {minfo} peak={mpk:.2f}dBFS")
        manifest["steps"].append({
            "step": "mix",
            "output": str(mixed),
            "what": "换声主唱 + 原伴奏（amix normalize=0 + alimiter 0.95）",
            "vocal_gain_db": args.vocal_gain_db,
            "accomp_gain_db": args.accomp_gain_db,
            "probe": minfo,
            "peak_dbfs": round(mpk, 2),
            "rms_dbfs": round(mrms, 2),
        })

    print("[4/5] 输出 mp3 交付件")
    mp3dir = out / "mp3"
    make_delivery(out, mp3dir, manifest["title"], "")
    for s in manifest["steps"]:
        w = Path(s["output"])
        if w.exists():
            s["mp3"] = str(mp3dir / (w.stem + ".mp3"))

    print("[5/5] 写 manifest")
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2)[:1500])


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""《下等马》DS娘版全流程：换声（关索引）→ 诊断 → 伴奏回混 → 母带 → mp3。

★ 为什么换声默认 index_rate=0（本轮实测的关键修复）：
  added 索引按语料音域建立。DS娘语料（CosyVoice TTS）实测
  >600 Hz 占 0.0012%、最高 657 Hz，而原唱（泠鸢）最高稳定 831 Hz。
  ⇒ 索引里没有 700 Hz+ 的帧 ⇒ 高音查询帧被检索到低八度邻居 ⇒ 高音降八度/掉音。
  实测（115.7 s 段，源 783.6 Hz）：index_rate=0.75 → 410.5 Hz；=0 → 769.7 Hz。
  且 index_rate=0 音色保真不降反升（LTAS 0.7356 vs 0.7283）。

用法：
  python p4_xdm_fixmix.py --model <权重> --index <索引> --tag e150
  python p4_xdm_fixmix.py --model <权重> --tag e150 --skip-conv   # 只重混
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

PIPE = Path(r"D:/AI-Song/_pipeline_xiadengma")
VOCALS = PIPE / "01_sep" / "step2_kara" / "vocals" / "Vocals.wav"
INST = PIPE / "01_sep" / "step2_kara" / "vocals" / "instrumental.wav"
FFMPEG_DIR = Path(r"D:/AI-Song/01_人声分离_MSST-GUI/env")
SKILL_MIX = Path.home() / ".workbuddy" / "skills" / "ai-song-cover-pipeline" / "scripts" / "mix_master.py"
COSY_PY = Path(r"D:/AI-Voice/01_情感配音_CosyVoice/env/Scripts/python.exe")


def sh(cmd, **kw):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    env = dict(os.environ)
    env["PATH"] = str(FFMPEG_DIR) + os.pathsep + env.get("PATH", "")
    return subprocess.run([str(c) for c in cmd], env=env, **kw)


def audio_stats(p):
    y, sr = sf.read(str(p), dtype="float32", always_2d=True)
    rms = 20 * np.log10(max(float(np.sqrt((y ** 2).mean())), 1e-12))
    return {
        "file": Path(p).name,
        "dur": round(len(y) / sr, 2), "sr": sr, "ch": y.shape[1],
        "peak": round(float(np.abs(y).max()), 4),
        "rms_dbfs": round(rms, 2),
        "silent_pct": round(float((np.abs(y) < 1e-9).mean() * 100), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--index", default="", help="可空（index_rate=0 时不需要索引）")
    ap.add_argument("--tag", default="fix")
    ap.add_argument("--outdir", default="")
    ap.add_argument("--index-rate", type=float, default=0.0)
    ap.add_argument("--skip-conv", action="store_true")
    ap.add_argument("--skip-mix", action="store_true")
    a = ap.parse_args()

    outdir = Path(a.outdir) if a.outdir else Path(f"E:/AI-RVC/work/dsx_xdm_{a.tag}")
    outdir.mkdir(parents=True, exist_ok=True)
    vocal = outdir / f"lead_dsx_{a.tag}.wav"
    v44 = outdir / f"vocal_{a.tag}_44k.wav"
    sil = outdir / "silence_44k.wav"
    mixwav = outdir / f"下等马_DS娘_{a.tag}_mix_master.wav"
    mixmp3 = outdir / f"下等马_DS娘_{a.tag}_mix_master.mp3"
    report = outdir / f"pipeline_report_{a.tag}.json"

    rep = {"tag": a.tag, "model": a.model, "index": a.index,
           "index_rate": a.index_rate, "outdir": str(outdir)}

    # ---------- 1. 换声 ----------
    if not a.skip_conv:
        cmd = [P.PY, "-m", "infer.cli", "--model", a.model,
               "--input", str(VOCALS), "--output", str(vocal),
               "--f0-method", "rmvpe", "--index-rate", str(a.index_rate),
               "--protect", "0.33", "--pitch", "0", "--format", "wav", "--overwrite"]
        if a.index and a.index_rate > 0:
            cmd += ["--index", a.index]
        t0 = time.time()
        r = sh(cmd, cwd=str(P.REPO), capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:]); print(r.stderr[-2000:])
            raise SystemExit("换声失败")
        rep["conv_seconds"] = round(time.time() - t0, 1)
        print(r.stdout.strip()[-200:])
    rep["vocal"] = audio_stats(vocal)

    # ---------- 2. 诊断（音准 + 缺陷扫描 + 交叉验证）----------
    diag = {}
    for script, extra in [
        ("diag_pitch_dsx.py", []),
        ("scan_defects.py", []),
    ]:
        j = outdir / f"{Path(script).stem}.json"
        cmd = [P.PY, str(ROOT / "scripts" / "diag" / script),
               "--src", str(VOCALS), "--conv", str(vocal), "--json", str(j)]
        r = sh(cmd, capture_output=True, text=True)
        if r.returncode == 0 and j.exists():
            diag[script] = json.loads(j.read_text(encoding="utf-8"))
        else:
            diag[script] = {"error": (r.stderr or "")[-300:]}
    rep["diagnostics"] = diag

    # ---------- 3. 伴奏回混 ----------
    if not a.skip_mix:
        sh(["ffmpeg", "-y", "-v", "error", "-i", str(vocal),
            "-ar", "44100", "-ac", "1", "-c:a", "pcm_s16le", str(v44)],
           capture_output=True, text=True)
        sh(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
            "-i", "anullsrc=r=44100:cl=stereo", "-t", "187.51",
            "-c:a", "pcm_s16le", str(sil)], capture_output=True, text=True)
        mrep = outdir / f"mix_report_{a.tag}.json"
        r = sh([COSY_PY, str(SKILL_MIX),
                "--inst", str(INST), "--lead", str(v44), "--harm", str(sil),
                "--ref-lead", str(VOCALS), "--out", str(mixwav),
                "--report", str(mrep), "--preset", "match"],
               capture_output=True, text=True)
        if r.returncode == 0:
            rep["mix"] = json.loads(mrep.read_text(encoding="utf-8"))
        else:
            print(r.stdout[-1500:]); print(r.stderr[-1500:])
            raise SystemExit("混音失败")
        sh(["ffmpeg", "-y", "-v", "error", "-i", str(mixwav),
            "-c:a", "libmp3lame", "-b:a", "320k", str(mixmp3)],
           capture_output=True, text=True)
        rep["mix_wav"] = audio_stats(mixwav)

    report.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "diagnostics"},
                     ensure_ascii=False, indent=2))
    print(f"\n报告 -> {report}")


if __name__ == "__main__":
    main()

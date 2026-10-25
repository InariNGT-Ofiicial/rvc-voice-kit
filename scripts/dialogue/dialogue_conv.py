# -*- coding: utf-8 -*-
"""台词换声：两条源通道 × 两个检查点。

通道 T = TTS（GPT-SoVITS kurumi 日语合成，与真人同文本）
通道 H = 真人（茉子 / 日立真子 原声，游戏内语音）
模型 = 芳乃 RVC e120 / e200，rmvpe + index_rate 0.75 + pitch 0（不变调）
"""
import sys as _sys
from pathlib import Path as _pth

_sys.path.insert(0, str(_pth(__file__).resolve().parents[2]))
from rvcx import paths as _paths  # noqa: E402

import os
import subprocess
import sys
from pathlib import Path

REPO = _paths.REPO
PY = _paths.PY
FFMPEG_DIR = _paths.FFMPEG_DIR
DLG = _paths.WORK / "dialogue"
OUT = DLG / "out"

IDXNAME = "added_IVF1419_Flat_nprobe_1_fangno_p2_v2.index"
MODELS = {
    "e120": (_paths.MODELS / "芳乃_RVC_步数120" / "model.pth",
             _paths.MODELS / "芳乃_RVC_步数120" / IDXNAME),
    "e200": (_paths.MODELS / "芳乃_RVC_步数200" / "model.pth",
             _paths.MODELS / "芳乃_RVC_步数200" / IDXNAME),
}
ARMS = {"T": "src_tts", "H": "src_real"}


def env():
    e = {**os.environ, "PYTHONPATH": str(REPO)}
    if (FFMPEG_DIR / "ffmpeg.exe").exists():
        e["PATH"] = str(FFMPEG_DIR) + os.pathsep + e.get("PATH", "")
    return e


def conv(model: Path, index: Path, src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(PY), "-m", "infer.cli",
           "--model", str(model), "--index", str(index),
           "--input", str(src), "--output", str(dst),
           "--pitch", "0", "--f0-method", "rmvpe",
           "--index-rate", "0.75", "--protect", "0.33",
           "--format", "wav", "--overwrite"]
    r = subprocess.run(cmd, cwd=str(REPO), env=env(), capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    ok = r.returncode == 0 and dst.exists()
    if not ok:
        print(f"  !! {src.name} -> {model.stem}: rc={r.returncode}", flush=True)
        print((r.stdout or "")[-500:], (r.stderr or "")[-500:], flush=True)
    return ok


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    total, ok_n = 0, 0
    for arm, d in ARMS.items():
        srcs = sorted((DLG / d).glob("*.wav"))
        print(f"[{arm}] {d}: {len(srcs)} 条", flush=True)
        for mname, (mp, ip) in MODELS.items():
            if only and only != mname:
                continue
            for s in srcs:
                total += 1
                dst = OUT / f"{s.stem}_{arm}_{mname}.wav"
                if dst.exists() and dst.stat().st_size > 1000:
                    ok_n += 1
                    continue
                if conv(mp, ip, s, dst):
                    ok_n += 1
                    print(f"  ok {s.stem} {arm} {mname} "
                          f"{dst.stat().st_size/1048576:.2f} MB", flush=True)
    print(f"完成 {ok_n}/{total}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""中文台词换声：6 条中文源 × 4 个系统。

系统（sid 直接进文件名，页面与指标都靠它对齐）：
  e120      芳乃 e120 · index_rate 0.75   ← 与日语批主力配置一致
  e200      芳乃 e200 · index_rate 0.75
  e120_ir0  芳乃 e120 · index_rate 0.0    ← 关掉索引检索
  e120_ir050 芳乃 e120 · index_rate 0.50

★ 为什么要扫 index_rate：索引库（added_*.index）里全是**日语**帧向量。
  中文音素在库里找不到近邻时，检索会把特征拉向日语分布 ⇒ 这是「中文会不会
  被日语音系带跑」最可能的机制。所以中文这批必须单独验证 ir 的影响。
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
DLG = _paths.WORK / "dialogue_zh"
SRC = DLG / "src_tts"
OUT = DLG / "out"

M120 = _paths.MODELS / "芳乃_RVC_步数120"
M200 = _paths.MODELS / "芳乃_RVC_步数200"
IDX = M120 / "added_IVF1419_Flat_nprobe_1_fangno_p2_v2.index"

SYSTEMS = {
    "e120":       dict(model=M120 / "model.pth", index=IDX, ir=0.75, f0="rmvpe"),
    "e200":       dict(model=M200 / "model.pth", index=IDX, ir=0.75, f0="rmvpe"),
    "e120_ir0":   dict(model=M120 / "model.pth", index=IDX, ir=0.0,  f0="rmvpe"),
    "e120_ir050": dict(model=M120 / "model.pth", index=IDX, ir=0.50, f0="rmvpe"),
}


def env():
    e = {**os.environ, "PYTHONPATH": str(REPO)}
    if (FFMPEG_DIR / "ffmpeg.exe").exists():
        e["PATH"] = str(FFMPEG_DIR) + os.pathsep + e.get("PATH", "")
    return e


def conv(model, index, ir, f0, src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(PY), "-m", "infer.cli",
           "--model", str(model), "--index", str(index),
           "--input", str(src), "--output", str(dst),
           "--pitch", "0", "--f0-method", f0,
           "--index-rate", f"{ir}", "--protect", "0.33",
           "--format", "wav", "--overwrite"]
    r = subprocess.run(cmd, cwd=str(REPO), env=env(), capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    ok = r.returncode == 0 and dst.exists() and dst.stat().st_size > 1000
    if not ok:
        print(f"  !! {src.name} -> {dst.name}: rc={r.returncode}", flush=True)
        print((r.stdout or "")[-600:], (r.stderr or "")[-600:], flush=True)
    return ok


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    srcs = sorted(SRC.glob("*.wav"))
    total = ok_n = 0
    for sid, cfg in SYSTEMS.items():
        if only and only != sid:
            continue
        print(f"[{sid}] ir={cfg['ir']} model={cfg['model'].parent.name}", flush=True)
        for s in srcs:
            total += 1
            dst = OUT / f"{s.stem}_{sid}.wav"
            if dst.exists() and dst.stat().st_size > 1000:
                ok_n += 1
                continue
            if conv(cfg["model"], cfg["index"], cfg["ir"], cfg["f0"], s, dst):
                ok_n += 1
                print(f"  ok {s.stem} -> {dst.name} "
                      f"{dst.stat().st_size/1048576:.2f} MB", flush=True)
    print(f"完成 {ok_n}/{total}")


if __name__ == "__main__":
    main()

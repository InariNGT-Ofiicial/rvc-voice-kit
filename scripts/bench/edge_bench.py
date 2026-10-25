# -*- coding: utf-8 -*-
"""Edge TTS 延迟基线（冷启动 = 进程起步，热 = 同进程第二次）。

用法：
  D:/AI-Voice/01_情感配音_CosyVoice/env/Scripts/python.exe edge_bench.py \
      --text "..." --out x.wav --runs 3
"""
import argparse
import asyncio
import json
import os
import sys
import time

T0 = time.time()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--voice", default="zh-CN-XiaoyiNeural")
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import edge_tts

    t_import = time.time()
    rows = []
    for i in range(a.runs):
        t0 = time.time()
        p = os.path.join(a.out, f"edge_{i}.wav")
        comm = edge_tts.Communicate(a.text, a.voice)
        await comm.save(p)
        ms = int((time.time() - t0) * 1000)
        rows.append({"run": i, "synth_ms": ms, "bytes": os.path.getsize(p),
                     "path": p.replace("\\", "/"), "warm": i > 0})
        print(f"[run {i}] {ms}ms", file=sys.stderr)

    print(json.dumps({
        "python_boot_ms": int((t_import - T0) * 1000),
        "voice": a.voice,
        "runs": rows,
    }, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())

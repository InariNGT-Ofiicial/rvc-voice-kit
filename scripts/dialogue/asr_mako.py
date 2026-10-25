# -*- coding: utf-8 -*-
"""用本地 faster-whisper large-v3 转写茉子真人录音，取台词原文。

目的：让「TTS 通道」与「真人通道」能喂**同一文本**，否则两条通道没法做可控对撞。
"""
import sys as _sys
from pathlib import Path as _pth

_sys.path.insert(0, str(_pth(__file__).resolve().parents[2]))
from rvcx import paths as _paths  # noqa: E402

import glob
import json
import os
import sys
from pathlib import Path

SRC = _paths.WORK / "dialogue" / "src_real" / "mako"
OUT = _paths.WORK / "dialogue" / "mako_asr.json"

# 模型在本地缓存里，禁掉联网避免超时
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from faster_whisper import WhisperModel  # noqa: E402


def main():
    files = sorted(SRC.glob("*.wav"))
    if not files:
        raise SystemExit("找不到茉子录音")
    print(f"待转写 {len(files)} 个文件", flush=True)
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")
    rows = []
    for f in files:
        segs, info = model.transcribe(str(f), language="ja", beam_size=5, vad_filter=True)
        text = "".join(s.text for s in segs).strip()
        dur = info.duration
        rows.append(dict(file=f.name, dur=round(dur, 2), text=text,
                         lang=info.language, prob=round(info.language_probability, 3)))
        print(f"{f.name:10s} {dur:5.2f}s  {text}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""CosyVoice2 延迟基线：一次加载模型，多次合成，逐次计时。

用法（必须用 CosyVoice 的 env）：
  D:/AI-Voice/01_情感配音_CosyVoice/env/Scripts/python.exe tts_bench.py \
      --text "今天天气不错，我们出去走走吧。" --runs 3 --out D:/.../audio/tts
输出最后一行是 JSON，便于机器解析。
"""
import argparse
import json
import os
import sys
import time

ROOT = r"D:\AI-Voice\01_情感配音_CosyVoice\CosyVoice_源码"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "third_party", "Matcha-TTS"))

MODEL = os.path.join(ROOT, "pretrained_models", "CosyVoice2-0.5B")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--prompt", default=r"D:\AI-Voice\音色预设\DS娘_晓伊A3\reference.wav")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    t_proc0 = time.time()  # 进程起点（含 import）
    try:
        import wetext_local_patch
        wetext_local_patch.apply()
    except Exception as e:
        print(f"[warn] wetext 补丁未生效: {e}", file=sys.stderr)

    import torch
    import torchaudio
    from cosyvoice.cli.cosyvoice import AutoModel

    t_import = time.time()
    cosy = AutoModel(model_dir=MODEL)
    t_load = time.time()

    rows = []
    for i in range(a.runs):
        t0 = time.time()
        first = None
        n_chunk = 0
        for j in cosy.inference_cross_lingual(a.text, a.prompt, stream=False):
            n_chunk += 1
            dur = j["tts_speech"].shape[-1] / cosy.sample_rate
            if first is None:
                first = time.time() - t0  # 首块（流式场景下≈可播延迟）
            p = os.path.join(a.out, f"tts_{i}.wav")
            torchaudio.save(p, j["tts_speech"], cosy.sample_rate)
        total = time.time() - t0
        rows.append({
            "run": i,
            "first_chunk_ms": int(first * 1000),
            "total_ms": int(total * 1000),
            "chunks": n_chunk,
            "audio_sec": round(dur, 2),
            "path": p.replace("\\", "/"),
            "warm": i > 0,
        })
        print(f"[run {i}] total={total*1000:.0f}ms first={first*1000:.0f}ms audio={dur:.2f}s",
              file=sys.stderr)

    summary = {
        "python_boot_ms": int((t_import - t_proc0) * 1000),
        "model_load_ms": int((t_load - t_import) * 1000),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "sample_rate": cosy.sample_rate,
        "runs": rows,
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""常驻 TTS 服务（CosyVoice2）——极简 HTTP，stdin/stdout 之外只暴露 POST /say。

跑在 CosyVoice 的 env 里（它有 torch/cosyvoice，但没有 faiss/rmvpe，所以不能兼做 RVC）：
  <cosy-env>/Scripts/python.exe svc_tts.py --port 8788 \
      --src-root <CosyVoice 源码目录> --prompt <参考音色 wav>

请求：POST /say  body={"text": "...", "prompt": "可选参考音频"}
响应：audio/wav 字节流（24 kHz）

设计要点：
  - 模型只在启动时加载一次（冷启动 ~10.6s，之后每句 ~5s）。
  - 合成串行化（一把锁）：CosyVoice 的 session 不是线程安全的，并发请求会互相踩。
  - RK：不做长文本切分（那是调用方的事），>120 字直接 400，逼调用方切。

路径不写死：优先 CLI 参数，其次环境变量 RVCX_COSY_SRC / RVCX_COSY_REF。
本服务跑在 CosyVoice 的 Python 3.10 环境里，**不能** import rvcx.paths（它要 tomllib，3.11+）。
"""
import argparse
import io
import json
import os
import sys
import threading
import time

# 由 main() 填充（CLI > 环境变量）
ROOT = ""
MODEL = ""
DEFAULT_PROMPT = ""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402

LOCK = threading.Lock()
STATE = {"cosy": None, "sr": 24000}


def load():
    try:
        import wetext_local_patch
        wetext_local_patch.apply()
    except Exception as e:
        print(f"[warn] wetext 补丁未生效: {e}", file=sys.stderr)
    import torch
    import torchaudio
    from cosyvoice.cli.cosyvoice import AutoModel
    t0 = time.time()
    STATE["cosy"] = AutoModel(model_dir=MODEL)
    STATE["sr"] = STATE["cosy"].sample_rate
    print(f"[svc_tts] model loaded in {time.time()-t0:.1f}s sr={STATE['sr']}", flush=True)
    # 预热一次，避免第一句请求吃 kernel 编译
    with torch.inference_mode():
        for _ in STATE["cosy"].inference_cross_lingual("预热", DEFAULT_PROMPT, stream=False):
            pass
    print("[svc_tts] warmup done", flush=True)


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):
        print(f"[svc_tts] {fmt % a}", file=sys.stderr, flush=True)

    def _json(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/health"):
            return self._json(200, {"ok": STATE["cosy"] is not None, "sr": STATE["sr"]})
        self._json(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._json(400, {"error": "bad json"})
        if not self.path.startswith("/say"):
            return self._json(404, {"error": "not found"})
        text = (req.get("text") or "").strip()
        if not text:
            return self._json(400, {"error": "text required"})
        if len(text) > 120:
            return self._json(400, {"error": "text too long (>120); split it upstream"})
        prompt = req.get("prompt") or DEFAULT_PROMPT

        import torchaudio  # 延迟导入，import 只做一次
        t0 = time.time()
        with LOCK:
            try:
                buf = io.BytesIO()
                for j in STATE["cosy"].inference_cross_lingual(text, prompt, stream=False):
                    torchaudio.save(buf, j["tts_speech"], STATE["sr"], format="wav")
            except Exception as e:
                return self._json(500, {"error": f"{type(e).__name__}: {e}"})
        data = buf.getvalue()
        print(f"[svc_tts] synth {len(text)} chars in {time.time()-t0:.2f}s, {len(data)} bytes", flush=True)
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8788)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--src-root", default=os.environ.get("RVCX_COSY_SRC", ""),
                    help="CosyVoice 源码目录（含 pretrained_models/）")
    ap.add_argument("--prompt", default=os.environ.get("RVCX_COSY_REF", ""),
                    help="参考音色 wav（zero-shot 用）")
    ap.add_argument("--model-dir", default="", help="模型目录（默认 <src-root>/pretrained_models/CosyVoice2-0.5B）")
    a = ap.parse_args()
    if not a.src_root:
        raise SystemExit("缺 --src-root（或环境变量 RVCX_COSY_SRC）")
    if not a.prompt:
        raise SystemExit("缺 --prompt（或环境变量 RVCX_COSY_REF）")
    ROOT = a.src_root
    MODEL = a.model_dir or os.path.join(ROOT, "pretrained_models", "CosyVoice2-0.5B")
    DEFAULT_PROMPT = a.prompt
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "third_party", "Matcha-TTS"))
    load()
    srv = ThreadingHTTPServer((a.host, a.port), H)
    print(f"[svc_tts] listening on http://{a.host}:{a.port}", flush=True)
    srv.serve_forever()

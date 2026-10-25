# -*- coding: utf-8 -*-
"""OpenAI 兼容 TTS 门面：把「CosyVoice(8788) + RVC(8789)」包装成 /v1/audio/speech。

为什么要它：
  AstrBot 内置 `openai_tts_api` provider（注册名 openai_tts_api）走的正是
  client.audio.speech.create(model=..., voice=..., response_format="wav", input=...)
  → 只要本机有个说这个协议的端点，AstrBot 侧**零代码、零插件**就能用上现有音色。

只依赖标准库，可以跑在任何一套 env（也能跑在 AstrBot 自己的 env 里）：
  python svc_openai_shim.py --port 8790

接口：
  GET  /v1/models            → {"data":[{"id":"fangno120", ...}]}
  POST /v1/audio/speech      → audio/wav 字节
       body: {"model":"fangno120","input":"要念的文本","voice":"fangno120","response_format":"wav"}
  GET  /health               → {"ok":true,"tts":..,"rvc":..}
"""
import argparse
import json
import os
import tempfile
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TTS = "http://127.0.0.1:8788/say"
RVC = "http://127.0.0.1:8789/convert"
VOICE_ID = "fangno120"

# ★ 本机若设了 http_proxy/https_proxy，urllib 会把 127.0.0.1 也送进代理（→ 502 Bad Gateway）。
#   显式建一个不走代理的 opener。
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _post(url, payload, timeout=180):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with _OPENER.open(req, timeout=timeout) as r:
        return r.read()


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):
        print(f"[shim] {fmt % a}", flush=True)

    def _send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            return self._json(200, {"object": "list", "data": [
                {"id": VOICE_ID, "object": "model", "owned_by": "local"},
            ]})
        if self.path.startswith("/health"):
            return self._json(200, {"ok": True, "tts": TTS, "rvc": RVC})
        self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not self.path.startswith("/v1/audio/speech"):
            return self._json(404, {"error": {"message": f"not found: {self.path}"}})
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._json(400, {"error": {"message": "bad json"}})
        text = (req.get("input") or req.get("text") or "").strip()
        if not text:
            return self._json(400, {"error": {"message": "input is required"}})
        t0 = time.time()
        try:
            wav = _post(TTS, {"text": text})
            with tempfile.TemporaryDirectory() as d:
                dry = os.path.join(d, "dry.wav")
                wet = os.path.join(d, "wet.wav")
                with open(dry, "wb") as f:
                    f.write(wav)
                out = json.loads(_post(RVC, {"in": dry, "out": wet}) or b"{}")
                if not out.get("ok"):
                    raise RuntimeError(f"rvc: {out}")
                with open(wet, "rb") as f:
                    data = f.read()
        except Exception as e:
            return self._json(500, {"error": {"message": f"{type(e).__name__}: {e}"}})
        ms = int((time.time() - t0) * 1000)
        print(f"[shim] {len(text)} chars → {len(data)} bytes in {ms}ms", flush=True)
        self._send(200, data, "audio/wav")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--voice-id", default=VOICE_ID)
    a = ap.parse_args()
    VOICE_ID = a.voice_id
    print(f"[shim] listening on http://{a.host}:{a.port}  (api_base = http://{a.host}:{a.port}/v1)", flush=True)
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()

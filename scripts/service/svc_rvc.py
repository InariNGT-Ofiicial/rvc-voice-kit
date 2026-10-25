# -*- coding: utf-8 -*-
"""常驻换声服务（RVC）——模型只加载一次，热调用 ~0.4s/句。

必须跑在 RVC 的 venv 里（它有 faiss + rmvpe；CosyVoice env 没有）：
  <rvc-venv>/Scripts/python.exe svc_rvc.py --port 8789 \
      --model <model.pth> --index <added.index>

请求：POST /convert {"in": "...wav", "out": "...wav", "index_rate": 0.75, "pitch": 0}
响应：{"ok": true, "ms": 387, "out": "..."}

⚠ 实测坑：faiss 打不开含中文的 index 路径 → 启动时把模型/索引按 ASCII 名软链或复制到 mirrors/。

路径不写死：RVCX_REPO / RVCX_FFMPEG_DIR 环境变量优先，其次 rvcx.paths，最后按 cwd 推断。
"""
import argparse
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path


def _resolve_paths() -> tuple[Path, Path | None]:
    """返回 (REPO, FFMPEG_DIR)。环境变量 > rvcx.paths > cwd 推断。"""
    repo_env = os.environ.get("RVCX_REPO", "").strip()
    ff_env = os.environ.get("RVCX_FFMPEG_DIR", "").strip()
    repo = Path(repo_env) if repo_env else None
    ff = Path(ff_env) if ff_env else None
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from rvcx import paths as _p
        repo = repo or _p.REPO
        ff = ff or _p.FFMPEG_DIR
    except Exception:
        if repo is None:
            cand = Path.cwd() / "repo"
            repo = cand if cand.exists() else Path("repo")
    return repo, ff


REPO, FFMPEG_DIR = _resolve_paths()

LOCK = threading.Lock()
STATE = {"vc": None, "cfg": None, "index": "", "sr": 48000}


def ascii_mirror(p: str, name: str) -> str:
    """把含中文的模型/索引复制到 ASCII 目录，绕过 faiss 的 FileIOReader 限制。"""
    src = Path(p)
    if all(ord(c) < 128 for c in str(src)):
        return str(src)
    dst_dir = Path(__file__).parent / "mirrors"
    dst_dir.mkdir(exist_ok=True)
    dst = dst_dir / name
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        print(f"[svc_rvc] mirroring {src.name} -> {dst}", flush=True)
        shutil.copy2(src, dst)
    return str(dst)


def load(model: str, index: str, index_rate: float):
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("index_root", str(REPO / "logs"))
    os.environ.setdefault("outside_index_root", str(REPO / "assets" / "indices"))
    os.environ.setdefault("rmvpe_root", str(REPO / "assets" / "rmvpe"))
    if FFMPEG_DIR and (FFMPEG_DIR / "ffmpeg.exe").exists():
        os.environ["PATH"] = str(FFMPEG_DIR) + os.pathsep + os.environ.get("PATH", "")

    import soundfile as sf  # noqa: F401  (确认环境完整)
    from configs.config import Config
    from infer.vc.modules import VC

    sys.argv = [sys.argv[0]]  # Config() 自己 argparse，必须先清空
    t0 = time.time()
    cfg = Config()
    vc = VC(cfg)
    m = ascii_mirror(model, "model.pth")
    os.environ["weight_root"] = str(Path(m).parent)
    vc.get_vc(Path(m).name)
    STATE["vc"], STATE["cfg"] = vc, cfg
    STATE["index"] = ascii_mirror(index, "added.index") if index else ""
    STATE["sr"] = cfg.sample_rate if hasattr(cfg, "sample_rate") else 48000
    print(f"[svc_rvc] loaded in {time.time()-t0:.1f}s index={STATE['index']}", flush=True)


def convert(inp: str, outp: str, index_rate: float, pitch: int) -> dict:
    import soundfile as sf
    t0 = time.time()
    st, res = STATE["vc"].vc_single(0, inp, pitch, "rmvpe", STATE["index"], index_rate, 0, 1.0, 0.33)
    if not res or res[1] is None:
        raise RuntimeError(st)
    Path(outp).parent.mkdir(parents=True, exist_ok=True)
    sf.write(outp, res[1], res[0])
    return {"ok": True, "ms": int((time.time() - t0) * 1000), "out": outp, "sr": res[0]}


from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):
        print(f"[svc_rvc] {fmt % a}", file=sys.stderr, flush=True)

    def _json(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._json(200 if self.path.startswith("/health") else 404,
                   {"ok": STATE["vc"] is not None, "index": STATE["index"]})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._json(400, {"error": "bad json"})
        try:
            with LOCK:
                r = convert(req["in"], req["out"], float(req.get("index_rate", 0.75)),
                            int(req.get("pitch", 0)))
        except Exception as e:
            return self._json(500, {"error": f"{type(e).__name__}: {e}"})
        self._json(200, r)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8789)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--index", default="")
    a = ap.parse_args()
    load(a.model, a.index, 0.75)
    srv = ThreadingHTTPServer((a.host, a.port), H)
    print(f"[svc_rvc] listening on http://{a.host}:{a.port}", flush=True)
    srv.serve_forever()

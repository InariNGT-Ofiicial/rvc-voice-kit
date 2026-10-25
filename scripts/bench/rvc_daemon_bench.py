# -*- coding: utf-8 -*-
"""RVC 常驻进程延迟基线（热路径）。

做法与 infer/cli.py 完全一致，只是把「建 Config + get_vc(加载模型)」只做一次，
之后循环 vc_single()。这就是「常驻语音服务」里 RVC 该有的形态。

用法（cwd 必须是 RVC repo 根）：
  E:/AI-RVC/venv/Scripts/python.exe rvc_daemon_bench.py \
      --model "D:/AI-Song/03_音色模型_SenrenBanka/芳乃_RVC_步数120/model.pth" \
      --index "D:/.../added_xxx.index" --input a.wav --runs 3
最后一行打印 JSON。
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(r"E:/AI-RVC/repo")
os.chdir(REPO)
sys.path.insert(0, str(REPO))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("weight_root", str(REPO / "assets" / "weights"))
os.environ.setdefault("index_root", str(REPO / "logs"))
os.environ.setdefault("outside_index_root", str(REPO / "assets" / "indices"))
os.environ.setdefault("rmvpe_root", str(REPO / "assets" / "rmvpe"))
# RVC 的 preprocess 写死 cmd=["ffmpeg"]，必须把 ffmpeg 目录塞进 PATH
FFDIR = r"D:/AI-Song/01_人声分离_MSST-GUI/env"
os.environ["PATH"] = FFDIR + os.pathsep + os.environ.get("PATH", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--index", default="")
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--runs", type=int, default=3)
    # ★ 默认 0（不是 0.75）：added 索引在高音帧会匹配到低八度邻居，
    #   令 RVC 照着错音高合成（ir=0 → 769.7 Hz 正确；ir=0.75 → 410.5 Hz 低八度）。
    ap.add_argument("--index-rate", type=float, default=0.0)
    a = ap.parse_args()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)

    t_proc = time.time()
    import torch  # noqa: F401
    import soundfile as sf
    from configs.config import Config
    from infer.vc.modules import VC

    # ★ RVC 的 Config() 自己会 argparse(sys.argv)，必须先清空，否则报 unrecognized arguments
    sys.argv = [sys.argv[0]]

    t_import = time.time()
    config = Config()
    vc = VC(config)
    # 与 infer/cli.py 一致：weight_root 指向 model.pth 所在目录，get_vc 只传文件名
    os.environ["weight_root"] = str(Path(a.model).parent)
    vc.get_vc(Path(a.model).name)
    t_ready = time.time()

    rows = []
    for i in range(a.runs):
        t0 = time.time()
        status, result = vc.vc_single(
            0, a.input, 0, "rmvpe", a.index, a.index_rate, 0, 1.0, 0.33,
        )
        infer_ms = (time.time() - t0) * 1000
        if not result or result[1] is None:
            print(json.dumps({"error": status}, ensure_ascii=False))
            return 1
        out = os.path.join(a.outdir, f"hot_{i}.wav")
        sf.write(out, result[1], result[0])
        rows.append({
            "run": i,
            "infer_ms": int(infer_ms),
            "sr": result[0],
            "audio_sec": round(len(result[1]) / result[0], 2),
            "warm": i > 0,
        })
        print(f"[run {i}] infer={infer_ms:.0f}ms audio={rows[-1]['audio_sec']}s", file=sys.stderr)

    print(json.dumps({
        "python_boot_ms": int((t_import - t_proc) * 1000),
        "model_load_ms": int((t_ready - t_import) * 1000),
        "device": str(getattr(config, "device", "?")),
        "runs": rows,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

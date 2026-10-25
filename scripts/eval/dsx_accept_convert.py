# -*- coding: utf-8 -*-
"""DS娘验收 —— 换声矩阵驱动（把输入按检查点批量转成 DS娘 音色）。

纪律要点：
  * **索引路径必须 ASCII**：faiss 打不开含中文的 index 路径，
    且**不上抛、只打 traceback 后静默按 index_rate=0 跑** ⇒ 表现为「质量莫名下降」。
    所以先把 model.pth / added_*.index 复制到 ASCII 目录再引用。
  * **一条一个文件**：不拼接。输出目录 out/<ckpt>_<rate>/<group>/<name>.wav。
  * 参数固定：f0=rmvpe / pitch=0 / protect=0.33，index_rate 由 --index-rate 指定。
    （MEMORY 判据 9：语料音域窄时 index_rate=0 是安全默认。）
  * **目录模式**：infer.cli 传目录时只加载一次模型，比逐文件快一个量级；
    但 CUDA Graph 私有池会累积 ⇒ 长音频必须分段渲染（见 seg_convert.py）。
    本矩阵每条 ≤ 8 s，池累积可忽略。
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rvcx import paths as P  # noqa: E402

WORK = Path(r"E:/AI-RVC/work/dsx_accept")
STAGE = WORK / "in"
ASCII_VOICES = Path(r"E:/AI-RVC/work/voices_ascii")
WEIGHTS = Path(r"E:/AI-RVC/repo/assets/weights")
INDICES = Path(r"E:/AI-RVC/repo/assets/indices")
INDEX_NAME = "added_IVF1929_Flat_nprobe_1_dsxiaoyi_a3_v1_v2.index"

XVAL = Path(r"E:/AI-RVC/dataset/dsxiaoyi_xval")
PROBE_IN = Path(r"E:/AI-RVC/probe/in")


def stage_inputs():
    """把两路输入拷到 ASCII 暂存目录（目录模式按目录名输出文件名）。"""
    STAGE.mkdir(parents=True, exist_ok=True)
    xd = STAGE / "xval"
    xd.mkdir(exist_ok=True)
    for f in sorted(XVAL.glob("*.wav")):
        dst = xd / f.name
        if not dst.is_file():
            shutil.copy2(f, dst)
    rd = STAGE / "real"
    rd.mkdir(exist_ok=True)
    for f in sorted(PROBE_IN.glob("real_fangno_*.wav")):
        dst = rd / f.name
        if not dst.is_file():
            shutil.copy2(f, dst)
    return {"xval": xd, "real": rd}


def stage_ascii(ckpt: str, weight_glob: str) -> Path:
    """把权重与索引复制到 ASCII 目录（faiss 只认非 ASCII 失败）。"""
    d = ASCII_VOICES / ckpt
    d.mkdir(parents=True, exist_ok=True)
    w = sorted(WEIGHTS.glob(weight_glob))
    if not w:
        raise SystemExit(f"[convert] 找不到权重 {WEIGHTS / weight_glob}")
    shutil.copy2(w[-1], d / "model.pth")
    idx = INDICES / INDEX_NAME
    if idx.is_file():
        shutil.copy2(idx, d / INDEX_NAME)
    return d


def run_batch(vdir: Path, src_dir: Path, dst_dir: Path, index_rate: float,
              protect: float, pitch: int, f0: str):
    cmd = [str(P.PY), "-m", "infer.cli",
           "--model", str(vdir / "model.pth"),
           "--input", str(src_dir), "--output", str(dst_dir),
           "--pitch", str(pitch), "--f0-method", f0,
           "--index-rate", str(index_rate), "--protect", str(protect),
           "--format", "wav", "--overwrite"]
    if index_rate > 0 and (vdir / INDEX_NAME).is_file():
        cmd += ["--index", str(vdir / INDEX_NAME)]
    r = subprocess.run(cmd, cwd=str(P.REPO), env=P.child_env(),
                       capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return r.returncode == 0, (r.stdout or "") + (r.stderr or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", default="e20,e40,e100,e150",
                    help="逗号分隔；每项是 '名:权重glob'")
    ap.add_argument("--rates", default="0", help="逗号分隔的 index_rate")
    ap.add_argument("--protect", type=float, default=0.33)
    ap.add_argument("--pitch", type=int, default=0)
    ap.add_argument("--f0", default="rmvpe")
    ap.add_argument("--groups", default="xval,real")
    a = ap.parse_args()

    specs = []
    for item in a.ckpts.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, glob = item.partition(":")
        specs.append((name, glob or f"dsxiaoyi_a3_v1_{name}_s*.pth"))
    rates = [float(x) for x in a.rates.split(",") if x.strip()]
    groups = [g.strip() for g in a.groups.split(",") if g.strip()]

    WORK.mkdir(parents=True, exist_ok=True)
    src_dirs = stage_inputs()

    for name, glob in specs:
        vdir = stage_ascii(name, glob)
        for rate in rates:
            tag = f"{name}_ir{rate:g}"
            for g in groups:
                outd = WORK / "out" / tag / g
                outd.mkdir(parents=True, exist_ok=True)
                n_have = len(list(outd.glob("*.wav")))
                n_want = len(list(src_dirs[g].glob("*.wav")))
                if n_have >= n_want and n_want:
                    print(f"  跳过 {tag}/{g}（已存在 {n_have}/{n_want}）", flush=True)
                    continue
                t0 = time.time()
                ok, log = run_batch(vdir, src_dirs[g], outd, rate,
                                    a.protect, a.pitch, a.f0)
                n = len(list(outd.glob("*.wav")))
                print(f"  {tag:12s} {g:5s} {'OK ' if ok else 'FAIL'} "
                      f"{n}/{n_want} 条  {time.time()-t0:.1f}s", flush=True)
                if not ok or n < n_want:
                    print(log[-1200:], file=sys.stderr)
    print(f"\n完成。输出根目录 {WORK / 'out'}")


if __name__ == "__main__":
    main()

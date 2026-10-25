# -*- coding: utf-8 -*-
"""DDSP 基线：用同一句干声、同一参考集跑 DDSP，产出与 sweep.py 同口径的指标。

为什么必须重跑：skill 里记的 DDSP 数字（0.239 vs 源 0.468）是另一套语料/参考集，
和本机 fangno 参考集不同口径，直接比就是错的。
用法：python compare_ddsp.py [--step 20000] [--device cuda]
"""
import sys as _sys
from pathlib import Path as _pth

_sys.path.insert(0, str(_pth(__file__).resolve().parents[2]))
from rvcx import paths as _paths  # noqa: E402

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from sweep import load, ref_ltas, metrics, ROOT  # noqa: E402

# ★ 02_模型推理_DDSP-SVC-6.2 只是外层目录，真正的整合包在里面一层
#   按 main_reflow.py 反查最稳，不写死层数
DDSP = next((p.parent for p in Path(_paths.DDSP_ROOT or ".").glob("02_*/**/main_reflow.py")), None)
MODEL_DIR = Path(_paths.DDSP_ROOT or ".") / "03_音色模型_SenrenBanka" / "芳乃_步数20000" / "exp" / "reflow-senren-yoshino"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="20000")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--src", default=str(ROOT / "probe" / "in"))
    ap.add_argument("--refdir", default=str(ROOT / "dataset" / "fangno"))
    ap.add_argument("--json", default=str(ROOT / "probe" / "ddsp_metrics.json"))
    ap.add_argument("--reuse", action="store_true", help="已有输出就跳过推理，只重算指标")
    a = ap.parse_args()

    if DDSP is None:
        raise SystemExit("找不到 DDSP 整合包目录")
    ckpt = MODEL_DIR / f"model_{a.step}.pt"
    if not ckpt.exists():
        raise SystemExit(f"缺模型 {ckpt}")

    srcs = sorted(Path(a.src).glob("*.wav"))
    if not srcs:
        raise SystemExit(f"没有输入 wav: {a.src}")
    out_dir = ROOT / "probe" / "ddsp"
    out_dir.mkdir(parents=True, exist_ok=True)

    ref, selfr, ref_mf = ref_ltas(a.refdir, cache=str(ROOT / "cache" / "ref_ltas.npz"))
    per = []
    for src in srcs:
        out = out_dir / src.name
        if a.reuse and out.exists():
            print(f"[reuse] {out.name}", flush=True)
            per.append(metrics(load(str(src)), load(str(out)), ref, ref_mf))
            continue
        cmd = [str(DDSP / "env" / "python.exe"), "main_reflow.py",
               "-i", str(src), "-m", str(ckpt), "-o", str(out),
               "-id", "1", "-k", "0", "-pe", "rmvpe", "-d", a.device]
        print(" ".join(cmd), flush=True)
        r = subprocess.run(cmd, cwd=str(DDSP), capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode != 0 or not out.exists():
            print((r.stdout or "")[-800:], (r.stderr or "")[-800:], file=sys.stderr)
            raise SystemExit(f"DDSP 推理失败: {src.name}")
        per.append(metrics(load(str(src)), load(str(out)), ref, ref_mf))

    met = {}
    for k in per[0]:
        vals = [p[k] for p in per if p.get(k) is not None]
        met[k] = round(float(np.mean(vals)), 4) if vals else None
    met.update({"model": f"DDSP_{a.step}", "rate": 1.0,
                "ref_self_r": round(float(selfr.mean()), 4), "n_clips": len(per)})

    print(json.dumps(met, ensure_ascii=False, indent=2))
    Path(a.json).write_text(json.dumps(met, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {a.json}")


if __name__ == "__main__":
    main()

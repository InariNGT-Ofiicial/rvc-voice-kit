# -*- coding: utf-8 -*-
"""台词换声的量化：音色位移（Δr）+ 时长 + F0 + 削波/归零。

★ 台词场景和唱歌不同：只测「像不像」不够，还得测「字有没有糊」——那部分在 dialogue_cer.py
  （要用 faster-whisper，跑在另一个解释器里）。
"""
import sys as _sys
from pathlib import Path as _pth

_sys.path.insert(0, str(_pth(__file__).resolve().parents[2]))
from rvcx import paths as _paths  # noqa: E402

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from sweep import load, metrics, ref_ltas  # noqa: E402

DLG = _paths.WORK / "dialogue"
REFDIR = str(_paths.REF or (_paths.DATASET / "fangno"))
CACHE = str(_paths.WORK / "cache" / "ref_ltas.npz")


def dur(p):
    i = sf.info(str(p))
    return len(sf.read(str(p), dtype="float32", always_2d=False)[0]) / i.samplerate


def main():
    ref, selfr, ref_mf = ref_ltas(REFDIR, cache=CACHE)
    print(f"参考集自测上限 r = {selfr.mean():+.4f}")

    rows = []
    arms = {"T": "src_tts", "H": "src_real"}
    for arm, d in arms.items():
        for src in sorted((DLG / d).glob("*.wav")):
            lid = src.stem
            sy = load(str(src))
            base = metrics(sy, sy, ref, ref_mf)
            rows.append(dict(arm=arm, lid=lid, model="src", r_lta=base["r_lta"],
                             dr=0.0, dur=round(dur(src), 2), dur_ratio=1.0,
                             f0_diff_hz=None, peak=round(float(np.max(np.abs(sy))), 3),
                             zero=round(float(np.mean(np.abs(sy) <= 0.0)), 5)))
            for mname in ("e120", "e200"):
                out = DLG / "out" / f"{lid}_{arm}_{mname}.wav"
                if not out.exists():
                    print(f"!! 缺 {out.name}", file=sys.stderr)
                    continue
                oy = load(str(out))
                met = metrics(sy, oy, ref, ref_mf)
                rows.append(dict(arm=arm, lid=lid, model=mname,
                                 r_lta=met["r_lta"], dr=met["dr"],
                                 dur=round(dur(out), 2),
                                 dur_ratio=round(dur(out) / dur(src), 3),
                                 f0_diff_hz=met["f0_diff_hz"], peak=met["peak"],
                                 zero=met["zero_ratio"]))
                print(f"{lid} {arm} {mname}: r={met['r_lta']:+.4f} Δr={met['dr']:+.4f} "
                      f"dur×{dur(out)/dur(src):.3f} f0={met['f0_diff_hz']} "
                      f"zero={met['zero_ratio']}", flush=True)

    (DLG / "metrics.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                      encoding="utf-8")

    # 汇总
    print("\n=== 汇总（4 句均值）===")
    for arm in arms:
        for mname in ("src", "e120", "e200"):
            sel = [r for r in rows if r["arm"] == arm and r["model"] == mname]
            if not sel:
                continue
            r = np.mean([x["r_lta"] for x in sel])
            dr = np.mean([x["dr"] for x in sel])
            zr = np.mean([x["zero"] for x in sel])
            print(f"{arm} {mname:5s} n={len(sel)} r_lta={r:+.4f} Δr={dr:+.4f} zero={zr:.4f}")


if __name__ == "__main__":
    main()

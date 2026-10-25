# -*- coding: utf-8 -*-
"""台词可懂度：把每条输出再喂回 ASR，与原文算字错率 CER。

★ 为什么台词必须加这条判据：LTAS/Δr 只测音色。台词交付的第一要求是「听得清」，
  换声把字糊掉，音色再像也没用。唱歌可以靠乐句兜住，说话不能。

注意：这是**回读 CER**，包含 whisper 自身的识别误差，所以源音频的 CER 不是 0，
      **只看相对变化**（输出 CER − 源 CER），不要当绝对可懂度。
"""
import sys as _sys
from pathlib import Path as _pth

_sys.path.insert(0, str(_pth(__file__).resolve().parents[2]))
from rvcx import paths as _paths  # noqa: E402

import json
import os
import re
import sys
from pathlib import Path

DLG = _paths.WORK / "dialogue"

# 与 dialogue_tts_ja.py 完全一致的台词文本（真人说的是同一句，只是 ASR 断句带标点）
TEXT = {
    "ja01": "私の質問だけに答えてください",
    "ja02": "顔見ずに、そのまま聞いてください",
    "ja03": "ああ、申し遅れました。私は日立真子と申します。",
    "ja04": "朝はまず、お風呂場の掃除をしているのですが、私、お風呂が好きなんです。",
}

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def norm(s: str) -> str:
    """只留假名/汉字/字母数字，去掉标点与空白——ASR 标点不可靠，比标点没意义。"""
    return re.sub(r"[^0-9A-Za-z\u3040-\u30ff\u4e00-\u9fff]", "", s)


def cer(ref: str, hyp: str) -> float:
    """字级编辑距离 / 参考长度。"""
    r, h = norm(ref), norm(hyp)
    if not r:
        return 0.0
    dp = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(h) + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return dp[len(h)] / len(r)


def main():
    from faster_whisper import WhisperModel

    files = []
    for arm, d in (("T", "src_tts"), ("H", "src_real")):
        files += [(arm, "src", p) for p in sorted((DLG / d).glob("*.wav"))]
    for arm in ("T", "H"):
        files += [(arm, "e120", p) for p in sorted((DLG / "out").glob(f"*_{arm}_e120.wav"))]
        files += [(arm, "e200", p) for p in sorted((DLG / "out").glob(f"*_{arm}_e200.wav"))]

    print(f"待转写 {len(files)} 条", flush=True)
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")
    rows = []
    for arm, mdl, p in files:
        lid = p.stem.split("_")[0]
        segs, _ = model.transcribe(str(p), language="ja", beam_size=5, vad_filter=True)
        hyp = "".join(s.text for s in segs).strip()
        c = cer(TEXT[lid], hyp)
        rows.append(dict(arm=arm, lid=lid, model=mdl, cer=round(c, 4), hyp=hyp))
        print(f"{lid} {arm} {mdl:5s} CER={c:.3f}  {hyp}", flush=True)

    (DLG / "cer.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
    print("\n=== CER 均值 ===")
    import statistics as st
    for arm in ("T", "H"):
        for m in ("src", "e120", "e200"):
            v = [r["cer"] for r in rows if r["arm"] == arm and r["model"] == m]
            if v:
                print(f"{arm} {m:5s} n={len(v)} CER均值={st.mean(v):.4f}")


if __name__ == "__main__":
    main()

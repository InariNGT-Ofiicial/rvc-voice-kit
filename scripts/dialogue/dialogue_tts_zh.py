# -*- coding: utf-8 -*-
"""中文台词源：用本机 CosyVoice2-0.5B（晓伊 A3 参考音色）cross_lingual 合成中文台词。

为什么换 TTS 引擎：日语那批用 GPT-SoVITS，但本机所有 GPT-SoVITS 音色
（kurumi / yoshino）都是日语数据训的，喂中文会先自己念错，测不出 RVC 的能力。
CosyVoice2 原生中文（zh-CN），且零样本克隆，能给出「干净、字字清楚」的中文源。

★ 与日语批保持可比：同样的「另一个角色的音色 + 同一批语义的台词」，
  变量只有语言。任务是问：**日语训练的芳乃 RVC，换中文会不会掉字、会不会把声调抹平。**

用 cosy env 跑：
  <cosy-env>/Scripts/python.exe dialogue_tts_zh.py
"""
import sys as _sys
from pathlib import Path as _pth

_sys.path.insert(0, str(_pth(__file__).resolve().parents[2]))
from rvcx import paths as _paths  # noqa: E402

import json
import os
import sys
import time
from pathlib import Path

COSY = _paths.COSY
SRC = COSY / "CosyVoice_源码"
MODEL = SRC / "pretrained_models" / "CosyVoice2-0.5B"
REF = _paths.COSY_REF
OUT = _paths.WORK / "dialogue_zh" / "src_tts"

# ★ 坑①（已踩）：instruct2 的指令若缺 "You are a helpful assistant. " 前缀 +
#   "<|endofprompt|>"，模型会把**指令本身念出来**（实测 zh01 回读 =
#   "请只回答我的问题，语速适中，句间自然停顿"，CER 1.25）。
# ★ 坑②（已踩）：即使补上前缀，长句结尾仍会漏念指令尾巴（zh04 尾句多出
#   "语速适中"）。⇒ 本批**不用 instruct2**，改 cross_lingual：只取音色、
#   无指令文本、不需要参考文本，彻底没有可泄漏的东西。
SPEED = 1.0
SEED = 20261025
GAP = 0.12          # 段间留白（s），防止分句合成后两句粘在一起

# 前 4 条是日语那 4 条的对应中译（语义对齐，便于跨语言比较）；
# 后 2 条是中文特有难点：语气词/数字/反问，以及多分句长句（连续齿音 sh/zh/ch）。
LINES = [
    dict(lid="zh01", text="请只回答我的问题。", note="对应 ja01，短陈述"),
    dict(lid="zh02", text="别看我的脸，就这样听我说。", note="对应 ja02，祈使+否定"),
    dict(lid="zh03", text="啊，失礼了。我叫日立真子。", note="对应 ja03，双句+语气词"),
    dict(lid="zh04", text="早上我首先会打扫浴室，因为我挺喜欢泡澡的。", note="对应 ja04，长宾语从句"),
    dict(lid="zh05", text="真是的，都说了三遍了，你怎么还不明白啊？", note="语气词+数字+反问，中文特有"),
    dict(lid="zh06", text="那个人明明什么都知道，却什么都不肯说，就一直笑着。", note="多分句长句，连续齿音"),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    os.chdir(COSY)                       # wetext_local_patch 要在当前目录可见
    sys.path.insert(0, str(COSY))
    sys.path.insert(0, str(SRC))
    sys.path.insert(0, str(SRC / "third_party" / "Matcha-TTS"))

    try:
        import wetext_local_patch
        wetext_local_patch.apply(verbose=False)
        print("[i] wetext 本地补丁已生效（数字/符号念对）", flush=True)
    except Exception as e:
        print(f"[warn] wetext 补丁未生效: {type(e).__name__}: {e}", flush=True)

    import numpy as np
    import soundfile as sf
    import torch
    from cosyvoice.cli.cosyvoice import AutoModel

    torch.manual_seed(SEED)
    t0 = time.time()
    cosy = AutoModel(model_dir=str(MODEL))
    print(f"[i] 模型就绪 {time.time()-t0:.1f}s  sr={cosy.sample_rate}  "
          f"ref={REF.name}", flush=True)

    rows = []
    for ln in LINES:
        t1 = time.time()
        parts = []
        for j in cosy.inference_cross_lingual(ln["text"], str(REF),
                                              stream=False, speed=SPEED):
            parts.append(j["tts_speech"].squeeze(0).cpu().numpy())
        a = np.concatenate(parts) if len(parts) == 1 else np.concatenate(
            sum([[p, np.zeros(int(GAP * cosy.sample_rate), np.float32)]
                 for p in parts[:-1]], []) + [parts[-1]])
        a = a.astype(np.float32)

        # ★ 教训（日语批踩过）：TTS 输出峰值可能 >1，直接写 PCM_16 会硬削顶，
        #   削顶的谐波会污染 RVC 的 HuBERT 特征 ⇒ 换声直接崩。先归一化再写。
        pk = float(np.max(np.abs(a))) or 1.0
        gain = 1.0
        if pk > 0.95:
            gain = 0.9 / pk
            a = a * gain
        sr = cosy.sample_rate
        p = OUT / f"{ln['lid']}.wav"
        sf.write(str(p), a, sr, subtype="PCM_16")

        rms = float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))
        rows.append(dict(lid=ln["lid"], text=ln["text"], note=ln["note"],
                         path=str(p), sr=sr, dur=round(len(a) / sr, 2),
                         segs=len(parts), peak=round(pk, 3),
                         gain=round(gain, 4), rms=round(rms, 4)))
        print(f"{ln['lid']} {len(a)/sr:5.2f}s segs={len(parts)} pk={pk:.3f} "
              f"gain={gain:.3f} rms={rms:.3f}  {ln['text']}", flush=True)

    (OUT.parent / "tts_manifest.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("done", flush=True)


if __name__ == "__main__":
    main()
